import os
import csv
import ast
from requests import get, Response
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import VectorStore
from langchain_chroma import Chroma

load_dotenv()


url = 'https://raw.githubusercontent.com/elastic/ecs/refs/tags/v9.0.0/generated/csv/fields.csv'


def fetch():
    resp: Response = get(url)

    rows = resp.text.splitlines()
    res = list(csv.DictReader(rows))

    return res


def load_to_vec_store(vec_store: VectorStore) -> list[str]:
    ecs_fields = fetch()
    print(f'found {len(ecs_fields)} fields')

    docs = [Document(str(field)) for field in ecs_fields]

    res = vec_store.add_documents(docs)
    print(f'added {len(res)} embeddings to vector_db')

    print(res[0])
    return res


def similarity_search(inputs):

    semantic_search_prompt = PromptTemplate(
        input_variables=['field_name', 'field_desc'],
        template="""Which fields are similar to field_name: {field_name},
        field_desc: {field_desc}?"""
    )

    relevant_docs = vec_store.similarity_search(
        semantic_search_prompt.format(**inputs)
    )

    relevant_ecs_fields = ''
    for doc in relevant_docs:
        field = ast.literal_eval(doc.page_content)
        relevant_ecs_fields += field['Field']+', '

    return relevant_ecs_fields


llm = ChatOpenAI(model='gpt-4o-mini')
prompt = PromptTemplate(
    input_variables=['field_name', 'field_desc', 'similar_fields'],
    template="""You are an expert field mapper.
    Map the given field to its corresponding ECS mappings based on field's
    name and description. Only choose one option from Similar Fields.
    If there's no match return "No Match".

    Example:
    User: server_ip: IP address of the application server
    Similar Fields: server.ip
    Answer: server.ip

    User: {field_name}: {field_desc}
    Similar Fields: {similar_fields}
    Answer:""")
vec_store = Chroma('ecs_collection', OpenAIEmbeddings(), './chroma_db')
parser = StrOutputParser()

if os.environ.get('INGEST_DOCS', 0) == '1':
    print('loading ecs fields to vector store')
    res = load_to_vec_store(vec_store)


chain = prompt | llm | parser


parallel_chain = RunnableParallel({
    'field_name': RunnableLambda(lambda x: x['field_name']),
    'field_desc': RunnableLambda(lambda x: x['field_desc']),
    'similar_fields': RunnableLambda(similarity_search)
})


tests = [
    ("request_id", "Unique identifier assigned to this specific request or transaction", "event.id"),
    ("user_id", "Internal ID used to represent the user who triggered this event", "user.id"),
    ("username", "name or email of the user", "user.name"),
    ("src_ip", "IP address where the request originated from", "source.ip"),
    ("dst_ip", "Destination IP address the request was sent to", "destination.ip"),
    ("user_agent", "Full string representing the client browser or application",
     "user_agent.original"),
    ("method", "HTTP verb used in the request (e.g., GET, POST)", "http.request.method"),
    ("status_code", "Numeric status code returned in response to the request",
     "http.response.status_code"),
    ("timestamp", "ISO timestamp when the event occurred", "@timestamp"),
    ("path", "URL route or path that was accessed", "url.path"),
    ("response_time_ms",
     "Time taken to process the request, in milliseconds", "event.duration"),
    ("error_msg", "Textual message describing an error encountered", "error.message"),
    ("client_ip", "IP address of the client making the connection", "client.ip"),
    ("email", "Email address associated with the user or account", "user.email"),
    ("event_type", "High-level category describing what kind of event this is", "event.type"),
    ("geo_location", "ISO code for Country Name", "source.geo.country_iso_code"),
    ("device", "Type of device or OS used to generate this event", "host.os.name"),
    ("session_id", "Identifier for a user's session during their interaction", "session.id"),
    ("referrer", "Page or URL that linked to the current request", "http.request.referrer"),
    ("latency", "Measured delay or time taken to respond to a request", "event.duration"),
    ("auth_method", "Authentication mechanism used by the user (e.g., password, token)",
     "user.authentication_method")
]
test_pass = test_fail = 0
for test in tests:

    args = parallel_chain.invoke({'field_name': test[0],
                                 'field_desc': test[1]})
    res = chain.invoke(args)

    if res == test[2]:
        test_pass += 1
        print('pass:', res, '->', args['similar_fields'])
    else:
        print('fail:', test[2], '!=', res, '->',
              test[0], args['similar_fields'],)

print(f'result: {test_pass}/{len(tests)}')
