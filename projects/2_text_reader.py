from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.retrieval_qa.base import RetrievalQA


load_dotenv()

llm = OpenAI(model='gpt-4o-mini', temperature=0.4)
parser = StrOutputParser()

# load docs
loader = TextLoader('docs.txt')
docs = loader.load()

# split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents=docs)

# convert docs to embeddings and stores in FAISS
vec_store = FAISS.from_documents(
    docs, OpenAIEmbeddings(model='text-embedding-3-small'))

# retriever fetches relevant documents
retriever = vec_store.as_retriever()


# user query from docs
query = "what are the key take away from the documents?"

# manual retrieval and query
result_docs = retriever.invoke(query)

# combine the result
retrieved_text = [''.join([doc.page_content for doc in result_docs])]


prompt = PromptTemplate(
    input_variables=['query', 'retrieved_text'],
    template=f"Based on the following text, answer the question: \n{query}" +
    f"\n\n{retrieved_text}"
)


chain = prompt | llm | parser

print('answer:', chain.invoke(
    {'query': query, 'retrieved_text': retrieved_text}))


# using RetrievalQA
chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

ans = chain.run(query)

print("RetrievalQA:", ans)
