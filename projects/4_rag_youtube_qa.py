from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled


from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda


load_dotenv()


def load_transcripts(video_id) -> str:
    try:
        # If you don’t care which language, this returns the “best” one
        transcript_list = YouTubeTranscriptApi.get_transcript(
            video_id, languages=["en"])

        # Flatten it to plain text
        transcript = " ".join(chunk["text"] for chunk in transcript_list)

        return transcript

    except TranscriptsDisabled:
        print("No captions available for this video.")
        return ''


def fetch_data(video_id: str):
    # fetch data from Youtube (Document Loader)
    transcript = load_transcripts(video_id)

    # split the transcript in small chunks (Text Splitter)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents(
        texts=[transcript], metadatas=[{'video_id': video_id}])

    # embedding (generate and store embeddings)
    embedding = OpenAIEmbeddings(model='text-embedding-3-small')
    vector_store = FAISS.from_documents(chunks, embedding)

    return vector_store


vector_store = fetch_data('zOgRmw1atFU')


retriever = vector_store.as_retriever(
    search_type='mmr', search_kwargs={'k': 4})


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      Context: {context}

      Question: {question}
      Answer:""",
    input_variables=['context', 'question']
)


parser = StrOutputParser()


def format_docs(retrieved_docs: list[Document]) -> str:
    return ''.join(doc.page_content for doc in retrieved_docs)


# Create Chains
parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

chain = prompt | llm | parser

final_chain = parallel_chain | chain

question = 'is macbook good for gaming?'
ans = final_chain.invoke(question)

print(ans)
