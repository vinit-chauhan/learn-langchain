from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()
model = ChatOpenAI(model='gpt-4o-mini')

loader = TextLoader('rag/loader/cricket.txt', encoding='utf-8')
docs = loader.load()

prompt = PromptTemplate(
    input_variables=['text'],
    template="Summarize the following doc: {text}"
)

parser = StrOutputParser()

chain = prompt | model | parser


print(chain.invoke({'text': docs[0].page_content}))
