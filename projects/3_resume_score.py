from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()
model = ChatOpenAI(model='gpt-4o-mini')

RESUME = 'rag/loader/resume.pdf'
URL = ''

pdf_loader = PyPDFLoader(RESUME)
docs = pdf_loader.lazy_load()

url = input('URL: ')
if not url:
    url = URL

print(f'fetching {url}')
web_loader = WebBaseLoader(url)
jobs = web_loader.lazy_load()

lst = zip(docs, jobs)

prompt = PromptTemplate(
    input_variables=['resume', 'job_desc'],
    template="Rate this candidate from 1-10 for the following Job role. Do not"
    "respond with anything else than the score and reason for score in 1 "
    "if core competencies are missing, it must reflect in score."
    "sentence.\nexample_output: \n Score: 8/10 Reason: The candidate is "
    "aligned with all the technologies."
    "\nResume: {resume}" + "\nJob Role:{job_desc}" + "\nScore:"
)

parser = StrOutputParser()

chain = prompt | model | parser

for (doc, job_desc) in lst:
    dct = {'resume': doc.page_content, 'job_desc': job_desc.page_content}
    # print(dct)
    print(chain.invoke(dct))
