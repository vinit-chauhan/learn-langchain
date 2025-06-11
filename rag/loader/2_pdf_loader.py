from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()
model = ChatOpenAI(model='gpt-4o-mini')

loader = PyPDFLoader('rag/loader/resume.pdf')
docs = loader.lazy_load()

# print(docs[0].page_content)

prompt = PromptTemplate(
    input_variables=['resume', 'job_desc'],
    template="Rate this candidate from 1-10 for the following Job role. Do "
    "not respond with anything else than the score."
    "\nexample_output: 8/10"
    "\nResume: {resume}"
    "\nJob Role:{job_desc}"
)

parser = StrOutputParser()

chain = prompt | model | parser

job_desc = """What You Will Be Doing
Working on green-field projects that evolve the developer experience in
 building and maintaining Security Integrations Design and build tooling
 and automation enabled by Gen AI Cultivate an atmosphere of mutual respect,
 collaboration and consensus-based decision-making Collaborate in the open
 with the Security and Ingest teams, Elastic users, and others supporting open
 source projects. What You Will Bring Experience with AI language models with a
 strong understanding of prompt engineering Knowledge of Python, Go and
 Javascript/Typescript in the context of developer tooling and data collection
 Knowledge of performance engineering and benchmarking
 A high level of autonomy. The distributed nature of the company offers freedom
 to work when and how one sees fit, but also requires team members to work
 independently, and reach out for help when stuck.
"""


for doc in docs:
    print(chain.invoke({'resume': doc.page_content, 'job_desc': job_desc}))
