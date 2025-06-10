from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    input_variables=['topic'],
    template="""Write 5 paragraph summary for the following topic.\
    The summary should contain brief background, technical details,\
    history, and new advancements.\n{topic}"""
)


model = ChatOpenAI(model='gpt-4o-mini')

parser = StrOutputParser()

chain = prompt | model | parser

with open('docs.txt', 'w') as f:
    res = chain.invoke({'topic': 'golang'})

    f.write(res)
