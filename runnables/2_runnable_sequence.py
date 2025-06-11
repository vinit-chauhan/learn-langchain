from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence

load_dotenv()

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

model = ChatOpenAI()

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

# chain Runnables using Runnable Sequence
chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

# chain using LCEL (LangChain Expression Language)
# chain = prompt1 | model | parser | prompt2 | model | parser

print(chain.invoke({'topic': 'AI'}))
