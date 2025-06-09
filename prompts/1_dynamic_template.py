from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import load_prompt


load_dotenv()

model = ChatOpenAI(model='gpt-4o-mini')

template = load_prompt('prompts/template.json')


chain = template | model


result = chain.invoke({
    'programming_language': 'golang',
    'topic': 'garbage collection',
    'length': '6-7 sentences',
})


print(result.content)
