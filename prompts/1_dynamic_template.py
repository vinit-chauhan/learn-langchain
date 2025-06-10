from constants import model_name

from langchain_openai import ChatOpenAI
from langchain.prompts import load_prompt

model = ChatOpenAI(model=model_name)


template = load_prompt('prompts/template.json')


chain = template | model


result = chain.invoke({
    'programming_language': 'golang',
    'topic': 'garbage collection',
    'length': '6-7 sentences',
})


print(result.content)
