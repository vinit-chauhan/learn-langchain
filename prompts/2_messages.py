from constants import model_name

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

model = ChatOpenAI(model=model_name)

messages = [
    SystemMessage('You are a helpful assistant'),
    HumanMessage('Tell me about langchain')
]


res = model.invoke(messages)

messages.append(AIMessage(res.content))

print(messages)
