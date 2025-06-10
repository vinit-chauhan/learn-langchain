from constants import model_name

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

model = ChatOpenAI(model=model_name)


template = PromptTemplate(
    input_variables=['user_input', 'history'],
    template="""You are a assistant who answers the question, and make sure
    you can refer to the chat history.

    History: {history}
    User: {user_input}
    Assistant:"""
)

chat_history: list[SystemMessage | HumanMessage | AIMessage] = [
    SystemMessage(
        'You are a assistant who answers the question, and make sure \
            you can refer to the chat history.'),
]

while True:
    user_input = input('User: ')
    if user_input == 'exit':
        print('Thank you!')
        break

    chat_history.append(HumanMessage(user_input))

    result = model.invoke(chat_history)

    chat_history.append(AIMessage(result.content))

    print(f'Assistant: {result.content}')
    print()
