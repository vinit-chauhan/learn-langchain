from typing import Tuple, List
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate


def fetch_history() -> List[Tuple[str, str]]:
    return [
        ('human', 'I want refund for my order #1235.'),
        ('assistant', 'you will get your refund for order #1235 in 2 to 3\
          business days.')
    ]


# Create template
chat_prompt = ChatPromptTemplate([
    ('system', 'You are a helpful assistant.'),
    MessagesPlaceholder('chat_history'),
    ('human', '{query}')
])

# load history from database
chat_history = fetch_history()

# hydrate prompt
prompt = chat_prompt.invoke({
    'chat_history': chat_history,
    'query': 'when will I get the refund?'
})


print(prompt)
