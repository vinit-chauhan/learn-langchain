from langchain_core.prompts import ChatPromptTemplate

history = []

messages = ChatPromptTemplate([
    ('system', "You are a helpful {domain} expert."),
    ('human', """Explain the following topic in 1 sentence.
Topic: {topic}""")
])


prompt = messages.invoke({'domain': 'golang', 'topic': 'goroutines'})
history.append(*prompt)

prompt = messages.invoke({'domain': 'python', 'topic': '*list'})
history.append(*prompt)

print(history)
