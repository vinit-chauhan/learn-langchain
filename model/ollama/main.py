from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
# from langchain.callbacks.streaming_stdout import
# StreamingStdOutCallbackHandler
# from langchain.callbacks.manager import CallbackManager


llm = OllamaLLM(model='gemma3')
# llm = OllamaLLM(
#     model='gemma3',
#     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

topic_prompt = PromptTemplate(
    input_variables=['topic'],
    template="""
    Give concise information on the following topic in 3 sentences.
    Topic: {topic}
    Answer:"""
)

explain_ez_prompt = PromptTemplate(
    input_variables=['explanation'],
    template="""Given the following topic description,
    rewrite the only the explanation like I'm 5 years old.
    Explanation: {explanation}
    Answer:"""
)
# Create chains for topic description and simplified explanation
topic_chain = topic_prompt | llm
exp_ez_chain = explain_ez_prompt | llm

# Generate a concise description for the topic
print('Generating topic description...')
topic_response = topic_chain.invoke({'topic': 'LLMs'})
print('Topic description generated.')

# Simplify the explanation for a younger audience
print('Simplifying explanation...')
simplified_response = exp_ez_chain.invoke({'explanation': topic_response})
print('Simplified explanation ready.')

# Output the results
print(simplified_response)
