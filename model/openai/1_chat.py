from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


load_dotenv()

model = ChatOpenAI(model='gpt-4o-mini')

prompt = PromptTemplate(
    input_variables=['topic'],
    template="""You are an expert professor teller,
 Give me a brief introduction of the following topic. Make sure to include the
 background, use-case and example for the topic. Background should be 1-2
 sentences, 2-3 sentence for use-case and appropriate length for example.

 Topic: {topic}
 Answer:"""
)


chain = prompt | model

topic = input("Enter any topic: \nUser: ")

response = chain.invoke({'topic': topic})

print(response.content)
