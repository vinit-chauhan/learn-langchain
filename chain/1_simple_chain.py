from constants import model

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)


parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'topic': 'cricket'})

print(result)

chain.get_graph().print_ascii()
