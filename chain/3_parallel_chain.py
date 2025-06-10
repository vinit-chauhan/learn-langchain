from constants import model

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel


model2 = ChatOpenAI(model='gpt-4o-mini')

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n'
    '{text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n'
    '{text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n'
    'notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model | parser

chain = parallel_chain | merge_chain

text = """
In a distributed computer system, you can only support two of the following \
guarantees:

Consistency - Every read receives the most recent write or an error
Availability - Every request receives a response, without guarantee that it \
contains the most recent version of the information
Partition Tolerance - The system continues to operate despite arbitrary \
partitioning due to network failures
Networks aren't reliable, so you'll need to support partition tolerance. \
You'll need to make a software tradeoff between consistency and availability.

CP - consistency and partition tolerance
Waiting for a response from the partitioned node might result in a timeout \
error. CP is a good choice if your business needs require atomic reads and \
writes.

AP - availability and partition tolerance
Responses return the most readily available version of the data available on \
any node, which might not be the latest. Writes might take some time to \
propagate when the partition is resolved.

AP is a good choice if the business needs to allow for eventual consistency \
or when the system needs to continue working despite external errors.
"""

result = chain.invoke({'text': text})

print(result)

chain.get_graph().print_ascii()
