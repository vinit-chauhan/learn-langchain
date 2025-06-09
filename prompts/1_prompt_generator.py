from langchain_core.prompts import PromptTemplate


template = PromptTemplate(
    input_variables=['programming_language', 'topic', 'length'],
    template="""You are an expert Software Engineer in {programming_language}
    programming language. Your goal is to explain the following topic to the
    user in most technical way. You should give a technical deep dive for the
    topic and its use case.

    Topic: {topic} in {programming_language}
    Length: {length}
    Explanation:"""
)

# Saves the template to json.
template.save('prompts/template.json')
