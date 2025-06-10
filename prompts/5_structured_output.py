from constants import model_name

from typing import TypedDict, Annotated, Literal
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model=model_name)


# TypedDict: Can modal the response in the given format. Use Annotated for
# additional context of the field. You can't validate the response type here.
# You might get different type in response than one mentioned in the model.
class Review(TypedDict):
    summary: Annotated[str, 'A brief description of the review']
    sentiment: Annotated[
        Literal["Positive", "Negative", "Neutral"],
        'Returns one of the following the sentiment value for the review. '
        'only possible values: "Positive", "Negative", "Neutral"']


structured_model = model.with_structured_output(Review)

res = structured_model.invoke("""The product overall is bad. \
                              But it's the best given the price point.""")


print(res)
