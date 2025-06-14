from dotenv import load_dotenv
from typing import Type

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, StructuredTool, BaseTool, ArgsSchema
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, ToolMessage, BaseMessage


load_dotenv()


@tool
def multiply(num1: int, num2: int) -> int:
    """Multiply two numbers num1 and num2 and returns the multiplication."""

    return num1 * num2


# Addition tool (BaseModel + StructuredTool)
class AddInput(BaseModel):
    a: int = Field(description='First number to add')
    b: int = Field(description='Second number to add')


def add_func(a: int, b: int) -> int:
    return a + b


add = StructuredTool(
    name='add',
    func=add_func,
    args_schema=AddInput,
    description='Adds two numbers'
)

# Division tool (BaseModel + BaseTool)


class DivisionInput(BaseModel):
    a: int = Field(description='dividend')
    b: int = Field(description='divisor')


class DivisionTool(BaseTool):
    name: str = 'division'
    description: str = 'divide two numbers'
    args_schema: ArgsSchema = DivisionInput

    def _run(self, a: int, b: int) -> float:
        if b == 0:
            raise ValueError('Err: Division By Zero not allowed')

        return a/b


division = DivisionTool()

llm = ChatOpenAI(model='gpt-4o-mini')
parser = StrOutputParser()
llm_with_tools = llm.bind_tools([multiply, add, division])


query = HumanMessage('What is 3 divide by 0')
messages: list[BaseMessage] = [query]


chain = llm_with_tools
res = chain.invoke(messages)
messages.append(res)

print(res)
use_tool = res.tool_calls[0]
tool_msg: ToolMessage = globals()[use_tool['name']].invoke(use_tool)
messages.append(tool_msg)


res = chain.invoke(messages)

print(res.content)
