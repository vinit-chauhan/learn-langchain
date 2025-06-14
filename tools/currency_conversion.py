import requests
import json
from typing import Annotated
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolCall
from langchain_core.tools import tool, InjectedToolArg

API_URL = 'https://v6.exchangerate-api.com/v6/7a3b7321d377c0966e3b4a47'

load_dotenv()


@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """
    This function fetches the currency conversion factor between a given base currency and a target currency
    """
    url = f'{API_URL}/pair/{base_currency}/{target_currency}'

    response = requests.get(url)

    return response.json()


@tool
def convert(base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """
    given a currency conversion rate this function calculates the target currency value from a given base currency value
    """

    return base_currency_value * conversion_rate


query = HumanMessage(
    'What is the conversion factor between USD and INR, and based on that can you convert 10 inr to usd')
messages = [query]

llm = ChatOpenAI()

llm_with_tools = llm.bind_tools([convert, get_conversion_factor])

ai_message: AIMessage = llm_with_tools.invoke(messages)

messages.append(ai_message)


for tool_call in ai_message.tool_calls:
    if tool_call['name'] == 'get_conversion_factor':
        tool_msg1 = get_conversion_factor.invoke(tool_call)
        conversion_rate = json.loads(tool_msg1.content)['conversion_rate']
        print(conversion_rate)
        messages.append(tool_msg1)
    if tool_call['name'] == 'convert':
        tool_call['args']['conversion_rate'] = conversion_rate
        tool_msg2 = convert.invoke(tool_call)
        messages.append(tool_msg2)

res = llm_with_tools.invoke(messages)
print(res)
