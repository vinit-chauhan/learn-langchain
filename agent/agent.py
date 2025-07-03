import os
import datetime
import requests

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.tools import tool, BaseTool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun
from langchain import hub

from pydantic import BaseModel, Field


load_dotenv()

WEATHER_API_KEY = os.environ['WEATHER_API_KEY']

prompt = hub.pull('hwchase17/react')

llm = ChatOpenAI(model='gpt-4o-mini')

search_tool = DuckDuckGoSearchRun()


class OddOrEvenInput(BaseModel):
    num: str = Field(description="check this number if it's even or odd")


@tool("odd_or_even", args_schema=OddOrEvenInput, )
def odd_or_even(num: str) -> str:
    """Returns if the given input number is 'odd' or 'even'."""

    return "Invalid number" if not num.isnumeric() else (
        "Even" if int(num) % 2 == 0 else "Odd")


class CurrentDateTimeInput(BaseModel):
    pass


@tool("current_date_time", args_schema=CurrentDateTimeInput)
def current_date_time() -> str:
    """Returns the current date and time in string"""
    return datetime.datetime.now().isoformat()


@tool
def weather_tool(city_name: str) -> str:
    """This function fetches the weather data of the given city"""

    url = 'http://api.weatherstack.com/current?' \
        f'access_key={WEATHER_API_KEY}&query={city_name}'
    resp = requests.get(url)
    return str(resp.json())


tools: list[BaseTool] = [
    weather_tool,
    search_tool,
    current_date_time,
    odd_or_even
]

agent = create_react_agent(
    llm=llm,
    prompt=prompt,
    tools=[
        weather_tool,
        search_tool,
        current_date_time,
        odd_or_even
    ],
)


agent_executor = AgentExecutor(
    verbose=True,
    agent=agent,
    tools=[
        weather_tool,
        search_tool,
        current_date_time,
        odd_or_even
    ],
)


# query = input('Question:')
test_query = "Find the capital of Madhya Pradesh, then find it's current" \
    " weather condition"
test_2 = "What's the date today, if it's odd day give me the weather of "\
    "windsor else just give me the date?"

response = agent_executor.invoke(
    {"input": test_2}
)

print(response['output'])
