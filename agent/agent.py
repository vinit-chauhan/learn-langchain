import os
import datetime
import requests

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool, BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain import hub


load_dotenv()

WEATHER_API_KEY = os.environ['WEATHER_API_KEY']

prompt = hub.pull('hwchase17/react')

llm = ChatOpenAI(model='gpt-4o-mini')

search_tool = DuckDuckGoSearchRun()


@tool
def odd_or_even(num: str) -> str:
    """Returns if a number is odd or even."""

    if num.isnumeric():
        return "Even" if int(num) % 2 == 0 else "Odd"

    return "Invalid number"


@tool
def current_date_time() -> str:
    """This tool runs and give you the current date and time"""

    return datetime.datetime.now().isoformat()


@tool
def weather_tool(city: str) -> str:
    """This function fetches the weather data of the given city"""

    url = 'http://api.weatherstack.com/current?' \
        f'access_key={WEATHER_API_KEY}&query={city}'
    resp = requests.get(url)
    return resp.json()


tools: list[BaseTool] = [
    weather_tool,
    search_tool,
    current_date_time,
    odd_or_even
]

agent = create_react_agent(
    llm=llm,
    prompt=prompt,
    tools=tools,
)


agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)


query = input('Question:')
test_query = "Find the capital of Madhya Pradesh, then find it's current" \
    " weather condition"
test_2 = "What's the date today, if it's odd day give me the wather of "\
    "windsor else just give me the date?"

response = agent_executor.invoke(
    {"input": test_2})

print(response['output'])
