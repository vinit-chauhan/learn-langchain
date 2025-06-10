import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

load_dotenv()
model_name = os.environ['OPENAI_MODEL']
model = ChatOpenAI(model=model_name)
