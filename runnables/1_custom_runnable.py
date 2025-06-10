from abc import ABC, abstractmethod
from typing import Any
from logging import warning


class FakeRunnable(ABC):
    @abstractmethod
    def invoke(self, data: Any) -> Any:
        pass


class RunnableConnector(FakeRunnable):
    def __init__(self, runnable_list: list[FakeRunnable]) -> None:
        super().__init__()
        self.runnable_list = runnable_list

    def invoke(self, data: Any) -> Any:
        for runnable in self.runnable_list:
            data = runnable.invoke(data)
        return data


class FakeLLM(FakeRunnable):
    def __init__(self) -> None:
        print('LLM created')

    def invoke(self, data: str) -> dict:
        return {'content': 'some dummy response', 'prompt': data}

    def predict(self, data: str) -> dict:
        warning("function (predict) is deprecated. use 'invoke' instead")
        return {'content': 'some dummy response', 'prompt': data}


class FakePromptTemplate(FakeRunnable):

    def __init__(self,  input_variables: list[str], template: str) -> None:
        super().__init__()
        self.input_variables = input_variables
        self.template = template

    def invoke(self, data: dict[str, Any]) -> str:
        return self.template.format(**data)

    def format(self, data: dict[str, Any]) -> str:

        warning("function (format) is deprecated. use 'invoke' instead")
        return self.template.format(**data)


class FakeStrOutputParser(FakeRunnable):

    def __init__(self) -> None:
        super().__init__()

    def invoke(self, data: dict) -> str:
        return data['content']


# Use the above things
template = FakePromptTemplate(
    input_variables=['topic'],
    template="write a story on the following topic:\n{topic}"
)

llm = FakeLLM()

parser = FakeStrOutputParser()

chain = RunnableConnector([template, llm, parser])

res = chain.invoke({'topic': 'golang'})


print(res)

# old way would give you warn log
prompt = template.format({'topic': 'golang'})

res = llm.predict(prompt)

print(res)
