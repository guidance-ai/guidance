from typing import Iterable

from .ast import ContentChunk, Node
from .client import Client, TransformersClient
from .model import Model
from .state import (
    APIState,
    CompletionState,
    Llama3TransformersState,
    OpenAIState,
    TransformersUnstructuredState,
)


class DummyClient(Client):
    def run(self, state: APIState, node: Node) -> Iterable[ContentChunk]:
        if isinstance(node, str):
            yield node
        else:
            raise NotImplementedError("Node must be a string")


class DummyCompletionState(CompletionState[str]):
    def get_state(self) -> str:
        return self.prompt


def chat():
    import json

    for s in [
        OpenAIState,
        TransformersUnstructuredState,
        Llama3TransformersState,
    ]:
        model = Model(DummyClient(), s())
        with model.system():
            model += "Talk like a pirate!"
        with model.user():
            model += "Hello, model!"
            model += "\nHow are you?"
        with model.assistant():
            model += "I'm doing well, thank you!"
        print("-" * 80)
        print(s.__name__)
        print("-" * 80)
        print(json.dumps(model._api_state.get_state(), indent=2))


def completion():
    for s in [
        DummyCompletionState,
    ]:
        model = Model(DummyClient(), s())
        model += "<|system|>\nTalk like a pirate!\n<|end_of_turn|>\n"
        model += "<|user|>\nHello, model!\n<|end_of_turn|>\n"
        model += "<|user|>\nHow are you?\n<|end_of_turn|>\n"
        model += "<|assistant|>\nI'm doing well, thank you!\n<|end_of_turn|>\n"
        print("-" * 80)
        print(s.__name__)
        print("-" * 80)
        print(model._api_state.get_state())


def transformers():
    from guidance import gen

    model = Model(TransformersClient(), TransformersUnstructuredState())
    with model.system():
        model += "Talk like a pirate!"
    with model.user():
        model += "Hello, model!"
    with model.assistant():
        model += gen()
    return model
