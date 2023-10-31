import guidance
from ..utils import get_llm
import pytest


def test_context_palm():
    """
    Test the context role opposed or combined with the system role
    """
    guidance.llm = get_llm("palm:chat-bison")

    chat = guidance(
        """
{{~#context~}}You are a helpful assistant{{~/context~}}
{{~#user~}}Hi, how are you?{{~/user~}}
{{~#assistant~}}Fine.{{~/assistant~}}
"""
    )
    out = chat()
    assert (
        str(out)
        == "<|im_start|>context\nYou are a helpful assistant<|im_end|><|im_start|>user\nHi, how are you?<|im_end|><|im_start|>assistant\nFine.<|im_end|>"
    )


@pytest.mark.parametrize("rest_call", [True, False])
def test_examples_palm(rest_call):
    guidance.llm = get_llm("palm:chat-bison", rest_call=rest_call)
    chat = guidance(
        """
{{~#context~}}
You always feel the same
{{~/context~}}
{{~#example~}}
    {{~#input~}}
Hi, how are you?
    {{~/input~}}
    {{~#output~}}
{{feeling}}
    {{~/output~}}
{{~/example~}}
{{~#user~}}
Hi, how are you?
{{~/user~}}
{{~#assistant~}}
{{gen 'answer' max_tokens=5 temperature=0}}
{{~/assistant~}}
"""
    )
    feeling = "I'm alright"
    out = chat(feeling=feeling)
    assert out["answer"].strip().startswith(feeling)

    feeling = "I'm fine"
    out = chat(feeling=feeling)
    assert out["answer"].strip().startswith(feeling)


def test_examples_palm_32k():
    guidance.llm = get_llm("palm:chat-bison-32k", rest_call=True)
    chat = guidance(
        """
{{~#context~}}
You always feel the same
{{~/context~}}
{{~#example~}}
    {{~#input~}}
Hi, how are you?
    {{~/input~}}
    {{~#output~}}
{{feeling}}
    {{~/output~}}
{{~/example~}}
{{~#user~}}
Hi, how are you?
{{~/user~}}
{{~#assistant~}}
{{gen 'answer' max_tokens=5 temperature=0}}
{{~/assistant~}}
"""
    )
    feeling = "I'm alright"
    out = chat(feeling=feeling)
    assert out["answer"].strip().startswith(feeling)

    feeling = "I'm fine"
    out = chat(feeling=feeling)
    assert out["answer"].strip().startswith(feeling)


def test_geneach_palm():
    """Test a geneach loop with ChatGPT."""
    guidance.llm = get_llm("palm:chat-bison")

    chat_loop = guidance(
        """
{{#system~}}
You are a helpful assistant
{{~/system}}

{{~#geneach 'conversation' stop=False}}
{{#user~}}
This is great!
{{~/user}}

{{#assistant~}}
{{gen 'this.response' temperature=0 max_tokens=3}}
{{~/assistant}}
{{#if @index > 0}}{{break}}{{/if}}
{{~/geneach}}"""
    )

    out = chat_loop()
    assert len(out["conversation"]) == 2


def test_syntax_match():
    """Test a geneach loop with ChatGPT."""
    guidance.llm = get_llm("palm:chat-bison")

    chat_loop = guidance(
        """
{{~#system~}}
You are a helpful assistant
{{~/system~}}

{{~#user~}}
This is great!
{{~/user~}}

{{~#assistant~}}
Indeed
{{~/assistant~}}"""
    )

    out = chat_loop()
    assert (
        str(out)
        == "<|im_start|>system\nYou are a helpful assistant<|im_end|><|im_start|>user\nThis is great!<|im_end|><|im_start|>assistant\nIndeed<|im_end|>"
    )


def test_nostream():
    guidance.llm = get_llm("palm:text-bison")
    a = guidance(
        """What's your name? {{gen 'name' stream=False max_tokens=5}}""", stream=False
    )
    a = a()
    assert len(a["name"]) > 0


def test_rest_nostream():
    guidance.llm = get_llm("palm:text-bison", rest_call=True)
    a = guidance(
        """What's your name? {{gen 'name' stream=False max_tokens=5}}""", stream=False
    )
    a = a()
    assert len(a["name"]) > 0


def test_rest_stream():
    import asyncio

    loop = asyncio.new_event_loop()

    async def f():
        guidance.llm = get_llm("palm:text-bison", rest_call=True)
        a = guidance(
            """What's your {{question}}? tell me the first chapter of the odissey {{gen 'name' stream=True max_tokens=50}}"""
        )
        a = await a(question="name")
        assert len(a["name"]) > 0

    loop.run_until_complete(f())


def test_rest_stream_chat():
    import asyncio

    loop = asyncio.new_event_loop()

    async def f():
        guidance.llm = get_llm("palm:chat-bison", rest_call=True)
        chat = guidance(
            """
{{~#user~}}
tell me the first chapter of the {{book}}
{{~/user~}}
{{~#assistant~}}
{{gen 'answer' max_tokens=50 temperature=0 stream=True}}
{{~/assistant~}}
"""
        )
        chat = await chat(book="odissey")
        print(chat)
        assert len(chat["answer"]) > 0

    loop.run_until_complete(f())


def test_rest_chat_nostream():
    guidance.llm = get_llm("palm:chat-bison", rest_call=True)
    prompt = guidance(
        """{{~#system~}}
You are a helpful assistant
{{~/system~}}
{{#user~}}
{{conversation_question}}
{{~/user}}
{{#assistant~}}
{{gen "answer" max_tokens=5 stream=False}}
{{~/assistant}}"""
    )
    prompt = prompt(conversation_question="Whats is the meaning of life??")
    assert len(prompt["answer"]) > 0
