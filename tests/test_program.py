import pytest
import asyncio
from .utils import get_openai_llm

@pytest.fixture
def openai_llm():
    return get_openai_llm("gpt-3.5-turbo")

@pytest.fixture
def event_loop():
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()

def test_variable_interpolation():
    """Test variable interpolation in prompt."""
    prompt = guidance("Hello, {{name}}!")
    assert str(prompt(name="Guidance")) == "Hello, Guidance!"

@pytest.mark.asyncio
async def test_chat_stream(openai_llm):
    """Test the behavior of `stream=True` for an OpenAI chat endpoint."""
    chat = guidance(
        """system
You are a helpful assistant.

user
{{command}}

assistant
{{gen 'answer' max_tokens=10}}""",
        llm=openai_llm,
        stream=True
    )
    out = await chat(command="How do I create a Fasttokenizer with hugging face auto?")
    assert len(out["answer"]) > 0

@pytest.mark.asyncio
async def test_chat_display(openai_llm):
    """Test the behavior of `stream=True` for an OpenAI chat endpoint."""
    chat = guidance(
        """system
You are a helpful assistant.

user
{{command}}

assistant
{{gen 'answer' max_tokens=10}}""",
        llm=openai_llm
    )
    out = await chat(command="How do I create a Fasttokenizer with hugging face auto?")
    assert len(out["answer"]) > 0

def test_agents(openai_llm):
    """Test agents, calling prompt twice."""
    prompt = guidance(
        '''system
You are a helpful assistant.
{{#geneach 'conversation' stop=False}}
user
{{set 'this.user_text' (await 'user_text')}}
assistant
{{gen 'this.ai_text' n=1 temperature=0 max_tokens=900}}{{/geneach}}''',
        llm=openai_llm,
        echo=True
    )
    prompt = prompt(user_text='Hi there')
    assert len(prompt['conversation']) == 2
    prompt = prompt(user_text='Please help')
    assert len(prompt['conversation']) == 3
