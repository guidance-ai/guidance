import guidance
from .utils import get_openai_llm

def test_variable_interpolation():
    """ Test variable interpolation in prompt
    """

    prompt = guidance("Hello, {{name}}!")
    assert str(prompt(name="Guidance")) == "Hello, Guidance!"

def test_chat_stream():
    """ Test the behavior of `stream=True` for an openai chat endpoint.
    """

    import asyncio
    loop = asyncio.new_event_loop()

    guidance.llm = get_openai_llm("gpt-3.5-turbo")

    async def f():
        chat = guidance("""<|im_start|>system
You are a helpful assistent.
<|im_end|>
<|im_start|>user
{{command}}
<|im_end|>
<|im_start|>assistant
{{gen 'answer' max_tokens=10}}""", stream=True)
        out = await chat(command="How do I create a Fasttokenizer with hugging face auto?")
        assert len(out["answer"]) > 0
    loop.run_until_complete(f())

def test_chat_display():
    """ Test the behavior of `stream=True` for an openai chat endpoint.
    """

    import asyncio
    loop = asyncio.new_event_loop()

    guidance.llm = get_openai_llm("gpt-3.5-turbo")

    async def f():
        chat = guidance("""<|im_start|>system
You are a helpful assistent.
<|im_end|>
<|im_start|>user
{{command}}
<|im_end|>
<|im_start|>assistant
{{gen 'answer' max_tokens=10}}""")
        out = await chat(command="How do I create a Fasttokenizer with hugging face auto?")
        assert len(out["answer"]) > 0
    loop.run_until_complete(f())

def test_agents():
    """Test agentes, calling prompt twice"""

    guidance.llm = get_openai_llm("gpt-3.5-turbo")

    prompt = guidance('''<|im_start|>system
You are a helpful assistant.<|im_end|>
{{#geneach 'conversation' stop=False}}
<|im_start|>user
{{set 'this.user_text' (await 'user_text')}}<|im_end|>
<|im_start|>assistant
{{gen 'this.ai_text' n=1 temperature=0 max_tokens=900}}<|im_end|>{{/geneach}}''', echo=True)
    prompt = prompt(user_text='Hi there')
    assert len(prompt['conversation']) == 2
    prompt = prompt(user_text='Please help')
    assert len(prompt['conversation']) == 3