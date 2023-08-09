import guidance
import pytest
from utils import get_llm

def test_chat_stream():
    """ Test the behavior of `stream=True` for an openai chat endpoint.
    """

    import asyncio
    loop = asyncio.new_event_loop()

    guidance.llm = get_llm("openai:gpt-3.5-turbo")

    async def f():
        chat = guidance("""<|im_start|>system
You are a helpful assistent.
<|im_end|>
<|im_start|>user
{{command}}
<|im_end|>
<|im_start|>assistant
{{gen 'answer' max_tokens=10 stream=True}}""")
        out = await chat(command="How do I create a Fasttokenizer with hugging face auto?")
        assert len(out["answer"]) > 0
    loop.run_until_complete(f())

def test_chat_display():
    """ Test the behavior of `stream=True` for an openai chat endpoint.
    """

    import asyncio
    loop = asyncio.new_event_loop()

    guidance.llm = get_llm("openai:gpt-3.5-turbo")

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
    """Test agents, calling prompt twice"""

    guidance.llm = get_llm("openai:gpt-3.5-turbo")

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

@pytest.mark.parametrize("llm", ["transformers:gpt2", "openai:text-curie-001"])
def test_stream_loop(llm):
    llm = get_llm(llm)
    program = guidance("""Generate a list of 5 company names:
{{#geneach 'companies' num_iterations=5~}}
{{@index}}. "{{gen 'this' max_tokens=5}}"
{{/geneach}}""", llm=llm)

    partials = []
    for p in program(stream=True, silent=True):
        partials.append(p.get("companies", []))
    assert len(partials) > 1
    assert len(partials[0]) < 5
    assert len(partials[-1]) == 5

@pytest.mark.parametrize("llm", ["transformers:gpt2", "openai:text-curie-001"])
def test_stream_loop_async(llm):
    """ Test the behavior of `stream=True` for an openai chat endpoint.
    """

    import asyncio
    loop = asyncio.new_event_loop()

    llm = get_llm(llm)

    async def f():
        program = guidance("""Generate a list of 5 company names:
{{#geneach 'companies' num_iterations=5~}}
{{@index}}. "{{gen 'this' max_tokens=5}}"
{{/geneach}}""", llm=llm)

        partials = []
        async for p in program(stream=True, async_mode=True, silent=True):
            partials.append(p.get("companies", []))
        assert len(partials) > 1
        assert len(partials[0]) < 5
        assert len(partials[-1]) == 5
    loop.run_until_complete(f())

def test_logging_on():
    program = guidance("""This is a test prompt{{#if flag}} yes.{{/if}}""", log=True)
    executed_program = program(flag=True)
    assert len(executed_program.log) > 0

def test_logging_off():
    program = guidance("""This is a test prompt{{#if flag}} yes.{{/if}}""", log=False)
    executed_program = program(flag=True)
    assert executed_program.log is False


def test_async_mode_exceptions():
    """
    Ensures that exceptions in async_mode=True don't hang the program and are
    re-raised back to the caller.
    """
    import asyncio
    loop = asyncio.new_event_loop()

    guidance.llm = get_llm("openai:gpt-3.5-turbo")

    async def call_async():
        program = guidance("""
{{#system~}}
You are a helpful assistant.
{{~/system}}

{{#user~}}
What is your name?
{{~/user}}

{{#assistant~}}
Hello my name is {{gen 'name' temperature=0 max_tokens=5}}.
{{~/assistant}}
""",
            async_mode=True
        )

        return await program()

    task = loop.create_task(call_async())
    completed_tasks, _ = loop.run_until_complete(
        asyncio.wait([task], timeout=5.0)
    )

    try:
        assert len(completed_tasks) == 1, "The task did not complete before timeout"
    finally:
        task.cancel()
        loop.run_until_complete(asyncio.sleep(0)) # give the loop a chance to cancel the tasks

    completed_task = list(completed_tasks)[0]

    assert isinstance(completed_task.exception(), AssertionError), \
        "Expect the exception to be propagated"

    loop.close()




# TGWUI test some have issues TODO
def test_basic_gen():
    model = get_llm("tgwui:http://127.0.0.1:9555",kwargs={'chat_mode':False})
    prompt = guidance("if you give a mouse a cookie, {{gen 'next_verse' temperature=0.7}}", llm=model)
    res = prompt()
    assert res is not None  # Assuming you expect a non-null result
    
def test_encode_to_decode():
    model = get_llm("tgwui:http://127.0.0.1:9555",kwargs={'chat_mode':False})
    string ="Hello World"
    tokens= model.encode(string)
    converted = model.decode(tokens)
    print(converted)
    assert string == converted
    
def test_chat_mode():
    model = get_llm("tgwui:http://127.0.0.1:9555",kwargs={'chat_mode':True})
    experts = guidance('''
    {{#system~}}
    You are a helpful and terse assistant.
    {{~/system}}

    {{#user~}}
    I want a response to the following question:
    {{query}}
    Name 3 world-class experts (past or present) who would be great at answering this?
    Don't answer the question yet.
    {{~/user}}

    {{#assistant~}}
    {{gen 'expert_names' temperature=0 max_tokens=300}}
    {{~/assistant}}

    {{#user~}}
    Great, now please answer the question as if these experts had collaborated in writing a joint anonymous answer.
    {{~/user}}

    {{#assistant~}}
    {{gen 'answer' temperature=0 max_tokens=500}}
    {{~/assistant}}
    ''', llm=model)
    res = experts(query='How can I be more productive?')
    print(res)
    assert res is not None  # Assuming you expect a non-null result

def test_basic_geneach():
    model = get_llm("tgwui:http://127.0.0.1:9555",kwargs={'chat_mode':False})
    prompt = guidance("""{{#geneach 'items' num_iterations=3}} "{{gen 'this'}}",{{/geneach}}""", llm=model)
    res = prompt()
    print(res)
    assert res is not None  # Assuming you expect a non-null result
    
def test_basic_pattern_gen():
    model = get_llm("tgwui:http://127.0.0.1:9555",kwargs={'chat_mode':False})
    prompt = guidance("strength: {{gen 'strength' pattern='[0-9]+' temperature=0.7}}", llm=model)
    res = prompt()
    print(res)
    assert isinstance(int(res), int)  # Assuming you expect a numeric string

def test_basic_select():
    model = get_llm("tgwui:http://127.0.0.1:9555",kwargs={'chat_mode':False})
    valid_weapons = ["sword", "axe", "mace", "spear", "bow", "crossbow"]
    prompt = guidance("weapon {{select 'weapon' options=valid_weapons}}", valid_weapons=valid_weapons, llm=model)
    res = prompt()
    print(res)
    assert res in valid_weapons  # Expect selected weapon to be in the valid_weapons list


def test_basic_stop():
    model = get_llm("tgwui:http://127.0.0.1:9555",kwargs={'chat_mode':False})
    prompt = guidance("how {{gen 'strength' stop=',' temperature=0.7}}", llm=model)
    res = str(prompt())
    print(res)
    assert res.endswith(',')

