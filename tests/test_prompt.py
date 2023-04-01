import guidance

def test_variable_interpolation():
    """ Test variable interpolation in prompt
    """

    prompt = guidance("Hello, {{name}}!")
    assert str(prompt(name="Guidance")) == "Hello, Guidance!"

def test_each():
    """ Test an each loop.
    """

    prompt = guidance("Hello, {{name}}!{{#each names}} {{this}}{{/each}}")
    assert str(prompt(name="Guidance", names=["Bob", "Sue"])) == "Hello, Guidance! Bob Sue"

def test_each_with_objects():
    """ Test an each loop with objects.
    """

    prompt = guidance("Hello, {{name}}!{{#each names}} {{this.name}}{{/each}}")
    out = prompt(
        name="Guidance",
        names=[{"name": "Bob"}, {"name": "Sue"}]
    )
    assert str(out) == "Hello, Guidance! Bob Sue"

def test_generate():
    """ Test that LM geneation works.
    """
    llm = guidance.llms.OpenAI("text-curie-001")

    prompt = guidance("Hello my name is{{generate 'name' max_tokens=5}}", llm=llm)
    out = prompt()
    assert len(out["name"]) > 1

def test_select():
    """ Test the behavior of `select`.
    """

    llm = guidance.llms.OpenAI("text-curie-001")
    prompt = guidance("Is Everest very tall?\nAnswer 'Yes' or 'No': '{{#select 'name'}}Yes{{or}}No{{/select}}", llm=llm)
    out = prompt()
    assert out["name"] in ["Yes", "No"]

def test_await():
    """ Test the behavior of `await`.
    """

    llm = guidance.llms.OpenAI("text-curie-001")
    prompt = guidance("""Is Everest very tall?
User response: '{{set 'user_response' (await 'user_response')}}'
Answer 'Yes' or 'No': '{{#select 'name'}}Yes{{or}}No{{/select}}""", llm=llm)
    waiting_prompt = prompt()
    assert str(waiting_prompt) == "Is Everest very tall?\nUser response: '{{set 'user_response' (await 'user_response')}}'\nAnswer 'Yes' or 'No': '{{#select 'name'}}Yes{{or}}No{{/select}}"
    out = waiting_prompt(user_response="Yes")
    assert str(out).startswith("Is Everest very tall?\nUser response: 'Yes'\nAnswer 'Yes' or 'No': '")
    assert out["name"] in ["Yes", "No"]

def test_if():
    """ Test the behavior of `if`.
    """

    llm = guidance.llms.OpenAI("text-curie-001")
    prompt = guidance("""Answer: {{#if flag}}Yes{{/if}}""", llm=llm)

    for flag in [True, "Yes", "yes", "YES", 1, 1.0, "True", "true", "TRUE"]:
        out = prompt(flag=flag)
        assert str(out) == "Answer: Yes"
    
    for flag in [False, "No", "no", "NO", 0, 0.0, "False", "false", "FALSE"]:
        out = prompt(flag=flag)
        assert str(out) == "Answer: "

def test_if_else():
    """ Test the behavior of `if` with an `else` clause.
    """

    llm = guidance.llms.OpenAI("text-curie-001")
    prompt = guidance("""Answer 'Yes' or 'No': '{{#if flag}}Yes{{else}}No{{/if}}'""", llm=llm)

    for flag in [True, "Yes", "yes", "YES", 1, 1.0, "True", "true", "TRUE"]:
        out = prompt(flag=flag)
        assert str(out) == "Answer 'Yes' or 'No': 'Yes'"
    
    for flag in [False, "No", "no", "NO", 0, 0.0, "False", "false", "FALSE"]:
        out = prompt(flag=flag)
        assert str(out) == "Answer 'Yes' or 'No': 'No'"


def test_hidden_block():
    """ Test the behavior of `if` with an `else` clause.
    """

    llm = guidance.llms.OpenAI("text-curie-001")
    prompt = guidance("""This is a test {{#block hidden=True}}example{{/block}}""", llm=llm)
    out = prompt()
    assert out.text == "This is a test "

def test_chat_stream():
    """ Test the behavior of `stream=True` for an openai chat endpoint.
    """

    import asyncio
    loop = asyncio.new_event_loop()

    import guidance
    guidance.llm = guidance.llms.OpenAI("gpt-4", chat_completion=True)

    async def f():
        chat = guidance("""<|im_start|>system
        You are a helpful assistent.
        <|im_end|>
        <|im_start|>user
        {{command}}
        <|im_end|>
        <im_start|>assistant
        {{generate 'answer' max_tokens=10}}""", stream=True)
        out = await chat(command="How do I create a Fasttokenizer with hugging face auto?-b")
        assert len(out["answer"]) > 0
    loop.run_until_complete(f())

def test_chat_echo():
    """ Test the behavior of `stream=True` for an openai chat endpoint.
    """

    import asyncio
    loop = asyncio.new_event_loop()

    import guidance
    guidance.llm = guidance.llms.OpenAI("gpt-4", chat_completion=True)

    async def f():
        chat = guidance("""<|im_start|>system
        You are a helpful assistent.
        <|im_end|>
        <|im_start|>user
        {{command}}
        <|im_end|>
        <im_start|>assistant
        {{generate 'answer' max_tokens=10}}""", echo=True)
        out = await chat(command="How do I create a Fasttokenizer with hugging face auto?-b")
        assert len(out["answer"]) > 0
    loop.run_until_complete(f())

def test_agents():
    """Test agentes, calling prompt twice"""
    import guidance
    guidance.llm = guidance.llms.OpenAI("gpt-4", chat_completion=True)
    prompt = guidance('''<|im_start|>system
    You are a helpful assistant.<|im_end|>
    {{#each 'conversation'}}
    <|im_start|>user
    {{set 'this.user_text' (await 'user_text')}}<|im_end|>
    <|im_start|>assistant
    {{generate 'this.ai_text' n=1 temperature=0 max_tokens=900}}<|im_end|>{{/each}}''', echo=True)
    prompt = prompt(user_text='Hi there')
    assert len(prompt['conversation']) == 2
    prompt = prompt(user_text='Please help')
    assert len(prompt['conversation']) == 3

def test_generate_n_greater_than_one():
    """Test agentes, calling prompt twice"""
    llm = guidance.llms.OpenAI("text-curie-001")
    prompt = guidance('''The best thing about the beach is{{generate 'best' n=3 temperature=0.7 max_tokens=5}}''', llm=llm)
    a = prompt()
    assert len(a["best"]) == 3

def test_generate_n_greater_than_one():
    llm = guidance.llms.OpenAI("text-curie-001")
    prompt = guidance('''The best thing about the beach is{{generate 'best' n=3 temperature=0.7 max_tokens=5}}''', llm=llm)
    a = prompt()
    assert len(a["best"]) == 3

def test_missing_list():
    llm = guidance.llms.OpenAI("text-curie-001")
    prompt = guidance('''List of ideas:{{#each ideas}}test{{this}}{{/each}}''', llm=llm)
    out = prompt()
    assert out.text == "List of ideas:"