from guidance import assistant, gen, models, system, user


def smoke_chat(lm: models.Chat, has_system_role: bool = True):
    if has_system_role:
        with system():
            lm += "You are a math wiz."

    with user():
        lm += "What is 1 + 1?"

    with assistant():
        lm += gen(max_tokens=10, name="text", temperature=0.5)
        lm += "Pick a number: "

    print(str(lm))
    assert len(lm["text"]) > 0
    assert str(lm).endswith("Pick a number: <|im_end|>")
    assert lm.metrics.prompt_tokens > 0
    assert lm.metrics.generated_tokens > 0
    assert lm.metrics.generated_tokens <= 10


def longer_chat_1(lm: models.Chat, has_system_role: bool = True):
    if has_system_role:
        with system():
            lm += "You are a math wiz."

    with user():
        lm += "What is 1 + 1?"

    with assistant():
        lm += gen(max_tokens=10, name="text")
        lm += "Pick a number: "

    print(str(lm))
    assert len(lm["text"]) > 0
    assert str(lm).endswith("Pick a number: <|im_end|>")

    with user():
        lm += "10. Now you pick a number between 0 and 20"

    with assistant():
        lm += gen(max_tokens=2, name="number")

    print(str(lm))
    assert len(lm["number"]) > 0


def longer_chat_2(lm: models.Chat, has_system_role: bool = True):
    if has_system_role:
        with system():
            lm += "You are a math wiz."

    with user():
        lm += "What is 1 + 1?"

    # This is the new part compared to longer_chat_1
    with assistant():
        lm += "2"

    with user():
        lm += "What is 2 + 3?"

    # Resume the previous
    with assistant():
        lm += gen(max_tokens=10, name="text")
        lm += "Pick a number: "

    print(str(lm))
    assert len(lm["text"]) > 0
    assert str(lm).endswith("Pick a number: <|im_end|>")

    with user():
        lm += "10. Now you pick a number between 0 and 20"

    with assistant():
        lm += gen(max_tokens=2, name="number")

    print(str(lm))
    assert len(lm["number"]) > 0
