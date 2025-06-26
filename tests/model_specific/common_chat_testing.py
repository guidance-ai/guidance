import pydantic

from guidance import assistant, gen, models, system, user
from guidance import json as gen_json


def smoke_chat(lm: models.Model, has_system_role: bool = True):
    # lm.engine.reset_metrics()
    if has_system_role:
        with system():
            lm += "You are a math wiz."

    with user():
        lm += "What is 1 + 1?"

    with assistant():
        lm += gen(max_tokens=10, name="text", temperature=0.5)

    print(str(lm))
    # print(f"{lm.engine.metrics=}")
    assert len(lm["text"]) > 0
    # assert lm.engine.metrics.engine_input_tokens > 2, "Expect some input tokens"
    # assert lm.engine.metrics.engine_output_tokens > 0, "Expect some output tokens"


def longer_chat_1(lm: models.Model, has_system_role: bool = True):
    if has_system_role:
        with system():
            lm += "You are a math wiz."

    with user():
        lm += "What is 1 + 1?"

    with assistant():
        lm += gen(max_tokens=10, name="text")

    print(str(lm))
    assert len(lm["text"]) > 0

    with user():
        lm += "10. Now you pick a number between 0 and 20"

    with assistant():
        lm += gen(max_tokens=2, name="number")

    print(str(lm))
    assert len(lm["number"]) > 0


def longer_chat_2(lm: models.Model, has_system_role: bool = True):
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

    print(str(lm))
    assert len(lm["text"]) > 0

    with user():
        lm += "10. Now you pick a number between 0 and 20"

    with assistant():
        lm += gen(max_tokens=2, name="number")

    print(str(lm))
    assert len(lm["number"]) > 0


def json_output_smoke(lm: models.Model):
    class NameHolder(pydantic.BaseModel):
        my_name: str
        my_age: int
        model_config = pydantic.ConfigDict(extra="forbid")

    with user():
        lm += "Hello, my name is Tweedledum and I am 10 years old. What is my twin brother's name and age?"

    EXTRACT_KEY = "my_output"
    with assistant():
        lm += gen_json(name=EXTRACT_KEY, schema=NameHolder)

    output_json = lm[EXTRACT_KEY]
    name_data = NameHolder.model_validate_json(output_json)
    print(name_data.model_dump_json(indent=4))
    assert name_data.my_name == "Tweedledee"
    assert name_data.my_age == 10
