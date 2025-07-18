from json import loads

import pytest

from guidance import assistant, gen, json, models, special_token, system, user


@pytest.fixture(scope="module")
def qwen3(llamacpp_model: models.Model, selected_model_name: str) -> models.LlamaCpp:
    if selected_model_name == "llamacpp_qwen3_0dot6b_cpu":
        return llamacpp_model
    pytest.skip(f"Selected model {selected_model_name} is not a Qwen3 model, skipping qwen3 fixture")


def test_qwen3_think_special_token(qwen3: models.LlamaCpp):
    lm = qwen3
    with system():
        lm += "You are a geography expert. Think step by step to answer the question in JSON format."
    with user():
        lm += """What is the capital of France? Answer with the correct letter.
    A) Berlin
    B) London 
    C) Paris
    D) Madrid
    """
    with assistant():
        lm += (
            special_token("<think>")
            + "\n"
            + gen(max_tokens=100, name="reasoning")
            + special_token("</think>")
            + "\n"
            + json(
                schema={
                    "type": "object",
                    "properties": {
                        "answer": {"enum": ["A", "B", "C", "D"]},
                    },
                    "required": ["answer"],
                    "additionalProperties": False,
                },
                name="answer",
            )
        )
    assert len(lm["reasoning"]) > 0
    assert "<think>" not in lm["reasoning"]
    assert loads(lm["answer"])["answer"] in ["A", "B", "C", "D"]
