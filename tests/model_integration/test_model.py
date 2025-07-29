from unittest.mock import patch

import pytest

import guidance
from guidance import gen, models, regex, select


def test_capture_casing(selected_model):
    # From issue 1172
    lm = selected_model
    lm += "This is a test of the capture function: " + gen(name="CamelCaseName", max_tokens=10)
    assert "CamelCaseName" in lm
    assert isinstance(lm["CamelCaseName"], str)
    assert len(lm["CamelCaseName"]) > 0


def test_capture_casing_listappend(selected_model):
    lm = selected_model
    lm += """Write three story title options about the arctic circle:
OUTLINE
"""
    lm += '1. "' + gen(name="camelStory", max_tokens=20, list_append=True, stop='"') + '"'
    lm += '2. "' + gen(name="camelStory", max_tokens=20, list_append=True, stop='"') + '"'
    lm += '3. "' + gen(name="camelStory", max_tokens=20, list_append=True, stop='"') + '"'
    assert isinstance(lm["camelStory"], list)
    assert len(lm["camelStory"]) == 3
    assert all(isinstance(item, str) for item in lm["camelStory"])


def test_fstring(selected_model):
    lm = selected_model
    lm += f"this is a test {select(['item1', 'item2'])}"
    assert str(lm) in ["this is a test item1", "this is a test item2"]


def test_fstring_custom(selected_model):
    lm = selected_model

    @guidance
    def my_function(lm):
        return lm + f"another {select(['item1', 'item2'])}"

    lm += f"this is a test {my_function()}"
    assert str(lm) in ["this is a test another item1", "this is a test another item2"]


def test_token_count(selected_model):
    lm = selected_model
    lm2 = lm + (" 1 1 1 1 1" + gen(max_tokens=9) + gen(max_tokens=9))
    assert (
        18 <= lm2._get_usage().output_tokens <= 20
    )  # note we allow ourselves to be off by one because it is hard to know when we are continuing vs starting a new token in the parser


def test_token_healing(selected_model):
    """Tests a bug where the space is incorrectly forced as token 220, while it should be not forced it might be extended"""
    model_obj = getattr(selected_model.engine, "model_obj", None)
    if model_obj is None or type(model_obj).__name__ != "GPT2LMHeadModel":
        pytest.skip("Test for GPT2 bug only")
    gpt2 = selected_model
    lm = gpt2 + ("This is a story of 10 or 5 or " + regex(r"[0-9]+"))
    assert len(lm) > len("This is a story of 10 or 5 or ")


def test_stream(selected_model):
    lm = selected_model.stream()
    lm += select(["item1", "item2"])
    *_, last_lm = lm
    assert str(last_lm) in ["item1", "item2"]


def test_stream_propagate_errors(selected_model):
    lm = selected_model.stream()

    @guidance
    def my_function(lm):
        raise Exception()

    lm += my_function()
    with pytest.raises(Exception):
        list(lm)


def test_stream_add_multiple(selected_model):
    """Test to make sure multiple additions to a ModelStream are all respected"""
    lm = selected_model.stream()
    lm += select(["item1", "item2"])
    lm += ""
    *_, last_lm = lm
    assert str(last_lm) in ["item1", "item2"]


def test_associativity(selected_model: models.Model):
    prompt = "pi = "
    grammar = gen("number", regex=r"\d")
    engine = selected_model.engine

    from copy import deepcopy

    original_get_logits = engine.get_logits
    prompt_tokens_1_list = []
    prompt_tokens_2_list = []

    def get_logits_1(*, token_ids, **kwargs):
        prompt_tokens_1_list.append(deepcopy(token_ids))
        return original_get_logits(token_ids=token_ids, **kwargs)

    def get_logits_2(*, token_ids, **kwargs):
        prompt_tokens_2_list.append(deepcopy(token_ids))
        return original_get_logits(token_ids=token_ids, **kwargs)

    with patch.object(engine, "get_logits", side_effect=get_logits_1):
        _ = selected_model
        # Get the index of the prompt tokens after the first call
        ix_1 = len(prompt_tokens_1_list)
        _ += prompt + grammar

    with patch.object(engine, "get_logits", side_effect=get_logits_2):
        _ = selected_model + prompt
        # Get the index of the prompt tokens after the first call
        ix_2 = len(prompt_tokens_2_list)
        _ += grammar

    prompt_tokens_1 = prompt_tokens_1_list[ix_1]
    prompt_tokens_2 = prompt_tokens_2_list[ix_2]

    # Main assertion: the prompt tokens should be the same
    assert prompt_tokens_1 == prompt_tokens_2

    # Further assert that the tokenization matches the expected tokenization
    expected_prompt_tokens = engine.tokenizer.encode(prompt.encode())
    if engine.tokenizer.bos_token is not None and expected_prompt_tokens[:1] != [engine.tokenizer.bos_token_id]:
        expected_prompt_tokens = [engine.tokenizer.bos_token_id] + expected_prompt_tokens
        expected_prompt_tokens = engine.tokenizer.recode(expected_prompt_tokens)
    # token healing may cause the prompt seen by the model to be shorter
    assert len(expected_prompt_tokens) >= len(prompt_tokens_1)
    assert prompt_tokens_1 == expected_prompt_tokens[: len(prompt_tokens_1)]
