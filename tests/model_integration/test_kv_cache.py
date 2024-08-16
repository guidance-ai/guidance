import pytest

import guidance
from guidance import byte_range, gen, models, select, zero_or_more

def test_cache_linear(selected_model: models.Model):
    a = "Hello"
    b = "Hello, world!"
    c = "Hello, world! How are you?"

    a_tokens = selected_model.engine.tokenizer.encode(a.encode())
    b_tokens = selected_model.engine.tokenizer.encode(b.encode())
    c_tokens = selected_model.engine.tokenizer.encode(c.encode())

    # prime with a string not sharing a prefix to ensure we are not using the cache
    lm = selected_model + "buzzbuzz" + gen(max_tokens=1)

    selected_model.engine.reset_metrics()
    lm1 = selected_model + a + gen(max_tokens=1)
    metrics1 = selected_model.engine.metrics
    assert metrics1.engine_input_tokens == len(a_tokens)

    selected_model.engine.reset_metrics()
    lm2 = selected_model + b + gen(max_tokens=1)
    metrics2 = selected_model.engine.metrics
    shared_prefix = 0
    for i in range(len(a_tokens)):
        if a_tokens[i] == b_tokens[i]:
            shared_prefix += 1
        else:
            break
    assert metrics2.engine_input_tokens == len(b_tokens) - shared_prefix

    selected_model.engine.reset_metrics()
    lm3 = selected_model + c + gen(max_tokens=1)
    metrics3 = selected_model.engine.metrics
    shared_prefix = 0
    for i in range(len(b_tokens)):
        if b_tokens[i] == c_tokens[i]:
            shared_prefix += 1
        else:
            break
    assert metrics3.engine_input_tokens == len(c_tokens) - shared_prefix

def test_cache_diverged(selected_model: models.Model):
    a = "Hello, world!"
    b = "Hello, goodbye."
    c = "Hello, how are you?"

    a_tokens = selected_model.engine.tokenizer.encode(a.encode())
    b_tokens = selected_model.engine.tokenizer.encode(b.encode())
    c_tokens = selected_model.engine.tokenizer.encode(c.encode())

    # prime with a string not sharing a prefix to ensure we are not using the cache
    lm = selected_model + "buzzbuzz" + gen(max_tokens=1)

    selected_model.engine.reset_metrics()
    lm1 = selected_model + a + gen(max_tokens=1)
    metrics1 = selected_model.engine.metrics
    print(selected_model.engine._cached_token_ids)
    print(a_tokens)
    print([selected_model.engine.tokenizer.decode([tok]) for tok in a_tokens])
    assert metrics1.engine_input_tokens == len(a_tokens)

    selected_model.engine.reset_metrics()
    lm2 = selected_model + b + gen(max_tokens=1)
    metrics2 = selected_model.engine.metrics
    shared_prefix = 0
    for i in range(len(a_tokens)):
        if a_tokens[i] == b_tokens[i]:
            shared_prefix += 1
        else:
            break
    assert metrics2.engine_input_tokens == len(b_tokens) - shared_prefix

    selected_model.engine.reset_metrics()
    lm3 = selected_model + c + gen(max_tokens=1)
    metrics3 = selected_model.engine.metrics
    shared_prefix = 0
    for i in range(len(b_tokens)):
        if b_tokens[i] == c_tokens[i]:
            shared_prefix += 1
        else:
            break
    assert metrics3.engine_input_tokens == len(c_tokens) - shared_prefix
    print(a_tokens, b_tokens, c_tokens)