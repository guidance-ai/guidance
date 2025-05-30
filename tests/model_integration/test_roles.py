import guidance
from guidance import assistant, user, system, models, gen


@guidance
def plan_for_goal(lm, goal: str):
    # This is condensed out of the chat.ipynb notebook

    # Some general instruction to the model
    with system():
        lm += "You are a helpful assistant."

    LM_EXPECT = "<|im_start|>system\nYou are a helpful assistant."
    assert str(lm) == LM_EXPECT, f"{str(lm)=}"

    with user():
        lm2 = lm + f"""I want to {goal}. Make some very short suggestions"""

    assert str(lm) == "<|im_start|>system\nYou are a helpful assistant.", f"{str(lm)=}"
    LM2_EXPECT = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nI want to Add numbers. Make some very short suggestions"
    assert str(lm2) == LM2_EXPECT, f"{str(lm2)=}"

    n_opts = 2
    options = []
    with assistant():
        for _ in range(n_opts):
            lm3 = lm2 + gen(name="option", temperature=1.0, max_tokens=50)
            LM3_BASE = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nI want to Add numbers. Make some very short suggestions<|im_end|>\n<|im_start|>assistant\n"
            selected = lm3["option"]
            assert selected == "something", f"{selected=}"
            assert str(lm3) == LM3_BASE + selected, f"{str(lm3)=}"
            assert str(lm2) == LM2_EXPECT, f"{str(lm2)=}"
            options.append(selected)
    assert len(options) == 2

    return lm


def test_plan_for_goal(chat_model: models.Model):
    lm = chat_model + plan_for_goal("Add numbers")
    assert lm is not None
