import guidance
from guidance import assistant, user, system, models, gen

@guidance
def plan_for_goal(lm, goal: str):
    
    # Some general instruction to the model
    with system():
        lm += "You are a helpful assistant."

    
    EXPECTED_LM = "<|im_start|>system\nYou are a helpful assistant."
    print(f"{str(lm)=}")
    assert str(lm) == EXPECTED_LM

    # Simulate a simple request from the user
    # Note that we switch to using 'lm2' here, because these are intermediate steps (so we don't want to overwrite the current lm object)
    with user():
        lm2 = lm + f"""\
        I want to {goal}
        Can you please generate one option for how to accomplish this?
        Please make the option very short, at most one line."""

    EXPECTED_LM2 = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nI want to Add numbers\nCan you please generate one option for how to accomplish this?\nPlease make the option very short, at most one line."
    print(f"{str(lm)=}")
    print(f"{str(lm2)=}")
    assert str(lm) == EXPECTED_LM
    assert str(lm2) == EXPECTED_LM2

    # Generate several options. Note that this means several sequential generation requests
    n_options = 2
    with assistant():
        options = []
        for i in range(n_options):
            lm3 = lm2 + gen(name='option', temperature=1.0, max_tokens=50)
            print(f"{str(lm3)=}")
            print("--")
            nxt_opt = lm3["option"]
            print(f"{nxt_opt=}")
            assert "<|" not in nxt_opt
            options.append(nxt_opt)
            print("------")


    print(f"{str(lm)=}")
    print(f"{str(lm2)=}")
    print("==\n")

    # Have the user request pros and cons
    with user():
        lm2 += f"""\
        I want to {goal}
        Can you please comment on the pros and cons of each of the following options, and then pick the best option?
        ---
        """
        for i, opt in enumerate(options):
            lm2 += f"Option {i}: {opt}\n"
        lm2 += f"""\
        ---
        Please discuss each option very briefly (one line for pros, one for cons), and end by saying Best=X, where X is the number of the best option."""

    # Get the pros and cons from the model
    with assistant():
        lm2 += gen(name='prosandcons', temperature=0.0, max_tokens=600, stop="Best=") + "Best=" + gen("best", regex="[0-9]+")
    
    return lm



def test_plan_for_goal(chat_model: models.Model):
    lm = chat_model + plan_for_goal("Add numbers")
    assert lm is not None
