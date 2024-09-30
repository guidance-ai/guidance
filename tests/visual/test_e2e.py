""" Various E2E tests.
"""
# TODO(nopdive): Refactor into appropriate unit / integration / E2E. Hook into rest of test system.
import pytest


def test_state_handler():
    from guidance import system, user, gen, models
    m0 = models.Mock()

    with system():
        m1 = m0 + "You are responsible for autocompleting a sentence."
    with user():
        m2 = m1 + "Roses are red and " + gen(name="suffix", regex='[A-Za-z]{2,5}', max_tokens=5)

    assert m2['suffix'] is not None



@pytest.mark.skip("Testing for visual module. May be removed later.")
def test_tooling():
    from typing import Optional
    import torch

    import guidance
    from guidance.chat import ChatTemplate

    class QWen2_ChatTemplate(ChatTemplate):
        def get_role_start(self, role_name: str, **kwargs):
            if role_name == "system":
                return "<|im_start|>system\n"
            elif role_name == "user":
                return "<|im_start|>user\n"
            else:
                return "<|im_start|>assistant\n"

        def get_role_end(self, role_name: Optional[str] = None):
            return "<|im_end|>"

    model = "Qwen/Qwen2-0.5B-Instruct"
    lm = guidance.models.Transformers(
        model,
        device_map="auto",
        trust_remote_code=True,
        chat_template=QWen2_ChatTemplate,
        # chat_template=LLAMA3_1_ChatTemplate,
        # attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )

    base_lm = lm

    @guidance
    def add(lm, input1, input2):
        lm += f" = {int(input1) + int(input2)}"
        return lm

    lm = (
            base_lm
            + """\
    1 + 1 = add(1, 1) = 2
    3 + 5 = add(3, 5) = 8
    11 + 9 ="""
    )
    lm = lm + guidance.gen(max_tokens=30, tools=[add])

    assert lm is not None