import guidance
import pytest

def test_basic():
    """ Test the basic behavior of the MPTChat model.
    """

    # skip if no GPU
    import torch
    if torch.cuda.device_count() == 0:
        pytest.skip("No GPU, so skipping large model test.")

    # just make sure it runs
    llm = guidance.llms.transformers.MPTChat('mosaicml/mpt-7b-chat', device=0)
    out = guidance("""
{{#system~}}
You are an assistant.
{{~/system}}

{{#user~}}
How tall is the Eiffel Tower?
{{~/user}}

{{#assistant~}}
{{gen 'answer' max_tokens=10}}
{{~/assistant}}
""", llm=llm)()
    assert len(out["answer"]) > 0