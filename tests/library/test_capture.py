from guidance import select, capture
from ..utils import get_model

def test_capture():
    model = get_model("transformers:gpt2")
    model += 'This is' + capture(select(values=['bad', 'quite bad']), name="my_var")
    assert model["my_var"] in ["bad", "quite bad"]