from guidance import select
from ..utils import get_model

def test_reset_pos():
    model = get_model("transformers:gpt2")
    model += 'This is' + select(values=['bad', 'quite bad'])
    assert str(model) in ["This isbad", "This isquite bad"]