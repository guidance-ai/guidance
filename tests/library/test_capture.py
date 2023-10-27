from guidance import select, capture, models
from ..utils import get_model

def test_capture():
    model = models.LocalMock()
    model += 'This is' + capture(select(options=['bad', 'quite bad']), name="my_var")
    assert model["my_var"] in ["bad", "quite bad"]