from guidance import select, models
from ..utils import get_model

def test_reset_pos():
    model = models.LocalMock()
    model += 'This is' + select(values=['bad', 'quite bad'])
    assert str(model) in ["This isbad", "This isquite bad"]