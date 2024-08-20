import pytest
from guidance import commit_point, one_or_more, byte_range, models

@pytest.mark.xfail(reason="Commit points are not supported")
def test_hidden():
    model = models.Mock()
    model += " one" + commit_point(" two", hidden=True) + " three"
    assert str(model) == " one three"
