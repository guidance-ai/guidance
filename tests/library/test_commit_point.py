from guidance import commit_point, models

def test_hidden():
    model = models.LocalMock()
    model += " one" + commit_point(" two", hidden=True) + " three"
    assert str(model) == " one three"