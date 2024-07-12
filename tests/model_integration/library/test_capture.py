import guidance

def test_capture_within_role(model_with_role_tags: guidance.models.Model):
    lm = model_with_role_tags
    test_text = "This is some text in a role."
    with guidance.user():
        lm += guidance.capture(test_text, "test")
    assert lm["test"] == test_text
