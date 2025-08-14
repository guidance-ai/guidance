from guidance import models


def test_chat_completion(selected_model: models.Model):
    engine = selected_model._interpreter.engine

    grammar = """%llguidance {}

start: "I choose the letter " myletter
myletter[capture="value"]: LETTER_CHOICE
LETTER_CHOICE: "A" | "B"
"""

    messages = [{"role": "user", "content": "Give me a capital letter"}]

    result, captures = engine.chat_completion(messages, grammar)

    assert len(captures) == 1
    assert captures["value"] in ["A", "B"]
    if captures["value"] == "A":
        assert result == "I choose the letter A"
    elif captures["value"] == "B":
        assert result == "I choose the letter B"
