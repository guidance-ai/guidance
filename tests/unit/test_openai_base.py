from guidance._grammar import gen
from guidance.models._openai_base import OpenAIRuleMixin


def test_temperature_zero_is_passed_through(monkeypatch):
    """temperature=0.0 is the canonical deterministic/greedy setting and must
    not be dropped by a truthiness check (it is falsy in Python).
    """
    captured: dict = {}

    def fake_run(self, node, **kwargs):
        captured.update(kwargs)
        return iter([])

    monkeypatch.setattr(OpenAIRuleMixin, "run", fake_run)
    interp = OpenAIRuleMixin.__new__(OpenAIRuleMixin)
    node = gen(temperature=0.0)
    list(interp.rule(node))
    assert captured.get("temperature") == 0.0


def test_temperature_none_not_passed(monkeypatch):
    captured: dict = {}

    def fake_run(self, node, **kwargs):
        captured.update(kwargs)
        return iter([])

    monkeypatch.setattr(OpenAIRuleMixin, "run", fake_run)
    interp = OpenAIRuleMixin.__new__(OpenAIRuleMixin)
    node = gen()
    list(interp.rule(node))
    assert "temperature" not in captured
