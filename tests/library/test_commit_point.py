from guidance import commit_point, models, Tool, select, capture, string


def test_hidden():
    model = models.Mock()
    model += " one" + commit_point(" two", hidden=True) + " three"
    assert str(model) == " one three"


def test_commit_point(selected_model):
    lm = selected_model
    tools = [Tool(callable=lambda x: x)]
    stop_pattern = select([string(lm.engine.tokenizer.eos_token)])
    gen_grammar = select(
        [commit_point(stop_pattern, hidden=True)]
        + [
            capture(commit_point(x.call_grammar, hidden=True), name=f"tool{i}")
            for i, x in enumerate(tools)
        ]
    )
