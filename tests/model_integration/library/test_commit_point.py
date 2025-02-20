import pytest
from guidance import Tool, capture, commit_point, models, select, string

@pytest.mark.xfail(reason="Commit points are not supported")
def test_commit_point(selected_model: models.Model):
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
