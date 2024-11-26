import pytest
import tempfile

from guidance.bench._powerlift import retrieve_langchain
from pathlib import Path

def test_retrieve_langchain_err(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    with pytest.raises(ValueError):
        gen = retrieve_langchain()
        _ = list(gen)

@pytest.mark.skip("Waiting on CI upgrades. Need access to env var LANGCHAIN_API_KEY.")
def test_retrieve_langchain_basic():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Run once
        first_results = list(retrieve_langchain(cache_dir=tmp_dir))
        langchain_cache_path = Path(tmp_dir, "langchain")
        assert Path.exists(langchain_cache_path)

        # Run another time to trigger the cache
        second_results = list(retrieve_langchain(cache_dir=tmp_dir))
        for first, second in zip(first_results, second_results):
            assert first.inputs.equals(second.inputs)