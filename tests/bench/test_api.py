from guidance.bench._api import bench, AVAILABLE_MODELS
from pathlib import Path
import tempfile
import pytest

@pytest.mark.skip("Waiting on CI upgrades. Need access to env var LANGCHAIN_API_KEY.")
def test_bench():
    # TODO(nopdive): Parameterize models once CI is upgraded.
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "bench.db"
        db_url = f"sqlite:///{db_path}"
        status_df, result_df = bench(db_url, "bench-test", models=AVAILABLE_MODELS[:1], debug_mode=True)

        assert len(status_df) > 0
        assert len(result_df) > 0
        assert (status_df['status'] == 'COMPLETE').all()