from guidance.bench._api import bench
from pathlib import Path

def test_bench():
    db_url = Path(Path.cwd(), "bench.db")
    bench(db_url, "guidance")