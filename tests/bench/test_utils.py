from guidance.bench._utils import lib_bench_dir
import tempfile
from pathlib import Path

def test_lib_bench_dir_basic():
    expected_dir = Path.home() / ".guidance-bench"
    actual_dir = lib_bench_dir()

    assert expected_dir == actual_dir
    assert Path.exists(actual_dir)


def test_lib_bench_dir_env_var(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp_dir:
        expected_dir = Path(tmp_dir) / "guidance-bench"
        monkeypatch.setenv("GUIDANCE_BENCH_DIR", expected_dir)

        actual_dir = lib_bench_dir()
        assert expected_dir == actual_dir
        assert Path.exists(actual_dir)