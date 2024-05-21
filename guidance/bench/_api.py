"""User facing API for benchmarking."""

from typing import List, Optional, Tuple, Union
from pathlib import Path
from guidance.bench._utils import lib_bench_dir

"""Available models to run benchmark against."""
AVAILABLE_MODELS = [
    "guidance-mistral-7b-instruct",
    "base-mistral-7b-instruct",
    "guidance-phi-3-mini-4k-instruct",
    "base-phi-3-mini-4k-instruct",
    "guidance-llama2-7b-32k-instruct",
    "base-llama2-7b-32k-instruct",
]


def bench(
    db_url: str,
    experiment_name: str,
    models: List[str] = AVAILABLE_MODELS,
    force_recreate: bool = False,
    timeout: int = 3600,
    cache_dir: Union[str, Path] = lib_bench_dir() / "cache",
    debug_mode: bool = False,
) -> Tuple[object, object]:
    """Benchmarks guidance against preset tasks.

    This runs on a single machine, one trial at a time.
    To run this the first time you will need API_LANGCHAIN_KEY set as an environment variable.

    Args:
        db_url (str): Database connection string.
        experiment_name (str): Name of experiment to create / run.
        models (List[str], optional): Models to benchmark. Defaults to AVAILABLE_MODELS.
        force_recreate (bool, optional): Recreate the database before benchmarking. Defaults to False.
        timeout (int, optional): Max execution time per trial. Defaults to 3600.
        cache_dir (Union[str, Path], optional): Cache to store external datasets. Defaults to lib_bench_dir() / "cache".
        debug_mode (bool): Set this when you require a debugger to step line by line in the trial_runner.

    Returns:
        Tuple[object, object]: (status, results) data frames where status relates to trials, results are wide form aggregates of each model.
    """
    from guidance.bench._powerlift import bench as inner_bench

    status_df, result_df = inner_bench(
        db_url, experiment_name, models, force_recreate, timeout, cache_dir, debug_mode
    )
    return status_df, result_df
