"""User facing API for benchmarking."""

from typing import Tuple
import pandas as pd


def prepare_powerlift_store(store):
    pass


def bench(db_url: str, experiment_name: str, force_recreate: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pass