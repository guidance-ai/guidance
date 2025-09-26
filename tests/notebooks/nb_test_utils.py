import pathlib
from typing import Any

import papermill as pm

BASE_NB_PATH = pathlib.Path("./notebooks").absolute()


def run_notebook(notebook_path: pathlib.Path, params: dict[str, Any] | None = None):
    assert notebook_path.exists(), f"Checking for: {notebook_path}"
    output_nb = notebook_path.stem + ".papermill_out" + notebook_path.suffix
    output_path = notebook_path.parent / output_nb

    # Just make sure nothing throws an exception
    pm.execute_notebook(input_path=notebook_path, output_path=output_path, parameters=params)
