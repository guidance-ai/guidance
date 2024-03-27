import pathlib

from typing import Any, Dict, Optional

import papermill as pm
import pytest

BASE_NB_PATH = pathlib.Path("./notebooks").absolute()


def run_notebook(notebook_path: pathlib.Path, params: Optional[Dict[str, Any]] = None):
    output_nb = notebook_path.stem + ".papermill_out" + notebook_path.suffix
    output_path = TestTutorials.BASE_TUTORIAL_PATH / output_nb

    # Just make sure nothing throws an exception
    pm.execute_notebook(
        input_path=notebook_path, output_path=output_path, parameters=params
    )


class TestTutorials:
    BASE_TUTORIAL_PATH = BASE_NB_PATH / "tutorials"

    def test_regex_constraints(self):
        nb_path = TestTutorials.BASE_TUTORIAL_PATH / "regex_constraints.ipynb"
        run_notebook(nb_path)

    def test_token_healing(self):
        nb_path = TestTutorials.BASE_TUTORIAL_PATH / "token_healing.ipynb"
        run_notebook(nb_path)


class TestModels:
    BASE_MODEL_PATH = BASE_NB_PATH / "api_examples" / "models"

    @pytest.mark.needs_credentials
    def test_azure_openai(self, rate_limiter):
        nb_path = TestModels.BASE_MODEL_PATH / "AzureOpenAI.ipynb"
        run_notebook(nb_path, params=dict(call_delay_secs=rate_limiter))
