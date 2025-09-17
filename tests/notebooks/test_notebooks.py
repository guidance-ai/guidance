import logging
import pathlib
from typing import Any, Optional

import papermill as pm
import pytest

from ..utils import slowdown

BASE_NB_PATH = pathlib.Path("./notebooks").absolute()


def run_notebook(notebook_path: pathlib.Path, params: dict[str, Any] | None = None):
    assert notebook_path.exists(), f"Checking for: {notebook_path}"
    output_nb = notebook_path.stem + ".papermill_out" + notebook_path.suffix
    output_path = notebook_path.parent / output_nb

    # Just make sure nothing throws an exception
    pm.execute_notebook(input_path=notebook_path, output_path=output_path, parameters=params)


def test_guarantee_valid_syntax():
    call_delay_secs = slowdown()
    nb_path = BASE_NB_PATH / "guaranteeing_valid_syntax.ipynb"
    run_notebook(
        nb_path,
        params=dict(call_delay_secs=call_delay_secs, requested_log_level=logging.DEBUG),
    )


def test_anachronism():
    call_delay_secs = slowdown()
    # Note, this might fail if the dataset can't be fetched from the Bigbench git repo
    nb_path = BASE_NB_PATH / "anachronism.ipynb"
    run_notebook(
        nb_path,
        params=dict(call_delay_secs=call_delay_secs, requested_log_level=logging.DEBUG),
    )


def test_engine_chat_completion():
    call_delay_secs = slowdown()
    nb_path = BASE_NB_PATH / "engine_chat_completion.ipynb"
    run_notebook(
        nb_path,
        params=dict(call_delay_secs=call_delay_secs, requested_log_level=logging.DEBUG),
    )


class TestTutorials:
    BASE_TUTORIAL_PATH = BASE_NB_PATH / "tutorials"

    @pytest.mark.skip("Appear to have multiple overlapping problems")
    def test_chat(self):
        call_delay_secs = slowdown()

        nb_path = TestTutorials.BASE_TUTORIAL_PATH / "chat.ipynb"
        run_notebook(
            nb_path,
            params=dict(call_delay_secs=call_delay_secs, requested_log_level=logging.DEBUG),
        )

    def test_regex_constraints(self):
        nb_path = TestTutorials.BASE_TUTORIAL_PATH / "regex_constraints.ipynb"
        run_notebook(nb_path)

    def test_token_healing(self):
        nb_path = TestTutorials.BASE_TUTORIAL_PATH / "token_healing.ipynb"
        run_notebook(nb_path)

    def test_tool_calling(self):
        nb_path = TestTutorials.BASE_TUTORIAL_PATH / "tool_calling.ipynb"
        run_notebook(nb_path)


class TestModels:
    BASE_MODEL_PATH = BASE_NB_PATH / "api_examples" / "models"

    def test_azure_openai(self):
        call_delay_secs = slowdown()
        nb_path = TestModels.BASE_MODEL_PATH / "AzureOpenAI.ipynb"
        run_notebook(nb_path, params=dict(call_delay_secs=call_delay_secs))

    def test_openai(self):
        nb_path = TestModels.BASE_MODEL_PATH / "OpenAI.ipynb"
        run_notebook(nb_path)


class TestArtOfPromptDesign:
    BASE_APD_PATH = BASE_NB_PATH / "art_of_prompt_design"

    @pytest.mark.skip(reason="Having trouble running")
    def test_prompt_boundaries_and_token_healing(self):
        nb_path = TestArtOfPromptDesign.BASE_APD_PATH / "prompt_boundaries_and_token_healing.ipynb"
        run_notebook(nb_path)

    @pytest.mark.skip("Tool calling unstable")
    def test_react(self, selected_model_name):
        if selected_model_name in ["transformers_phi2_gpu"]:
            # I don't know why; it doesn't make sense, but
            msg = f"react notebook disagrees with {selected_model_name}; reasons obscure"
            pytest.skip(msg)
        nb_path = TestArtOfPromptDesign.BASE_APD_PATH / "react.ipynb"
        run_notebook(nb_path)

    def test_use_clear_syntax(self):
        call_delay_secs = slowdown()

        nb_path = TestArtOfPromptDesign.BASE_APD_PATH / "use_clear_syntax.ipynb"
        run_notebook(nb_path, params=dict(call_delay_secs=call_delay_secs))
