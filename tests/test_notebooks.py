import os
import pathlib
from typing import Any, Dict, Optional
from urllib.parse import urlparse, parse_qs

import papermill as pm
import pytest

BASE_NB_PATH = pathlib.Path("./notebooks").absolute()


def run_notebook(notebook_path: pathlib.Path, params: Optional[Dict[str, Any]] = None):
    assert notebook_path.exists(), f"Checking for: {notebook_path}"
    output_nb = notebook_path.stem + ".papermill_out" + notebook_path.suffix
    output_path = TestTutorials.BASE_TUTORIAL_PATH / output_nb

    # Just make sure nothing throws an exception
    pm.execute_notebook(
        input_path=notebook_path, output_path=output_path, parameters=params
    )


class TestTutorials:
    BASE_TUTORIAL_PATH = BASE_NB_PATH / "tutorials"

    @pytest.mark.needs_credentials
    def test_chat(self, rate_limiter):
        azureai_endpoint = os.getenv("AZUREAI_CHAT_ENDPOINT", None)

        parsed_url = urlparse(azureai_endpoint)
        parsed_query = parse_qs(parsed_url.query)
        azureai_deployment = pathlib.Path(parsed_url.path).parts[3]
        version = parsed_query["api-version"]
        min_azureai_endpoint = f"{parsed_url.scheme}://{parsed_url.netloc}"

        os.environ["AZUREAI_CHAT_BASE_ENDPOINT"] = min_azureai_endpoint
        os.environ["AZUREAI_CHAT_API_VERSION"] = version[0]
        os.environ["AZUREAI_CHAT_DEPLOYMENT"] = azureai_deployment
        nb_path = TestTutorials.BASE_TUTORIAL_PATH / "chat.ipynb"
        run_notebook(nb_path, params=dict(call_delay_secs=rate_limiter))

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
        azureai_endpoint = os.getenv("AZUREAI_CHAT_ENDPOINT", None)

        parsed_url = urlparse(azureai_endpoint)
        parsed_query = parse_qs(parsed_url.query)
        azureai_deployment = pathlib.Path(parsed_url.path).parts[3]
        version = parsed_query["api-version"][0]
        min_azureai_endpoint = f"{parsed_url.scheme}://{parsed_url.netloc}"

        os.environ["AZUREAI_CHAT_BASE_ENDPOINT"] = min_azureai_endpoint
        os.environ["AZUREAI_CHAT_API_VERSION"] = version
        os.environ["AZUREAI_CHAT_DEPLOYMENT"] = azureai_deployment

        nb_path = TestModels.BASE_MODEL_PATH / "AzureOpenAI.ipynb"
        run_notebook(nb_path, params=dict(call_delay_secs=rate_limiter))


class TestArtOfPromptDesign:
    BASE_APD_PATH = BASE_NB_PATH / "art_of_prompt_design"

    @pytest.mark.skip(reason="Having trouble running")
    @pytest.mark.use_gpu
    def test_prompt_boundaries_and_token_healing(self):
        nb_path = (
            TestArtOfPromptDesign.BASE_APD_PATH
            / "prompt_boundaries_and_token_healing.ipynb"
        )
        run_notebook(nb_path)

    @pytest.mark.use_gpu
    def test_react(self, selected_model_name):
        if selected_model_name in ["phi2gpu"]:
            # I don't know why; it doesn't make sense, but
            msg = (
                f"react notebook disagrees with {selected_model_name}; reasons obscure"
            )
            pytest.skip(msg)
        nb_path = TestArtOfPromptDesign.BASE_APD_PATH / "react.ipynb"
        run_notebook(nb_path)

    @pytest.mark.use_gpu
    @pytest.mark.needs_credentials
    def test_use_clear_syntax(self, rate_limiter):
        azureai_endpoint = os.getenv("AZUREAI_CHAT_ENDPOINT", None)

        parsed_url = urlparse(azureai_endpoint)
        parsed_query = parse_qs(parsed_url.query)
        azureai_deployment = pathlib.Path(parsed_url.path).parts[3]
        version = parsed_query["api-version"]
        min_azureai_endpoint = f"{parsed_url.scheme}://{parsed_url.netloc}"

        os.environ["AZUREAI_CHAT_BASE_ENDPOINT"] = min_azureai_endpoint
        os.environ["AZUREAI_CHAT_API_VERSION"] = version[0]
        os.environ["AZUREAI_CHAT_DEPLOYMENT"] = azureai_deployment
        nb_path = TestArtOfPromptDesign.BASE_APD_PATH / "use_clear_syntax.ipynb"
        run_notebook(nb_path, params=dict(call_delay_secs=rate_limiter))
