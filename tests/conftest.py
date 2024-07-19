import os
import pathlib
import random
import tempfile
import time
import uuid

import pytest
import requests
import torch
import transformers

from guidance import models

# Ensure that asserts from tests/utils.py are rewritten by pytest to show helpful messages
pytest.register_assert_rewrite("tests.utils")

from .utils import get_model

SELECTED_MODEL_ENV_VARIABLE = "GUIDANCE_SELECTED_MODEL"

AVAILABLE_MODELS = {
    "gpt2cpu": dict(name="transformers:gpt2", kwargs=dict()),
    "phi2cpu": dict(name="transformers:microsoft/phi-2", kwargs={"trust_remote_code": True}),
    "azure_guidance": dict(
        name="azure_guidance:",
        kwargs={},
    ),
    "transformers_phi3cpu_mini_4k_instruct": dict(
        name="transformers:microsoft/Phi-3-mini-4k-instruct",
        kwargs={"trust_remote_code": True},
    ),
    "transformers_llama3cpu_8b": dict(
        # Note that this model requires an appropriate
        # HF_TOKEN environment variable
        name="transformers:meta-llama/Meta-Llama-3-8B-Instruct",
        kwargs={"trust_remote_code": True, "torch_dtype": torch.bfloat16},
    ),
    "transformers_llama3gpu_8b": dict(
        # Note that this model requires an appropriate
        # HF_TOKEN environment variable
        name="transformers:meta-llama/Meta-Llama-3-8B-Instruct",
        kwargs={"trust_remote_code": True, "torch_dtype": torch.bfloat16, "device_map": "cuda:0"},
    ),
    "hfllama_gemma2cpu_9b": dict(
        # Note that this model requires an appropriate
        # HF_TOKEN environment variable
        name="huggingface_hubllama:bartowski/gemma-2-9b-it-GGUF:gemma-2-9b-it-IQ2_XS.gguf",
        kwargs={"verbose": True, "n_ctx": 4096},
    ),
    "transformers_gemma2cpu_9b": dict(
        # Note that this model requires an appropriate
        # HF_TOKEN environment variable
        name="transformers:google/gemma-2-9b-it",
        kwargs={},
    ),
    "transformers_gemma2gpu_9b": dict(
        # Note that this model requires an appropriate
        # HF_TOKEN environment variable
        name="transformers:google/gemma-2-9b-it",
        kwargs={
            "device_map": "cuda:0",
            "quantization_config": transformers.BitsAndBytesConfig(load_in_4bit=True),
        },
    ),
    "hfllama_phi3cpu_mini_4k_instruct": dict(
        name="huggingface_hubllama:microsoft/Phi-3-mini-4k-instruct-gguf:Phi-3-mini-4k-instruct-q4.gguf",
        kwargs={"verbose": True, "n_ctx": 4096},
    ),
    "hfllama7b": dict(
        name="huggingface_hubllama:TheBloke/Llama-2-7B-GGUF:llama-2-7b.Q5_K_M.gguf",
        kwargs={"verbose": True, "n_ctx": 4096},
    ),
    "transformers_phi3_small_8k_instruct": dict(
        name="transformers:microsoft/Phi-3-small-8k-instruct",
        kwargs={"trust_remote_code": True, "load_in_8bit": True, "device_map": "cuda:0"},
    ),
    "transformers_mistral_7b": dict(name="transformers:mistralai/Mistral-7B-v0.1", kwargs=dict()),
    "hfllama_mistral_7b": dict(
        name="huggingface_hubllama:TheBloke/Mistral-7B-Instruct-v0.2-GGUF:mistral-7b-instruct-v0.2.Q8_0.gguf",
        kwargs={"verbose": True, "n_ctx": 2048},
    ),
    "gpt2gpu": dict(name="transformers:gpt2", kwargs={"device_map": "cuda:0"}),
    "phi2gpu": dict(
        name="transformers:microsoft/phi-2",
        kwargs={"trust_remote_code": True, "device_map": "cuda:0"},
    ),
    "hfllama_7b_gpu": dict(
        name="huggingface_hubllama:TheBloke/Llama-2-7B-GGUF:llama-2-7b.Q5_K_M.gguf",
        kwargs={"verbose": True, "n_gpu_layers": -1, "n_ctx": 4096},
    ),
}


def pytest_addoption(parser):
    default_model = os.getenv(SELECTED_MODEL_ENV_VARIABLE, "gpt2cpu")
    parser.addoption(
        "--selected_model",
        action="store",
        default=default_model,
        type=str,
        choices=AVAILABLE_MODELS.keys(),
        help=f"LLM to load when needed. Set default via environment variable {SELECTED_MODEL_ENV_VARIABLE}",
    )


@pytest.fixture(scope="session")
def selected_model_name(pytestconfig) -> str:
    return pytestconfig.getoption("selected_model")


@pytest.fixture(scope="session")
def selected_model_info(selected_model_name: str):
    model_info = AVAILABLE_MODELS[selected_model_name]
    return model_info


@pytest.fixture(scope="module")
def selected_model(selected_model_info: str) -> models.Model:
    """Get a concrete model for tests

    This fixture is for tests which are supposed
    to run against any LLM supported by guidance.
    Rather than loading a model themselves, each test
    can just use this fixture, and get a 'live'
    model.

    When running the tests, the model used is
    controlled by the '--selected_model' command
    line argument to pytest.
    """
    model = get_model(selected_model_info["name"], **(selected_model_info["kwargs"]))
    assert model is not None
    return model


@pytest.fixture(scope="module")
def llamacpp_model(selected_model, selected_model_name):
    if selected_model_name in [
        "hfllama7b",
        "hfllama_7b_gpu",
        "hfllama_gemma2cpu_9b",
        "hfllama_phi3cpu_mini_4k_instruct",
        "hfllama_mistral_7b",
    ]:
        return selected_model
    else:
        pytest.skip("Requires Llama-Cpp model")


@pytest.fixture(scope="function")
def rate_limiter() -> int:
    """Limit test execution rate

    Any test using this fixture will have a
    random delay inserted before the test runs.
    It can be used as a crude rate limiter for
    tests which call external APIs
    """
    delay_secs = random.randint(10, 30)
    time.sleep(delay_secs)
    return delay_secs


@pytest.fixture(scope="session")
def remote_image_url():
    return "https://picsum.photos/300/200"


@pytest.fixture(scope="session")
def local_image_path(remote_image_url):
    with tempfile.TemporaryDirectory() as temp_dir:
        td = pathlib.Path(temp_dir)
        filename = f"{str(uuid.uuid4())}.jpg"
        with open(td / filename, "wb") as file:
            response = requests.get(remote_image_url)
            file.write(response.content)
        assert (td / filename).exists()
        yield td / filename


@pytest.fixture(scope="session")
def local_image_bytes(local_image_path):
    with open(local_image_path, "rb") as f:
        return f.read()
