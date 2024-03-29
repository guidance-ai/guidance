import random
import time

import pytest

from guidance import models

from .utils import get_model

AVAILABLE_MODELS = {
    "gpt2cpu": dict(name="transformers:gpt2", kwargs=dict()),
    "phi2cpu": dict(
        name="transformers:microsoft/phi-2", kwargs={"trust_remote_code": True}
    ),
    "hfllama7b": dict(
        name="huggingface_hubllama:TheBloke/Llama-2-7B-GGUF:llama-2-7b.Q5_K_M.gguf",
        kwargs={"verbose": True},
    ),
    "transformers_mistral_7b": dict(
        name="transformers:mistralai/Mistral-7B-v0.1", kwargs=dict()
    ),
    "hfllama_mistral_7b": dict(
        name="huggingface_hubllama:TheBloke/Mistral-7B-Instruct-v0.2-GGUF:mistral-7b-instruct-v0.2.Q8_0.gguf",
        kwargs={"verbose": True},
    ),
    "gpt2gpu": dict(name="transformers:gpt2", kwargs={"device_map": "cuda:0"}),
    "phi2gpu": dict(
        name="transformers:microsoft/phi-2",
        kwargs={"trust_remote_code": True, "device_map": "cuda:0"},
    ),
    "hfllama_7b_gpu": dict(
        name="huggingface_hubllama:TheBloke/Llama-2-7B-GGUF:llama-2-7b.Q5_K_M.gguf",
        kwargs={"verbose": True, "n_gpu_layers": -1},
    ),
}


def pytest_addoption(parser):
    parser.addoption(
        "--selected_model",
        action="store",
        default="gpt2cpu",
        type=str,
        choices=AVAILABLE_MODELS.keys(),
    )


@pytest.fixture(scope="session")
def selected_model_name(pytestconfig) -> str:
    return pytestconfig.getoption("selected_model")


@pytest.fixture(scope="session")
def selected_model_info(selected_model_name: str):
    model_info = AVAILABLE_MODELS[selected_model_name]
    return model_info


@pytest.fixture(scope="session")
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
    return model


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
