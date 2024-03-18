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
        kwargs=dict(),
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
def selected_model(selected_model_name: str) -> models.Model:
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
    model_info = AVAILABLE_MODELS[selected_model_name]

    model = get_model(model_info["name"], **(model_info["kwargs"]))
    return model
