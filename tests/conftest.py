import pytest

from guidance import models

from .utils import get_model

AVAILABLE_MODELS = {
    "gpt2cpu": dict(name="transformers:gpt2", kwargs=dict()),
    "phi2cpu": dict(
        name="transformers:microsoft/phi-2", kwargs={"trust_remote_code": True}
    ),
    "gpt2gpu": dict(name="transformers:gpt2", kwargs={"device_map": "cuda:0"}),
    "phi2gpu": dict(
        name="transformers:microsoft/phi-2",
        kwargs={"trust_remote_code": True, "device_map": "cuda:0"},
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
def selected_model(pytestconfig) -> models.Model:
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
    model_key = pytestconfig.getoption("selected_model")

    model_info = AVAILABLE_MODELS[model_key]

    model = get_model(model_info["name"], **(model_info["kwargs"]))
    return model
