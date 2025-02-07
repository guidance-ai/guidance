import os
import pathlib
import random
import tempfile
import time
import uuid
import pytest
import requests
import importlib

from guidance import models

# The naming convention for the keys is "<loader>_<model>_<host>" where:
# - 'loader' is 'transformers' or 'llamacpp'
# - 'model' contains relevant information about the model itself
# - 'host' is 'cpu' or 'gpu' as appropriate

AVAILABLE_MODELS = {}

# GEMMA 2
AVAILABLE_MODELS["llamacpp_gemma2_9b_cpu"] = lambda: dict(
    # Note that this model requires an appropriate
    # HF_TOKEN environment variable
    name="huggingface_hubllama:bartowski/gemma-2-9b-it-GGUF:gemma-2-9b-it-IQ2_XS.gguf",
    kwargs={"verbose": True, "n_ctx": 4096},
)
AVAILABLE_MODELS["transformers_gemma2_9b_cpu"] = lambda: dict(
    # Note that this model requires an appropriate
    # HF_TOKEN environment variable
    name="transformers:google/gemma-2-9b-it",
    kwargs={
        "quantization_config": importlib.import_module("transformers").BitsAndBytesConfig(load_in_8bit=True),
    },
)
AVAILABLE_MODELS["transformers_gemma2_9b_gpu"] = lambda: dict(
    # Note that this model requires an appropriate
    # HF_TOKEN environment variable
    name="transformers:google/gemma-2-9b-it",
    kwargs={
        "device_map": "cuda:0",
        "quantization_config": importlib.import_module("transformers").BitsAndBytesConfig(load_in_4bit=True),
    },
)

# GPT 2
AVAILABLE_MODELS["transformers_gpt2_cpu"] = lambda: dict(name="transformers:gpt2", kwargs=dict())
AVAILABLE_MODELS["transformers_gpt2_gpu"] = lambda: dict(name="transformers:gpt2", kwargs={"device_map": "cuda:0"})

# LLAMA 2
AVAILABLE_MODELS["llamacpp_llama2_7b_cpu"] = lambda: dict(
    name="huggingface_hubllama:TheBloke/Llama-2-7B-GGUF:llama-2-7b.Q5_K_M.gguf",
    kwargs={"verbose": True, "n_ctx": 4096},
)
AVAILABLE_MODELS["llamacpp_llama2_7b_gpu"] = lambda: dict(
    name="huggingface_hubllama:TheBloke/Llama-2-7B-GGUF:llama-2-7b.Q5_K_M.gguf",
    kwargs={"verbose": True, "n_gpu_layers": -1, "n_ctx": 4096},
)

# LLAMA 3
AVAILABLE_MODELS["transformers_llama3_8b_cpu"] = lambda: dict(
    # Note that this model requires an appropriate
    # HF_TOKEN environment variable
    name="transformers:meta-llama/Meta-Llama-3-8B-Instruct",
    kwargs={"trust_remote_code": True, "torch_dtype": importlib.import_module("torch").bfloat16},
)
AVAILABLE_MODELS["transformers_llama3_8b_gpu"] = lambda: dict(
    # Note that this model requires an appropriate
    # HF_TOKEN environment variable
    name="transformers:meta-llama/Meta-Llama-3-8B-Instruct",
    kwargs={"trust_remote_code": True, "torch_dtype": importlib.import_module("torch").bfloat16, "device_map": "cuda:0"},
)

# MISTRAL
AVAILABLE_MODELS["transformers_mistral_7b_cpu"] = lambda: dict(
    name="transformers:mistralai/Mistral-7B-v0.1", kwargs=dict()
)
AVAILABLE_MODELS["llamacpp_mistral_7b_cpu"] = lambda: dict(
    name="huggingface_hubllama:TheBloke/Mistral-7B-Instruct-v0.2-GGUF:mistral-7b-instruct-v0.2.Q8_0.gguf",
    kwargs={"verbose": True, "n_ctx": 2048},
)

# PHI 2
AVAILABLE_MODELS["transformers_phi2_cpu"] = lambda: dict(
    name="transformers:microsoft/phi-2", kwargs={"trust_remote_code": True}
)
AVAILABLE_MODELS["transformers_phi2_gpu"] = lambda: dict(
    name="transformers:microsoft/phi-2",
    kwargs={"trust_remote_code": True, "device_map": "cuda:0"},
)

# PHI 3
AVAILABLE_MODELS["transformers_phi3_mini_4k_instruct_cpu"] = lambda: dict(
    name="transformers:microsoft/Phi-3-mini-4k-instruct",
    kwargs={"trust_remote_code": True},
)
AVAILABLE_MODELS["llamacpp_phi3_mini_4k_instruct_cpu"] = lambda: dict(
    name="huggingface_hubllama:microsoft/Phi-3-mini-4k-instruct-gguf:Phi-3-mini-4k-instruct-q4.gguf",
    kwargs={"verbose": True, "n_ctx": 4096},
)
AVAILABLE_MODELS["transformers_phi3_small_8k_instruct_gpu"] = lambda: dict(
    name="transformers:microsoft/Phi-3-small-8k-instruct",
    kwargs={"trust_remote_code": True, "load_in_8bit": True, "device_map": "cuda:0"},
)

# QWEN2DOT5
AVAILABLE_MODELS["transformers_qwen2dot5_0dot5b_cpu"] = lambda: dict(
    name="transformers:Qwen/Qwen2.5-0.5B", kwargs=dict()
)
AVAILABLE_MODELS["transformers_qwen2dot5_0dot5b_gpu"] = lambda: dict(
    name="transformers:Qwen/Qwen2.5-0.5B", kwargs={"device_map": "cuda:0"}
)
AVAILABLE_MODELS["transformers_qwen2dot5_0dot5b_instruct_cpu"] = lambda: dict(
    name="transformers:Qwen/Qwen2.5-0.5B-Instruct", kwargs=dict()
)
AVAILABLE_MODELS["transformers_qwen2dot5_0dot5b_instruct_gpu"] = lambda: dict(
    name="transformers:Qwen/Qwen2.5-0.5B-Instruct", kwargs={"device_map": "cuda:0"}
)


# Ensure that asserts from tests/utils.py are rewritten by pytest to show helpful messages
pytest.register_assert_rewrite("tests.utils")


from .utils import get_model

SELECTED_MODEL_ENV_VARIABLE = "GUIDANCE_SELECTED_MODEL"

def pytest_addoption(parser):
    default_model = os.getenv(SELECTED_MODEL_ENV_VARIABLE, "transformers_gpt2_cpu")
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
    model_info = AVAILABLE_MODELS[selected_model_name]()
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
        "llamacpp_llama2_7b_cpu",
        "llamacpp_llama2_7b_gpu",
        "llamacpp_gemma2_9b_cpu",
        "llamacpp_phi3_mini_4k_instruct_cpu",
        "llamacpp_mistral_7b_cpu",
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
