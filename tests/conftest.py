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

def pytest_addoption(parser):
    SELECTED_MODEL_ENV_VARIABLE = "GUIDANCE_SELECTED_MODEL"
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

    The naming convention for the keys is "<loader>_<model>_<host>" where:
    - 'loader' is 'transformers' or 'llamacpp'
    - 'model' contains relevant information about the model itself
    - 'host' is 'cpu' or 'gpu' as appropriate
    """

    AVAILABLE_MODELS = {}

    # GEMMA 2
    # Note that this model requires an appropriate HF_TOKEN environment variable
    AVAILABLE_MODELS["llamacpp_gemma2_9b_cpu"] = lambda: models.LlamaCpp(
        importlib.import_module("huggingface_hub").hf_hub_download(
            repo_id="bartowski/gemma-2-9b-it-GGUF", filename="gemma-2-9b-it-IQ2_XS.gguf"
        ),
        verbose=True,
        n_ctx=4096,
    )
    # Note that this model requires an appropriate HF_TOKEN environment variable
    AVAILABLE_MODELS["transformers_gemma2_9b_cpu"] = lambda: models.Transformers(
        "google/gemma-2-9b-it",
        quantization_config=importlib.import_module("transformers").BitsAndBytesConfig(
            load_in_8bit=True
        ),
    )
    # Note that this model requires an appropriate HF_TOKEN environment variable
    AVAILABLE_MODELS["transformers_gemma2_9b_gpu"] = lambda: models.Transformers(
        "google/gemma-2-9b-it",
        device_map="cuda:0",
        quantization_config=importlib.import_module("transformers").BitsAndBytesConfig(
            load_in_4bit=True
        ),
    )

    # GPT 2
    AVAILABLE_MODELS["transformers_gpt2_cpu"] = lambda: models.Transformers("gpt2")
    AVAILABLE_MODELS["transformers_gpt2_gpu"] = lambda: models.Transformers(
        "gpt2", device_map="cuda:0"
    )

    # LLAMA 2
    AVAILABLE_MODELS["llamacpp_llama2_7b_cpu"] = lambda: models.LlamaCpp(
        importlib.import_module("huggingface_hub").hf_hub_download(
            repo_id="TheBloke/Llama-2-7B-GGUF", filename="llama-2-7b.Q5_K_M.gguf"
        ),
        verbose=True,
        n_ctx=4096,
    )
    AVAILABLE_MODELS["llamacpp_llama2_7b_gpu"] = lambda: models.LlamaCpp(
        importlib.import_module("huggingface_hub").hf_hub_download(
            repo_id="TheBloke/Llama-2-7B-GGUF", filename="llama-2-7b.Q5_K_M.gguf"
        ),
        verbose=True,
        n_ctx=4096,
        n_gpu_layers=-1,
    )

    # LLAMA 3
    # Note that this model requires an appropriate HF_TOKEN environment variable
    AVAILABLE_MODELS["transformers_llama3_8b_cpu"] = lambda: models.Transformers(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        trust_remote_code=True,
        torch_dtype=importlib.import_module("torch").bfloat16,
    )
    # Note that this model requires an appropriate HF_TOKEN environment variable
    AVAILABLE_MODELS["transformers_llama3_8b_gpu"] = lambda: models.Transformers(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        trust_remote_code=True,
        torch_dtype=importlib.import_module("torch").bfloat16,
        device_map="cuda:0",
    )

    # MISTRAL
    AVAILABLE_MODELS["transformers_mistral_7b_cpu"] = lambda: models.Transformers(
        "mistralai/Mistral-7B-v0.1"
    )
    AVAILABLE_MODELS["llamacpp_mistral_7b_cpu"] = lambda: models.LlamaCpp(
        importlib.import_module("huggingface_hub").hf_hub_download(
            repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
            filename="mistral-7b-instruct-v0.2.Q8_0.gguf",
        ),
        verbose=True,
        n_ctx=2048,
    )

    # PHI 2
    AVAILABLE_MODELS["transformers_phi2_cpu"] = lambda: models.Transformers(
        "microsoft/phi-2", trust_remote_code=True
    )
    AVAILABLE_MODELS["transformers_phi2_gpu"] = lambda: models.Transformers(
        "microsoft/phi-2", trust_remote_code=True, device_map="cuda:0"
    )

    # PHI 3
    AVAILABLE_MODELS["transformers_phi3_mini_4k_instruct_cpu"] = lambda: models.Transformers(
        "microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True
    )
    AVAILABLE_MODELS["llamacpp_phi3_mini_4k_instruct_cpu"] = lambda: models.LlamaCpp(
        importlib.import_module("huggingface_hub").hf_hub_download(
            repo_id="microsoft/Phi-3-mini-4k-instruct-gguf", filename="Phi-3-mini-4k-instruct-q4.gguf"
        ),
        verbose=True,
        n_ctx=4096,
    )
    AVAILABLE_MODELS["transformers_phi3_small_8k_instruct_gpu"] = lambda: models.Transformers(
        "microsoft/Phi-3-small-8k-instruct", trust_remote_code=True, load_in_8bit=True, device_map="cuda:0"
    )

    # QWEN2DOT5
    AVAILABLE_MODELS["transformers_qwen2dot5_0dot5b_cpu"] = lambda: models.Transformers(
        "Qwen/Qwen2.5-0.5B"
    )
    AVAILABLE_MODELS["transformers_qwen2dot5_0dot5b_gpu"] = lambda: models.Transformers(
        "Qwen/Qwen2.5-0.5B", device_map="cuda:0"
    )
    AVAILABLE_MODELS["transformers_qwen2dot5_0dot5b_instruct_cpu"] = lambda: models.Transformers(
        "Qwen/Qwen2.5-0.5B-Instruct"
    )
    AVAILABLE_MODELS["transformers_qwen2dot5_0dot5b_instruct_gpu"] = lambda: models.Transformers(
        "Qwen/Qwen2.5-0.5B-Instruct", device_map="cuda:0"
    )

    return AVAILABLE_MODELS[selected_model_name]()

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
