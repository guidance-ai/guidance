import os

import pytest

from guidance import models


def pytest_addoption(parser):
    SELECTED_MODEL_ENV_VARIABLE = "GUIDANCE_SELECTED_MODEL"
    default_model = os.getenv(SELECTED_MODEL_ENV_VARIABLE, "transformers_gpt2_cpu")
    parser.addoption(
        "--selected_model",
        action="store",
        default=default_model,
        type=str,
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

    # GEMMA 2
    if selected_model_name == "llamacpp_gemma2_9b_cpu":
        # Note that this model requires an appropriate HF_TOKEN environment variable
        from huggingface_hub import hf_hub_download

        return models.LlamaCpp(
            hf_hub_download(repo_id="bartowski/gemma-2-9b-it-GGUF", filename="gemma-2-9b-it-IQ2_XS.gguf"),
            verbose=True,
            n_ctx=4096,
        )
    if selected_model_name == "transformers_gemma2_9b_cpu":
        # Note that this model requires an appropriate HF_TOKEN environment variable
        from transformers import BitsAndBytesConfig

        return models.Transformers(
            "google/gemma-2-9b-it",
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        )
    if selected_model_name == "transformers_gemma2_9b_gpu":
        # Note that this model requires an appropriate HF_TOKEN environment variable
        from transformers import BitsAndBytesConfig

        return models.Transformers(
            "google/gemma-2-9b-it",
            device_map="cuda:0",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        )

    # GPT 2
    if selected_model_name == "transformers_gpt2_cpu":
        return models.Transformers("gpt2")
    if selected_model_name == "transformers_gpt2_gpu":
        return models.Transformers("gpt2", device_map="cuda:0")

    # LLAMA 2
    if selected_model_name == "llamacpp_llama2_7b_cpu":
        from huggingface_hub import hf_hub_download

        return models.LlamaCpp(
            hf_hub_download(repo_id="TheBloke/Llama-2-7B-GGUF", filename="llama-2-7b.Q5_K_M.gguf"),
            verbose=True,
            n_ctx=4096,
        )
    if selected_model_name == "llamacpp_llama2_7b_gpu":
        from huggingface_hub import hf_hub_download

        return models.LlamaCpp(
            hf_hub_download(repo_id="TheBloke/Llama-2-7B-GGUF", filename="llama-2-7b.Q5_K_M.gguf"),
            verbose=True,
            n_ctx=4096,
            n_gpu_layers=-1,
        )

    # LLAMA 3
    if selected_model_name == "transformers_llama3_8b_cpu":
        # Note that this model requires an appropriate HF_TOKEN environment variable
        from torch import bfloat16

        return models.Transformers(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            trust_remote_code=True,
            torch_dtype=bfloat16,
        )
    if selected_model_name == "transformers_llama3_8b_gpu":
        # Note that this model requires an appropriate HF_TOKEN environment variable
        from torch import bfloat16

        return models.Transformers(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            trust_remote_code=True,
            torch_dtype=bfloat16,
            device_map="cuda:0",
        )
    # Llama 3.2
    if selected_model_name == "llamacpp_llama3.2_3b_cpu":
        from huggingface_hub import hf_hub_download

        return models.LlamaCpp(
            hf_hub_download(
                repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
                filename="Llama-3.2-3B-Instruct-Q6_K_L.gguf",
            ),
            verbose=True,
            n_ctx=4096,
        )

    # MISTRAL
    if selected_model_name == "transformers_mistral_7b_cpu":
        return models.Transformers("mistralai/Mistral-7B-v0.1")
    if selected_model_name == "llamacpp_mistral_7b_cpu":
        from huggingface_hub import hf_hub_download

        return models.LlamaCpp(
            hf_hub_download(
                repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
                filename="mistral-7b-instruct-v0.2.Q8_0.gguf",
            ),
            verbose=True,
            n_ctx=2048,
        )

    # PHI 2
    if selected_model_name == "transformers_phi2_cpu":
        return models.Transformers("microsoft/phi-2", trust_remote_code=True)
    if selected_model_name == "transformers_phi2_gpu":
        return models.Transformers("microsoft/phi-2", trust_remote_code=True, device_map="cuda:0")

    # PHI 3
    if selected_model_name == "transformers_phi3_mini_4k_instruct_cpu":
        return models.Transformers("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True)
    if selected_model_name == "llamacpp_phi3_mini_4k_instruct_cpu":
        from huggingface_hub import hf_hub_download

        return models.LlamaCpp(
            hf_hub_download(
                repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
                filename="Phi-3-mini-4k-instruct-q4.gguf",
            ),
            verbose=True,
            n_ctx=4096,
        )
    if selected_model_name == "transformers_phi3_small_8k_instruct_gpu":
        return models.Transformers(
            "microsoft/Phi-3-small-8k-instruct",
            trust_remote_code=True,
            load_in_8bit=True,
            device_map="cuda:0",
        )

    # PHI-4
    if selected_model_name == "transformers_phi4_mini_cpu":
        return models.Transformers("microsoft/Phi-4-mini-instruct", trust_remote_code=True)
    if selected_model_name == "transformers_phi4_mini_gpu":
        from torch import bfloat16

        return models.Transformers(
            "microsoft/Phi-4-mini-instruct",
            trust_remote_code=True,
            device_map="cuda:0",
            torch_dtype=bfloat16,
        )

    # QWEN2DOT5
    if selected_model_name == "transformers_qwen2dot5_0dot5b_cpu":
        return models.Transformers("Qwen/Qwen2.5-0.5B")
    if selected_model_name == "transformers_qwen2dot5_0dot5b_gpu":
        return models.Transformers("Qwen/Qwen2.5-0.5B", device_map="cuda:0")
    if selected_model_name == "transformers_qwen2dot5_0dot5b_instruct_cpu":
        return models.Transformers("Qwen/Qwen2.5-0.5B-Instruct")
    if selected_model_name == "transformers_qwen2dot5_0dot5b_instruct_gpu":
        return models.Transformers("Qwen/Qwen2.5-0.5B-Instruct", device_map="cuda:0")

    # QWEN3
    if selected_model_name == "llamacpp_qwen3_0dot6b_cpu":
        from huggingface_hub import hf_hub_download

        return models.LlamaCpp(
            hf_hub_download(
                repo_id="unsloth/Qwen3-0.6B-GGUF",
                filename="Qwen3-0.6B-BF16.gguf",
            ),
            n_ctx=4096,
        )

    if selected_model_name == "onnxruntime_phi4_mini_instruct":
        import json

        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        sub_dir = "gpu/gpu-int4-rtn-block-32"
        base_model_path = snapshot_download(repo_id="microsoft/Phi-4-mini-instruct-onnx", allow_patterns=f"{sub_dir}/*")

        kwargs = {}
        if torch.cuda.is_available():
            kwargs["execution_provider"] = "cuda"

        # modify context length in genai_config.json file
        config_path = os.path.join(base_model_path, sub_dir, "genai_config.json")
        config = json.load(open(config_path, "r"))
        config["model"]["context_length"] = 4096
        config["search"]["max_length"] = 4096
        json.dump(config, open(config_path, "w"))

        return models.OnnxRuntimeGenAI(model=os.path.join(base_model_path, sub_dir), **kwargs)

    raise ValueError(f"No support for selected_model_name {selected_model_name}")  # pragma: no cover


@pytest.fixture(scope="module")
def llamacpp_model(selected_model: models.Model, selected_model_name: str) -> models.LlamaCpp:
    if isinstance(selected_model, models.LlamaCpp):
        return selected_model
    pytest.skip(f"Selected model {selected_model_name} is not a LlamaCpp model, skipping llamacpp_model fixture")


@pytest.fixture(scope="module")
def onnxrt_model(selected_model: models.Model, selected_model_name: str) -> models.OnnxRuntimeGenAI:
    if isinstance(selected_model, models.OnnxRuntimeGenAI):
        return selected_model
    pytest.skip(f"Selected model {selected_model_name} is not an OnnxRuntimeGenAI model, skipping onnxrt_model fixture")


@pytest.fixture(scope="module")
def transformers_model(selected_model: models.Model, selected_model_name: str) -> models.Transformers:
    if isinstance(selected_model, models.Transformers):
        return selected_model
    pytest.skip(
        f"Selected model {selected_model_name} is not a Transformers model, skipping transformers_model fixture"
    )
