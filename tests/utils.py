import os
from typing import Any

import pytest
from huggingface_hub import hf_hub_download

import guidance

opanai_model_cache = {}

def env_or_fail(var_name: str) -> str:
    env_value = os.getenv(var_name, None)

    assert env_value is not None, f"Env '{var_name}' not found"

    return env_value

def get_model(model_name, caching=False, **kwargs):
    """Get an LLM by name."""
    if model_name.startswith("openai:"):
        return get_openai_model(model_name[7:], caching, **kwargs)
    elif model_name.startswith("transformers:"):
        return get_transformers_model(model_name[13:], caching, **kwargs)
    elif model_name.startswith("llama_cpp:"):
        return get_llama_cpp_model(model_name[10:], caching, **kwargs)
    elif model_name.startswith("azure_guidance:"):
        return get_azure_guidance_model(model_name[15:], caching, **kwargs)
    elif model_name.startswith("huggingface_hubllama"):
        name_parts = model_name.split(":")
        return get_llama_hugging_face_model(
            repo_id=name_parts[1], filename=name_parts[2], **kwargs
        )


def get_openai_model(model_name, caching=False, **kwargs):
    """Get an OpenAI LLM with model reuse and smart test skipping."""

    # we cache the models so lots of tests using the same model don't have to
    # load it over and over again
    key = model_name + "_" + str(caching) + "_" + str(kwargs)
    if key not in opanai_model_cache:
        opanai_model_cache[key] = guidance.models.OpenAI(
            model_name, caching=caching, **kwargs
        )
    lm = opanai_model_cache[key]

    return lm


def get_llama_hugging_face_model(repo_id: str, filename: str, **kwargs):
    downloaded_file = hf_hub_download(repo_id=repo_id, filename=filename)
    lm = guidance.models.LlamaCpp(downloaded_file, **kwargs)
    return lm


transformers_model_cache = {}


def get_transformers_model(model_name, caching=False, **kwargs):
    """Get model from Hugging Face"""

    # we cache the models so lots of tests using the same model don't have to
    # load it over and over again
    key = model_name + "_" + str(caching) + "_" + str(kwargs)
    if key not in transformers_model_cache:
        transformers_model_cache[key] = guidance.models.Transformers(
            model_name, **kwargs
        )

    return transformers_model_cache[key]


llama_cpp_model_cache = {}
llama_cpp_defaults = {"n_batch": 248}


def get_llama_cpp_model(model_name, caching=False, **kwargs):
    """Get a llama.cpp LLM with model reuse."""

    if (
        model_name is None
        or isinstance(model_name, str)
        and len(model_name.strip()) == 0
    ):
        model_name = os.environ.get("LLAMA_CPP_MODEL", "")
        if len(model_name.strip()) == 0:
            pytest.skip("No llama_cpp model found.")

    kwargs = kwargs.copy()
    for key, val in llama_cpp_defaults.items():
        if key not in kwargs:
            kwargs[key] = val

    # we cache the models so lots of tests using the same model don't have to
    # load it over and over again
    key = model_name + "_" + str(caching) + "_" + str(kwargs)
    if key not in llama_cpp_model_cache:
        llama_cpp_model_cache[key] = guidance.models.LlamaCpp(
            model_name, **kwargs
        )

    return llama_cpp_model_cache[key]


azure_guidance_model_cache = {}
azure_guidance_defaults = {}


def get_azure_guidance_model(model_name, caching=False, **kwargs):
    """Get Azure Guidance LLM with model reuse."""

    if (
        model_name is None
        or isinstance(model_name, str)
        and len(model_name.strip()) == 0
    ):
        model_name = os.getenv("AZURE_GUIDANCE_URL", "")
        if len(model_name.strip()) == 0:
            pytest.skip("No Azure Guidance model found.")

    kwargs = kwargs.copy()
    for key, val in azure_guidance_defaults.items():
        if key not in kwargs:
            kwargs[key] = val

    # we cache the models so lots of tests using the same model don't have to
    # load it over and over again
    key = model_name + "_" + str(caching) + "_" + str(kwargs)
    if key not in azure_guidance_model_cache:
        azure_guidance_model_cache[key] = guidance.models.AzureGuidance(
            model_name, **kwargs
        )

    return azure_guidance_model_cache[key]
