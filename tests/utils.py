import os
import guidance
import pytest

opanai_model_cache = {}

def get_model(model_name, caching=False, **kwargs):
    """ Get an LLM by name.
    """
    if model_name.startswith("openai:"):
        return get_openai_model(model_name[7:], caching, **kwargs)
    elif model_name.startswith("transformers:"):
        return get_transformers_model(model_name[13:], caching, **kwargs)
    elif model_name.startswith("llama_cpp:"):
        return get_llama_cpp_model(model_name[10:], caching, **kwargs)

def get_openai_model(model_name, caching=False, **kwargs):
    """ Get an OpenAI LLM with model reuse and smart test skipping.
    """

    # we cache the models so lots of tests using the same model don't have to
    # load it over and over again
    key = model_name+"_"+str(caching)+"_"+str(kwargs)
    if key not in opanai_model_cache:
        opanai_model_cache[key] = guidance.models.OpenAI(model_name, caching=caching, **kwargs)
    lm = opanai_model_cache[key]

    return lm

transformers_model_cache = {}

def get_transformers_model(model_name, caching=False, **kwargs):
    """Get model from Hugging Face
    """

    # we cache the models so lots of tests using the same model don't have to
    # load it over and over again
    key = model_name+"_"+str(caching)+"_"+str(kwargs)
    if key not in transformers_model_cache:
        transformers_model_cache[key] = guidance.models.Transformers(model_name, **kwargs)

    return transformers_model_cache[key]

llama_cpp_model_cache = {}
llama_cpp_defaults = {'n_batch': 248}

def get_llama_cpp_model(model_name, caching=False, **kwargs):
    """ Get a llama.cpp LLM with model reuse.
    """

    if model_name is None or isinstance(model_name, str) and len(model_name.strip()) == 0:
        model_name = os.environ.get("LLAMA_CPP_MODEL", "")
        if len(model_name.strip()) == 0:
            pytest.skip("No llama_cpp model found.")

    kwargs = kwargs.copy()
    for key, val in llama_cpp_defaults.items():
        if key not in kwargs:
            kwargs[key] = val

    # we cache the models so lots of tests using the same model don't have to
    # load it over and over again
    key = model_name+"_"+str(caching)+"_"+str(kwargs)
    if key not in llama_cpp_model_cache:
        llama_cpp_model_cache[key] = guidance.models.LlamaCpp(model_name, caching=caching, **kwargs)

    return llama_cpp_model_cache[key]
