import guidance
import pytest
import vertexai.language_models as palm

opanai_model_cache = {}
palm_model_cache = {}


def get_llm(model_name, caching=False, **kwargs):
    """Get an LLM by name."""
    if model_name.startswith("openai:"):
        return get_openai_llm(model_name[7:], caching, **kwargs)
    elif model_name.startswith("transformers:"):
        return get_transformers_llm(model_name[13:], caching, **kwargs)
    elif model_name.startswith("palm:"):
        return get_palm_llm(model_name[5:], caching, **kwargs)


def get_palm_llm(model_name, caching=False, **kwargs):
    """Get a Palm LLM with model reuse."""
    # we cache the models so lots of tests using the same model don't have to
    # load it over and over again
    key = model_name + "_" + str(caching)
    if key not in palm_model_cache:
        if not kwargs.get("rest_call", False):
            try:
                dummy_call = palm.ChatModel.from_pretrained("chat-bison")
            except:
                pytest.skip("Google credentials not found")
        palm_model_cache[key] = guidance.llms.PaLM(
            model_name, caching=caching, **kwargs
        )
    llm = palm_model_cache[key]
    if llm.rest_call and (llm.api_key is None or llm.project_id is None):
        pytest.skip("PaLM token/project_id not found")

    return llm


def get_openai_llm(model_name, caching=False, **kwargs):
    """Get an OpenAI LLM with model reuse and smart test skipping."""

    # we cache the models so lots of tests using the same model don't have to
    # load it over and over again
    key = model_name + "_" + str(caching)
    if key not in opanai_model_cache:
        opanai_model_cache[key] = guidance.llms.OpenAI(
            model_name, caching=caching, **kwargs
        )
    llm = opanai_model_cache[key]

    if llm.api_key is None:
        pytest.skip("OpenAI token not found")

    return llm


transformers_model_cache = {}


def get_transformers_llm(model_name, caching=False):
    """Get an OpenAI LLM with model reuse."""

    # we cache the models so lots of tests using the same model don't have to
    # load it over and over again
    key = model_name + "_" + str(caching)
    if key not in transformers_model_cache:
        transformers_model_cache[key] = guidance.llms.Transformers(
            model_name, caching=caching
        )

    return transformers_model_cache[key]
