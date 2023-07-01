import guidance
import pytest

opanai_model_cache = {}

def get_llm(model_name, caching=False, **kwargs):
    """ Get an LLM by name.
    """
    if model_name.startswith("openai:"):
        return get_openai_llm(model_name[7:], caching, **kwargs)
    elif model_name.startswith("transformers:"):
        return get_transformers_llm(model_name[13:], caching, **kwargs)
    elif model_name.startswith("exllama:"):
        return get_exllama_llm(model_name.split("exllama:")[1], caching, **kwargs)

def get_openai_llm(model_name, caching=False, **kwargs):
    """ Get an OpenAI LLM with model reuse and smart test skipping.
    """

    # we cache the models so lots of tests using the same model don't have to
    # load it over and over again
    key = model_name+"_"+str(caching)
    if key not in opanai_model_cache:
        opanai_model_cache[key] = guidance.llms.OpenAI(model_name, caching=caching, **kwargs)
    llm = opanai_model_cache[key]

    if llm.api_key is None:
        pytest.skip("OpenAI token not found")

    return llm

transformers_model_cache = {}

def get_transformers_llm(model_name, caching=False):
    """ Get an OpenAI LLM with model reuse.
    """

    # we cache the models so lots of tests using the same model don't have to
    # load it over and over again
    key = model_name+"_"+str(caching)
    if key not in transformers_model_cache:
        transformers_model_cache[key] = guidance.llms.Transformers(model_name, caching=caching)

    return transformers_model_cache[key]

def get_exllama_llm(model_name, caching=False):
    key = model_name+"_"+str(caching)
    print(model_name)
    if key not in transformers_model_cache:
        import os, glob
        from exllama_lib.model import ExLlama, ExLlamaCache, ExLlamaConfig
        from exllama_lib.tokenizer import ExLlamaTokenizer
        from exllama_lib.generator import ExLlamaGenerator
        model_directory = model_name
        tokenizer_path = os.path.join(model_directory, "tokenizer.model")
        model_config_path = os.path.join(model_directory, "config.json")
        st_pattern = os.path.join(model_directory, "*.safetensors")
        model_path = glob.glob(st_pattern)[0]

        config = ExLlamaConfig(model_config_path)               # create config from config.json
        config.model_path = model_path                          # supply path to model weights file

        model = ExLlama(config)                                 # create ExLlama instance and load the weights
        tokenizer = ExLlamaTokenizer(tokenizer_path)            # create tokenizer from tokenizer model file

        cache = ExLlamaCache(model)                             # create cache for inference
        generator = ExLlamaGenerator(model, tokenizer, cache)   # create generator
        transformers_model_cache[key] = guidance.llms.ExLlama(
            model=model,
            model_config=config,
            generator=generator,
            tokenizer=tokenizer,
            caching=caching
        )

    return transformers_model_cache[key]
