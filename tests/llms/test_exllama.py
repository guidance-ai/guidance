import pytest
import guidance
from ..utils import get_llm

skip_tests = True
try:
    from exllama_lib.model import ExLlama, ExLlamaCache, ExLlamaConfig
    from exllama_lib.tokenizer import ExLlamaTokenizer
    from exllama_lib.generator import ExLlamaGenerator
    skip_tests = False
except ImportError:
    print("Skipping exllama tests")

skip_reason="ExLlama not installed"

@pytest.mark.skipif(skip_tests, reason=skip_reason)
@pytest.mark.parametrize("llm", ["exllama:orca_mini_7B-GPTQ"])
def test_basic(llm):
    llm = get_llm(llm)
    with llm.session() as s:
        out = s("this is a test", max_tokens=5)
        print(out)

@pytest.mark.skipif(skip_tests, reason=skip_reason)
def test_basic_object_init():
    import os
    import os, glob

    model_directory =  "./orca_mini_7B-GPTQ/"
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

    llm = guidance.llms.ExLlama(model=generator, tokenizer=tokenizer)
    with llm.session() as s:
        out = s("this is a test", max_tokens=5)
        print(out)

@pytest.mark.skipif(skip_tests, reason=skip_reason)
@pytest.mark.parametrize("llm", ["exllama:orca_mini_7B-GPTQ"])
def test_repeat(llm):
    llm = get_llm(llm)
    with llm.session() as s:
        out1 = s("this is a test", max_tokens=5)
        out2 = s("this is a test like another", max_tokens=5)
        print(out2)

@pytest.mark.skipif(skip_tests, reason=skip_reason)
@pytest.mark.parametrize("llm", ["exllama:orca_mini_7B-GPTQ"])
def test_select(llm):
    llm = get_llm(llm)
    program = guidance('''Answer "yes" or "no": "{{#select 'answer'}}yes{{or}}no{{/select}}"''', llm=llm)
    out = program()
    assert out["answer"] in ["yes", "no"]