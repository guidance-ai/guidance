import pytest
import guidance
from ..utils import get_llm

skip_tests = True
try:
    from model import ExLlama, ExLlamaCache, ExLlamaConfig
    from tokenizer import ExLlamaTokenizer
    from generator import ExLlamaGenerator
    skip_tests = False
except ImportError:
    print("Skipping exllama tests")

j
@pytest.mark.skipif(skip_tests)
@pytest.mark.parametrize("llm", ["exllama:orca_mini_7B-GPTQ"])
def test_basic(llm):
    llm = get_llm(llm)
    with llm.session() as s:
        out = s("this is a test", max_tokens=5)
        print(out)

@pytest.mark.skipif(skip_tests)
def test_basic_object_init():
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    llm = guidance.llms.Transformers(model, tokenizer)
    with llm.session() as s:
        out = s("this is a test", max_tokens=5)
        print(out)

@pytest.mark.skipif(skip_tests)
@pytest.mark.parametrize("llm", ["exllama:orca_mini_7B-GPTQ"])
def test_repeat(llm):
    llm = get_llm(llm)
    with llm.session() as s:
        out1 = s("this is a test", max_tokens=5)
        out2 = s("this is a test like another", max_tokens=5)
        print(out2)

@pytest.mark.skipif(skip_tests)
@pytest.mark.parametrize("llm", ["exllama:orca_mini_7B-GPTQ"])
def test_select(llm):
    llm = get_llm(llm)
    program = guidance('''Answer "yes" or "no": "{{#select 'answer'}}yes{{or}}no{{/select}}"''', llm=llm)
    out = program()
    assert out["answer"] in ["yes", "no"]