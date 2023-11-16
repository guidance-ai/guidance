import guidance
from guidance import gen, models, commit_point, Tool, select, capture, string, substring
from ..utils import get_model
import re

def test_substring_equal_unconstrained():
    llama2 = get_model("llama_cpp:")
    lm = llama2 + 'ae galera ' + gen(max_tokens=10, name='test')
    lm2 = llama2 + 'ae galera ' + substring(lm['test'])
    assert str(lm) == str(lm2)