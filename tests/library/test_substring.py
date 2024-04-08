import guidance
from guidance import gen, models, commit_point, Tool, select, capture, string, substring
from ..utils import get_model
import re

def test_substring_equal_unconstrained(selected_model):
    target_model = selected_model
    lm = target_model + 'ae galera ' + gen(max_tokens=10, name='test')
    lm2 = target_model + 'ae galera ' + substring(lm['test'])
    assert str(lm) == str(lm2)