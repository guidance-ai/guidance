import guidance
from guidance import gen
from ..utils import get_model

def test_gen():
    lm = get_model("transformers:gpt2")
    lm += "Write a number: " + gen('text', max_tokens=3)
    assert len(lm["text"]) > 0

# def test_add_multi():
#     """ Test more than 2 arguments for `add`.
#     """

#     program = guidance("""Write a number: {{set 'user_response' (add 20 5 variable)}}""")
#     assert program(variable=10)["user_response"] == 35
#     assert program(variable=20.1)["user_response"] == 45.1

# def test_add_infix():
#     """ Basic infix test of `add`.
#     """

#     program = guidance("""Write a number: {{set 'user_response' 20 + variable}}""")
#     assert program(variable=10)["user_response"] == 30
#     assert program(variable=20.1)["user_response"] == 40.1

# if __name__ == "__main__":
#     # find all the test functions in this file
#     import sys, inspect
#     test_functions = [obj for name, obj in inspect.getmembers(sys.modules[__name__]) if (inspect.isfunction(obj) and name.startswith("test_"))]
#     # run each test function
#     for test_function in test_functions:
#         test_function()