import ast
from typing import cast

from .._guidance import guidance
from .._grammar import capture, select
from ._sequences import zero_or_more, optional
from ._subgrammar import subgrammar, lexeme


class Tool:
    def __init__(self, call_grammar=None, tool_call=None, callable=None):
        # call_grammar specifies how the tool can be called. Crucially, it has to capture the args in variable 'tool_args'
        # tool_call is a guidance function  actually calls the tool, and returns an lm object with whatever outputs it wants
        # callable: guidance function or regular callable, will be converted to grammar
        # TODO: hidden is not working yet
        first_option = (call_grammar is not None) and (tool_call is not None)
        second_option = callable is not None
        # either both are true or both false
        if first_option == second_option:
            raise Exception(
                "Must pass either (call_grammar, tool call) or callable, but not both or neither"
            )
        if second_option:
            call_grammar, tool_call = fn_to_grammar_call(callable)
            self.name = callable.__name__
        else:
            self.name = call_grammar.__name__

        self.call_grammar = call_grammar
        self.tool_call = tool_call



number = lexeme(r"\d+(\.\d+)?")
string = lexeme(r"\"[^\"]*\"")
boolean = lexeme(r"True|False")
none = lexeme(r"None")
identifier = lexeme(r"[a-zA-Z_][a-zA-Z0-9_]*")

# temp
old_capture = capture
capture = lambda x, name: x

arg = select([number, string, boolean, none])
args = capture(arg + zero_or_more("," + arg), name="tool_args")

kwarg = identifier + "=" + arg
kwargs = capture(kwarg + zero_or_more("," + kwarg), name="tool_kwargs")

capture = old_capture

argskwargs = select(
    [
        args,
        kwargs,
        args + "," + kwargs,
    ]
)

def basic_func_grammar(name):
    obj = name + "("
    obj += subgrammar(body=optional(argskwargs), skip_regex=r" *")
    obj += ")"
    return capture(obj, name="tool_call")


def fn_to_grammar_call(callable):
    # TODO later: validate the call. Here is code to get required and optional args of 'guidance_fn':
    # name = guidance_fn.__name__
    # required_args = []
    # optional_args = []
    # sig = inspect.signature(guidance_fn)
    # for i, x in enumerate(sig.parameters.values()):
    #     if i == 0:
    #         continue
    #     if x.default is x.empty:
    #         required_args.append(x.name)
    #     else:
    #         optional_args.append(x.name)
    name = callable.__name__
    call_grammar = basic_func_grammar(name)

    @guidance(dedent=False)
    def basic_tool_call(lm, call_str: str):
        call_node = cast(ast.Call, ast.parse(call_str).body[0].value) # type: ignore[attr-defined]
        args = ast.literal_eval(ast.Tuple(call_node.args))
        if len(call_node.keywords) == 0:
            kwds = {}
        else:
            kwds = ast.literal_eval(ast.Dict(*zip(*[(ast.Constant(kw.arg), kw.value) for kw in call_node.keywords])))
        lm += callable(*args, **kwds)
        return lm

    return call_grammar, basic_tool_call
