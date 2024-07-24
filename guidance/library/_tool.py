from .._guidance import guidance
from ._any_char import any_char
from .._grammar import select, capture, string
from ._subgrammar import subgrammar, lexeme
from ._sequences import zero_or_more, one_or_more
from ._any_char_but import any_char_but
from ._any_char import any_char


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


def positional_arg():
    return lexeme(r"[^=)]+")


def kwarg():
    return positional_arg() + "=" + positional_arg()


def basic_func_grammar(name):
    obj = string(name + "(")
    obj += capture(
        positional_arg(),
        name="tool_args",
    )
    obj += string(")")
    return obj


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
    def basic_tool_call(lm, args: str):
        args = args.split(",")
        positional = [x.strip() for x in args if "=" not in x]
        kwargs = dict([tuple(x.strip().split("=")) for x in args if "=" in x])
        lm += callable(*positional, **kwargs)
        return lm

    return call_grammar, basic_tool_call
