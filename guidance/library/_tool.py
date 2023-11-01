class Tool:
    def __init__(self, call_grammar, tool_call):
        # call_grammar specifies how the tool can be called. Crucially, it has to capture the args in variable 'tool_args'
        # tool_call is a guidance function  actually calls the tool, and returns an lm object with whatever outputs it wants
        # TODO: hidden is not working yet
        self.call_grammar = call_grammar
        self.tool_call = tool_call

def valid_chars():
    return any_char_but(['=', ')'])
def positional_arg():
    return one_or_more(valid_chars())
def kwarg():
    return one_or_more(valid_chars()) + '=' + one_or_more(valid_chars())

def basic_func_grammar(name):
    obj = string(name + '(')
    obj += capture(select([zero_or_more(positional_arg()), ''])
        + select([zero_or_more(kwarg()), '']), name='tool_args')
    obj += string(')')
    return obj

def fn_to_tool(self, callable):
    name = callable.__name__
    call_grammar = basic_func_grammar(name)
    @guidance
    def basic_tool_call(lm):
        args = lm['tool_args']
        args = args.split(',')
        positional = [x.strip() for x in args if '=' not in x]
        kwargs = dict([tuple(x.strip().split('=')) for x in args if '=' in x])
        lm += callable(*positional, **kwargs)
        return lm
    return Tool(call_grammar, basic_tool_call)

# @guidance
# def default_text_to_callable(lm, generated_text):

# class Tool:
#     def __init__(self, callable, scan_pattern=None, text_to_callable=None):
#         # callable is a guidance function
#         # Scan pattern is a pattern used to stop generation
#         # text_to_callable returns either None if the tool does not apply to the generated text, or calls 'callable'
#         # and executes the tool (whatever the tool does)
#         self.scan_pattern = scan_patter
#         self.text_to_callable = text_to_callable
#     def __call__(self, lm, text):




# Leaving this here for now in case we want to return to this pattern, but implemented a different version above for prototyping
class CallScanner:
    def __init__(self, scanner, stop=None, stop_regex=None):
        self.scanner = scanner
        self.stop = stop
        self.stop_regex = stop_regex
        assert self.stop is not None or self.stop_regex is not None, "Either stop or stop_regex must be specified."

    def __call__(self, lm, text):
        out = self.scanner(text)
        if isinstance(out, CallableAnswer) and out.callable is None:
            out.callable = lm.get(out.__name__, {"callable": None}).get("callable", None)
        return out

    
class CallableAnswer:
    def __init__(self, text, name, args_string, callable=None):
        self._text = text
        self.__name__ = name
        self.args_string = args_string
        self.callable = callable

    def __str__(self):
        return self._text

    def __call__(self, *args, **kwargs):
        if self.callable is None:
            raise NotImplementedError(f"Answer {self.__name__} has no function defined")
        return self.callable(*args, **self.__kwdefaults__, **kwargs)
    
    @property
    def __kwdefaults__(self):
        """We build this lazily in case the user wants to handle validation errors themselves."""
        return json.loads(self.args_string)

    def __repr__(self):
        return self._text + f"\nCallableAnswer(__name__={self.__name__}, __kwdefaults__={self.__kwdefaults__})"

def _default_extract_function_call(text):
    m = re.match(r"(.*?)\n?\n?```typescript\nfunctions.([^\(]+)\((.*?)\)```", text, re.DOTALL)
    if m:
        return CallableAnswer(text=m.group(1), name=m.group(2), args_string=m.group(3))
# _default_call_scanner = CallScanner(_extract_function_call, stop_regex=r"\n?\n?```typescript\nfunctions.[^\(]+\(.*?\)```")

# Old functions, to clean up
    # def add_call_scanner(self, scanner, stop=None, stop_regex=None):
    #     self._call_scanners.append(CallScanner(scanner, stop=stop, stop_regex=stop_regex))
    #     return self
    
    # def get_call_scanners(self):
    #     return self._call_scanners
