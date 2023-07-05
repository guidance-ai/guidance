from typing import Any
from IPython.display import clear_output, display, HTML
import html
import re
import copy

class CallScanner:
    def __init__(self, scanner, stop=None, stop_regex=None):
        self.scanner = scanner
        self.stop = stop
        self.stop_regex = stop_regex
        assert self.stop is not None or self.stop_regex is not None, "Either stop or stop_regex must be specified."

    def __call__(self, stop_string):
        return self.scanner(stop_string)
    
class CallableAnswer:
    def __init__(self, name, args_string, function=None):
        self.__name__ = name
        self.args_string = args_string

    def __call__(self, *args, **kwargs):
        if self._function is None:
            raise NotImplementedError(f"Answer {self.__name__} has no function defined")
        return self._function(*args, **self.__kwdefaults__, **kwargs)
    
    @property
    def __kwdefaults__(self):
        """We build this lazily in case the user wants to handle validation errors themselves."""
        return json.loads(self.args_string)

    def __repr__(self):
        return f"CallableAnswer(__name__={self.__name__}, __kwdefaults__={self.__kwdefaults__})"

def _extract_function_call(self, text):
        m = re.match(r"\n?\n?```typescript\nfunctions.([^\(]+)\((.*?)\)```", text, re.DOTALL)
        if m:
            return CallableAnswer(m.group(1), m.group(2))
_default_call_scanner = CallScanner(_extract_function_call, stop_regex=r"\n?\n?```typescript\nfunctions.[^\(]+\(.*?\)```")

class LM:
    def __init__(self, model, caching=True, call_scanners=[_default_call_scanner]):
        self.model = model
        self._state = ""
        self._children = []
        self._event_queue = None
        self._event_parent = None
        self._silent = None
        self._inplace = None
        self._variables = {}
        self._caching = caching
        self._endpoint_session = None
        self.endpoint = None
        self._call_scanners = call_scanners

    def add_call_scanner(self, call_scanner):
        self._call_scanners.append(call_scanner)
        return self

    def get_endpoint_session(self):
        return self._endpoint_session_call
    
    def _endpoint_session_call(self, *args, **kwargs):
        kwargs["caching"] = self._caching
        return self._endpoint_session(*args, **kwargs)

    def _html(self):
        display_out = html.escape(self._state)
        display_out = re.sub(r"&lt;\|\|_#NODISP_\|\|&gt;.*?&lt;\|\|_/NODISP_\|\|&gt;", "", display_out, flags=re.DOTALL)
        display_out = re.sub(r"&lt;\|\|_html:(.*?)_\|\|&gt;", lambda x: html.unescape(x.group(1)), display_out, flags=re.DOTALL)
        display_out = "<pre style='margin: 0px; padding: 0px; padding-left: 8px; margin-left: -8px; border-radius: 0px; border-left: 1px solid rgba(127, 127, 127, 0.2); white-space: pre-wrap; font-family: ColfaxAI, Arial; font-size: 15px; line-height: 23px;'>"+display_out+"</pre>"
        return display_out
    
    def _send_to_event_queue(self, value):
        if self._event_queue is not None:
            self._event_queue.put(value)
        if self._event_parent is not None:
            self._event_parent._send_to_event_queue(value)

    @property
    def silent(self):
        if self._silent is not None:
            return self._silent
        return False
    
    def _clone(self):
        new_lm = copy.copy(self)
        new_lm._event_queue = None
        if self._event_queue is not None:
            new_lm._event_parent = self
        new_lm._variables = self._variables.copy()
        new_lm._children = []
        self._children.append(new_lm)
        return new_lm
    
    def _inplace_append(self, value, force_silent=False):
        """This is used just internally."""
        self._state += str(value)
        if not self.silent and not force_silent:
            clear_output(wait=True)
            display(HTML(self._html()))
        self._send_to_event_queue(self)

    def _repr_html_(self):
        clear_output(wait=True)
        return self._html()
    
    def __str__(self) -> str:
        return re.sub(r"<\|\|_.*?_\|\|>", "", self._state)
    
    def __add__(self, value):
        assert not self._inplace
        new_lm = self._clone()
        new_lm._inplace_append(value)
        return new_lm
    
    def __iadd__(self, value):
        if not self._inplace:
            new_lm = self._clone()
        else:
            new_lm = self
        new_lm._inplace_append(value)
        return new_lm
    
    def __len__(self):
        return len(str(self))
    
    def __call__(self, s):
        return self + s
    
    def __setitem__(self, key, value):
        self._variables[key] = value

    def __getitem__(self, key):
        return self._variables[key]

    def __enter__(self):
        if hasattr(self, "instance__enter__"):
            return self.instance__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        if hasattr(self, "instance__exit__"):
            return self.instance__exit__(exc_type, exc_value, traceback)

    def __call__(self, s):
        return self + s
    
    def get_encoded(self, s):
        return self.endpoint.encode(s)
    
    def get_decoded(self, s):
        return self.endpoint.decode(s)
    
    def get_id_to_token(self, id):
        return self.get_decoded([id])

    def get_token_to_id(self, token):
        return self.get_encoded(token)[0]
    
    def get_cache(self):
        return self.endpoint.cache
    
    def tool_def(self, functions):

        new_lm = self + """
# Tools

"""
        if len(functions) > 0:
            new_lm += '''## functions

namespace functions {

'''
        for function in functions:
            new_lm += f"""// {function['description']}
type {function['name']} = (_: {{"""
            for prop_name,prop_data in function["parameters"]["properties"].items():
                if "description" in prop_data:
                    new_lm += f"\n// {prop_data['description']}\n"
                new_lm += prop_name
                if prop_name not in function["parameters"]["required"]:
                    new_lm += "?"
                new_lm += ": "
                if "enum" in prop_data:
                    for enum in prop_data["enum"]:
                        new_lm += f'"{enum}"'
                        if enum != prop_data["enum"][-1]:
                            new_lm += " | "
                else:
                    new_lm += prop_data["type"]
                
                if prop_name != list(function["parameters"]["properties"].keys())[-1]:
                    new_lm += ",\n"
            new_lm += """
}) => any;

"""
        new_lm += "} // namespace functions\n"
        return new_lm


class ChatLM(LM):

    def get_role_start(self, role_name):
        return f"<|im_start|>{role_name}\n"
    
    def get_role_end(self, role_name):
        return "<|im_end|>"