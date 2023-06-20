from typing import Any, Dict
import asyncio
import re
import json
import guidance

from .caches import DiskCache

class LLMMeta(type):
    def __init__(cls, *args, **kwargs):
        cls._cache = None
    @property
    def cache(cls):
        if cls._cache is None:
            cls._cache = DiskCache(cls.llm_name)
        return cls._cache
    @cache.setter
    def cache(cls, value):
        cls._cache = value

class LLM(metaclass=LLMMeta):
    cache_version = 1
    default_system_prompt = "You are a helpful assistant."
    llm_name: str = "unknown"

    def __init__(self):
        self.chat_mode = False  # by default models are not in role-based chat mode
        self.model_name = "unknown"

        # these should all start with the @ symbol and are variables programs can use when running with this LLM
        self.tool_def = guidance("""
# Tools

{{#if len(functions) > 0~}}
## functions

namespace functions {

{{#each functions item_name="function"~}}
// {{function.description}}
type {{function.name}} = (_: {
{{~#each function.parameters.properties}}
{{#if contains(this, "description")}}// {{this.description}}
{{/if~}}
{{@key}}{{#unless contains(function.parameters.required, @key)}}?{{/unless}}: {{#if contains(this, "enum")}}{{#each this.enum}}"{{this}}"{{#unless @last}} | {{/unless}}{{/each}}{{else}}{{this.type}}{{/if}}{{#unless @last}},{{/unless}}
{{~/each}}
}) => any;

{{/each~}}
} // namespace functions
{{~/if~}}""", functions=[])
        self.function_call_stop_regex = r"\n?\n?```typescript\nfunctions.[^\(]+\(.*?\)```"

    def extract_function_call(self, text):
        m = re.match(r"\n?\n?```typescript\nfunctions.([^\(]+)\((.*?)\)```", text, re.DOTALL)

        if m:
            return CallableAnswer(m.group(1), m.group(2))

    def __call__(self, *args, asynchronous=False, **kwargs):
        """Creates a session and calls the LLM with the given arguments.

        Note that this is a convenience wrapper so you don't have to call session(),
        for higher performance across multiple calls, use a session directly.
        """
        with self.session(asynchronous=asynchronous) as s:
            out = s(*args, **kwargs)
        return out

    def __getitem__(self, key):
        """Gets an attribute from the LLM."""
        return getattr(self, key)

    def session(self, asynchronous=False):
        """Creates a session for the LLM.

        This implementation is meant to be overridden by subclasses.
        """

        if asynchronous:
            return LLMSession(self)
        else:
            return SyncSession(LLMSession(self))

    def encode(self, string, **kwargs):
        return self._tokenizer.encode(string, **kwargs)

    def decode(self, tokens, **kwargs):
        return self._tokenizer.decode(tokens, **kwargs)
    
    def id_to_token(self, id):
        return self.decode([id])

    def token_to_id(self, token):
        return self.encode(token)[0]
    
    # allow for caches to be get and set on the object as well as the class
    @property
    def cache(self):
        if self._cache is not None:
            return self._cache
        else:
            return self.__class__.cache
    @cache.setter
    def cache(self, value):
        self._cache = value


class LLMSession:
    def __init__(self, llm):
        self.llm = llm
        self._call_counts = {} # tracks the number of repeated identical calls to the LLM with non-zero temperature

    def __enter__(self):
        return self

    async def __call__(self, *args, **kwargs):
        return self.llm(*args, **kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def _gen_key(self, args_dict):
        del args_dict["self"]  # skip the "self" arg
        return "_---_".join([str(v) for v in ([args_dict[k] for k in args_dict] + [self.llm.model_name, self.llm.__class__.__name__, self.llm.cache_version])])

    def _cache_params(self, args_dict) -> Dict[str, Any]:
        """get the parameters for generating the cache key"""
        key = self._gen_key(args_dict)
        # if we have non-zero temperature we include the call count in the cache key
        if args_dict.get("temperature", 0) > 0:
            args_dict["call_count"] = self._call_counts.get(key, 0)

            # increment the call count
            self._call_counts[key] = args_dict["call_count"] + 1
        args_dict["model_name"] = self.llm.model_name
        args_dict["cache_version"] = self.llm.cache_version
        args_dict["class_name"] = self.llm.__class__.__name__

        return args_dict


class SyncSession:
    def __init__(self, session):
        self._session = session

    def __enter__(self):
        self._session.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self._session.__exit__(exc_type, exc_value, traceback)

    def __call__(self, *args, **kwargs):
        return asyncio.get_event_loop().run_until_complete(
            self._session.__call__(*args, **kwargs)
        )

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
