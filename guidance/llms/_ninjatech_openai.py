import openai
import time
import requests
import time
import asyncio
import types
import collections
import re
import tiktoken
import pyparsing as pp
import json

from ._llm import LLM, LLMSession, SyncSession

import pyparsing as pp

role_start_tag = pp.Suppress(pp.Optional(pp.White()) + pp.Literal("<|im_start|>"))
role_start_name = pp.Word(pp.alphanums + "_")("role_name")
role_kwargs = pp.Suppress(pp.Optional(" ")) + pp.Dict(pp.Group(pp.Word(pp.alphanums + "_") + pp.Suppress("=") + pp.QuotedString('"')))("kwargs")
role_start = (role_start_tag + role_start_name + pp.Optional(role_kwargs) + pp.Suppress("\n")).leave_whitespace()
role_end = pp.Suppress(pp.Literal("<|im_end|>"))
role_content = pp.Combine(pp.ZeroOrMore(pp.CharsNotIn("<") | pp.Literal("<") + ~pp.FollowedBy("|im_end|>")))("role_content")
role_group = pp.Group(role_start + role_content + role_end)("role_group").leave_whitespace()
partial_role_group = pp.Group(role_start + role_content)("role_group").leave_whitespace()
roles_grammar = pp.ZeroOrMore(role_group) + pp.Optional(partial_role_group) + pp.StringEnd()


def prompt_to_messages(prompt):
    messages = []

    assert prompt.endswith("<|im_start|>assistant\n"), "When calling OpenAI chat models you must generate only directly inside the assistant role! The OpenAI API does not currently support partial assistant prompting."

    parsed_prompt = roles_grammar.parse_string(prompt)

    for role in parsed_prompt:
        if len(role["role_content"]) > 0: # only add non-empty messages (OpenAI does not support empty messages anyway)
            message = {'role': role["role_name"], 'content': role["role_content"]}
            if "kwargs" in role:
                for k, v in role["kwargs"].items():
                    message[k] = v
            messages.append(message)

    return messages

async def add_text_to_chat_mode_generator(chat_mode):
    in_function_call = False
    async for resp in chat_mode:
        if "choices" in resp:
            for c in resp['choices']:
                
                # move content from delta to text so we have a consistent interface with non-chat mode
                found_content = False
                if "content" in c['delta'] and c['delta']['content'] != "":
                    found_content = True
                    c['text'] = c['delta']['content']

                # capture function call data and convert to text again so we have a consistent interface with non-chat mode and open models
                if "function_call" in c['delta']:

                    # build the start of the function call (the follows the syntax that GPT says it wants when we ask it, and will be parsed by the @function_detector)
                    if not in_function_call:
                        start_val = "\n```typescript\nfunctions."+c['delta']['function_call']["name"]+"("
                        if not c['text']:
                            c['text'] = start_val
                        else:
                            c['text'] += start_val
                        in_function_call = True
                    
                    # extend the arguments JSON string
                    val = c['delta']['function_call']["arguments"]
                    if 'text' in c:
                        c['text'] += val
                    else:
                        c['text'] = val
                    
                if not found_content and not in_function_call:
                    break # the role markers are outside the generation in chat mode right now TODO: consider how this changes for uncontrained generation
            else:
                yield resp
        else:
            yield resp
    
    # close the function call if needed
    if in_function_call:
        yield {'choices': [{'text': ')```'}]}

def add_text_to_chat_mode(chat_mode):
    if isinstance(chat_mode, (types.AsyncGeneratorType, types.GeneratorType)):
        return add_text_to_chat_mode_generator(chat_mode)
    else:
        for c in chat_mode['choices']:
            c['text'] = c['message']['content']
        return chat_mode

class NinjaTechOpenAI(LLM):
    llm_name: str = "ninjatech_openai"

    def __init__(self, model=None, caching=True, max_retries=5, max_calls_per_min=60,
                 temperature=0.0, chat_mode="auto", encoding_name=None,
                 allowed_special_tokens={"<|endoftext|>", "<|endofprompt|>"}):
        super().__init__()

        if encoding_name is None:
            encoding_name = tiktoken.encoding_for_model(model).name

        # Auto detect chat completion mode.
        if chat_mode == "auto":
            # parse to determin if the model need to use the chat completion API
            chat_model_pattern = r'^(gpt-3\.5-turbo|gpt-4)(-\d+k)?(-\d{4})?$'
            if re.match(chat_model_pattern, model):
                chat_mode = True
            else:
                chat_mode = False

        # Set the endpoint based to the AiGateway chat completion API or the completion API.
        if chat_mode:
            # self.endpoint = "http://localhost:9080/v1/general_ai/chat_completion"
            self.endpoint = "https://ai-gateway.masoud.dev.myninja.ai/v1/general_ai/chat_completion"
        else:
            # self.endpoint = "http://localhost:9080/v1/general_ai/completion"
            self.endpoint = "https://ai-gateway.masoud.dev.myninja.ai/v1/general_ai/completion"
        
        # Set the encoding name.
        if encoding_name is None:
            encoding_name = tiktoken.encoding_for_model(model).name

        self._tokenizer = tiktoken.get_encoding(encoding_name)
        self.chat_mode = chat_mode
        self.allowed_special_tokens = allowed_special_tokens
        self.model_name = model
        self.caching = caching
        self.max_retries = max_retries
        self.max_calls_per_min = max_calls_per_min
        self.current_time = time.time()
        self.call_history = collections.deque()
        self.temperature = temperature
        self.caller = self._rest_call

    def session(self, asynchronous=False):
        if asynchronous:
            return OpenAISession(self)
        else:
            return SyncSession(OpenAISession(self))

    def role_start(self, role_name, **kwargs):
        assert self.chat_mode, "role_start() can only be used in chat mode"
        return "<|im_start|>"+role_name+"".join([f' {k}="{v}"' for k,v in kwargs.items()])+"\n"
    
    def role_end(self, role=None):
        assert self.chat_mode, "role_end() can only be used in chat mode"
        return "<|im_end|>"
    
    def end_of_text(self):
        return "<|endoftext|>"

    # Define a function to add a call to the deque
    def add_call(self):
        # Get the current timestamp in seconds
        now = time.time()
        # Append the timestamp to the right of the deque
        self.call_history.append(now)

    # Define a function to count the calls in the last 60 seconds
    def count_calls(self):
        # Get the current timestamp in seconds
        now = time.time()
        # Remove the timestamps that are older than 60 seconds from the left of the deque
        while self.call_history and self.call_history[0] < now - 60:
            self.call_history.popleft()
        # Return the length of the deque as the number of calls
        return len(self.call_history)
    
    def drop_nones(self, d):
        if type(d) == list:
            return [self.drop_nones(e) for e in d]
        elif type(d) == dict:
            for k, v in list(d.items()):
                if v is None:
                    del d[k]
                else:
                    d[k] = self.drop_nones(v)
        return d

    async def _rest_call(self, **kwargs):
        """ Call the OpenAI API using the REST API.
        """

        # Define the request headers
        headers = { "Content-Type": "application/json" }

        # Define the request data
        params = {
            "model": self.model_name,
            "max_tokens": kwargs.get("max_tokens", None),
            "temperature": kwargs.get("temperature", 0.0),
            "top_p": kwargs.get("top_p", None),
            "n": kwargs.get("n", None),
            "stop": kwargs.get("stop", None),
            "presence_penalty": kwargs.get("presence_penalty", 0),
            "frequency_penalty": kwargs.get("frequency_penalty", 0),
            "logprobs": kwargs.get("logprobs", None),
            "echo": kwargs.get("echo", False),
            "logit_bias": kwargs.get("logit_bias", None),
            "max_retries": kwargs.get("max_retries", 5),
            "ai_platform": kwargs.get("ai_platform", None),  # The default is using both OpenAI and Azure OpenAI.
        }
        data = {}
        if self.chat_mode:
            params["stream"] = False
            params["functions"] = kwargs.get("functions", None)
            params["function_call"] = kwargs.get("function_call", None)
            data["chat_completion_params"] = params
            data["messages"] = prompt_to_messages(kwargs["prompt"])
        else:
            params["best_of"] = kwargs.get("best_of", None)
            data["completion_params"] = params
            data["prompt"] = kwargs["prompt"]

        # Drop all None values from the request.
        data = self.drop_nones(data)

        # Send a POST request and get the response
        # An exception for timeout is raised if the server has not issued a response in time.
        try:
            response = requests.post(self.endpoint, headers=headers, json=data, timeout=60)
            status = response.status_code
            text = response.text
            if status != 200:
                raise Exception("Response is not 200: " + text)
            response = response.json()
        except requests.Timeout:
            raise Exception("Request timed out.")
        except requests.ConnectionError:
            raise Exception("Connection error occurred.")
        if self.chat_mode:
            response = add_text_to_chat_mode(response)
        return response
        
    async def _close_response_and_session(self, response, session):
        await response.release()
        await session.close()
    
    def encode(self, string):
        # note that is_fragment is not used used for this tokenizer
        return self._tokenizer.encode(string, allowed_special=self.allowed_special_tokens)
    
    def decode(self, tokens):
        return self._tokenizer.decode(tokens)


# define the syntax for the function definitions
start_functions = pp.Suppress(pp.Literal("## functions\n\nnamespace functions {\n\n"))
comment = pp.Combine(pp.Suppress(pp.Literal("//") + pp.Optional(" ")) + pp.restOfLine)
end_functions = pp.Suppress("} // namespace functions")
function_def_start = pp.Optional(comment)("function_description") + pp.Suppress(pp.Literal("type")) + pp.Word(pp.alphas + "_")("function_name") + pp.Suppress(pp.Literal("=") + pp.Literal("(_:") + pp.Literal("{"))
function_def_end = pp.Suppress(pp.Literal("})") + pp.Literal("=>") + pp.Literal("any;"))
parameter_type = (pp.Word(pp.alphas + "_")("simple_type") | pp.QuotedString('"')("enum_option") + pp.OneOrMore(pp.Suppress("|") + pp.QuotedString('"')("enum_option"))("enum")) + pp.Suppress(pp.Optional(","))
parameter_def = pp.Optional(comment)("parameter_description") + pp.Word(pp.alphas + "_")("parameter_name") + pp.Optional(pp.Literal("?"))("is_optional") + pp.Suppress(pp.Literal(":")) + pp.Group(parameter_type)("parameter_type")
function_def = function_def_start + pp.OneOrMore(pp.Group(parameter_def)("parameter")) + function_def_end
functions_def = start_functions + pp.OneOrMore(pp.Group(function_def)("function")) + end_functions

def get_json_from_parse(parse_out):
    functions = []
    for function in parse_out:
        function_name = function.function_name
        function_description = function.function_description
        parameters = {
            "type": "object",
            "properties": {},
            "required": []
        }
        for parameter in function:
            if isinstance(parameter, str):
                continue
            parameter_name = parameter.parameter_name
            parameter_description = parameter.parameter_description
            parameter_type = parameter.parameter_type
            is_optional = parameter.is_optional
            d = {}
            if parameter_type.simple_type:
                d["type"] = parameter_type.simple_type
            elif parameter_type.enum:
                d["type"] = "string"
                d["enum"] = [s for s in parameter_type]
            if parameter_description:
                d["description"] = parameter_description
            if not is_optional:
                parameters["required"].append(parameter_name)
            parameters["properties"][parameter_name] = d
        functions.append({
            "name": function_name,
            "description": function_description,
            "parameters": parameters
        })
    return functions

def extract_function_defs(prompt):
    """ This extracts function definitions from the prompt.
    """

    if "\n## functions\n" not in prompt:
        return None
    else:
        functions_text = prompt[prompt.index("\n## functions\n")+1:prompt.index("} // namespace functions")+24]
        parse_out = functions_def.parseString(functions_text)
        return get_json_from_parse(parse_out)


# Define a deque to store the timestamps of the calls
class OpenAISession(LLMSession):
    async def __call__(self, prompt, temperature=None, n=1, max_tokens=1000,
                       top_p=1.0, caching=None, **completion_kwargs):
        """ Generate a completion of the given prompt.
        """

        # set defaults
        if temperature is None:
            temperature = self.llm.temperature

        # get the arguments as dictionary for cache key generation
        args = locals().copy()

        # define the key for the cache
        cache_params = self._cache_params(args)
        llm_cache = self.llm.cache
        key = llm_cache.create_key(self.llm.llm_name, **cache_params)
        
        # check the cache
        # if key not in llm_cache or caching is False or (caching is not True and not self.llm.caching):
        if True:

            # ensure we don't exceed the rate limit
            while self.llm.count_calls() > self.llm.max_calls_per_min:
                await asyncio.sleep(1)

            functions = extract_function_defs(prompt)

            fail_count = 0
            while True:
                try_again = False
                try:
                    self.llm.add_call()
                    call_args = {
                        "model": self.llm.model_name,
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "n": n,
                        **completion_kwargs
                    }
                    if functions is None:
                        if "function_call" in call_args:
                            del call_args["function_call"]
                    else:
                        call_args["functions"] = functions
                    out = await self.llm.caller(**call_args)

                except (openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.APIError, openai.error.Timeout):
                    await asyncio.sleep(3)
                    try_again = True
                    fail_count += 1
                
                if not try_again:
                    break

                if fail_count > self.llm.max_retries:
                    raise Exception(f"Too many (more than {self.llm.max_retries}) OpenAI API errors in a row!")

            
            llm_cache[key] = out

        return llm_cache[key]
