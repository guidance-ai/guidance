import openai
import os
import time
import requests
import aiohttp
import copy
import time
import asyncio
import types
import collections
import json
import re
import regex

from ._llm import LLM, LLMSession, SyncSession


class MalformedPromptException(Exception):
    pass

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

# import pyparsing as pp

# role_start_tag = pp.Literal("<|im_start|>")
# role_start_name = pp.Word(pp.alphanums + "_")
# role_kwargs = pp.Dict(pp.Group(pp.Word(pp.alphanums + "_") + pp.Suppress("=") + pp.QuotedString('"')))
# role_start = role_start_tag + role_start_name + pp.Optional(role_kwargs) + pp.Suppress("\n")
# role_end = pp.Literal("<|im_end|>")
# role_content = pp.CharsNotIn("<|im_start|><|im_end|>")

# r'<\|im_start\|>([^\n]+)\n(.*?)(?=<\|im_end\|>|$)'

def prompt_to_messages(prompt):
    messages = []

    assert prompt.endswith("<|im_start|>assistant\n"), "When calling OpenAI chat models you must generate only directly inside the assistant role! The OpenAI API does not currently support partial assistant prompting."

    parsed_prompt = roles_grammar.parse_string(prompt)

    # pattern = r'<\|im_start\|>([^\n]+)\n(.*?)(?=<\|im_end\|>|$)'
    # matches = re.findall(pattern, prompt, re.DOTALL)

    # if not matches:
    #     return [{'role': 'user', 'content': prompt}]

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

class OpenAI(LLM):
    llm_name: str = "openai"

    def __init__(self, model=None, caching=True, max_retries=5, max_calls_per_min=60,
                 api_key=None, api_type="open_ai", api_base=None, api_version=None, deployment_id=None,
                 temperature=0.0, chat_mode="auto", organization=None, rest_call=False,
                 allowed_special_tokens={"<|endoftext|>", "<|endofprompt|>"},
                 token=None, endpoint=None, encoding_name=None):
        super().__init__()

        # map old param values
        # TODO: add deprecated warnings after some time
        if token is not None:    
            if api_key is None:
                api_key = token
        if endpoint is not None:
            if api_base is None:
                api_base = endpoint

        # fill in default model value
        if model is None:
            model = os.environ.get("OPENAI_MODEL", None)
        if model is None:
            try:
                with open(os.path.expanduser('~/.openai_model'), 'r') as file:
                    model = file.read().replace('\n', '')
            except:
                pass

        # fill in default deployment_id value
        if deployment_id is None:
            deployment_id = os.environ.get("OPENAI_DEPLOYMENT_ID", None)

        # auto detect chat completion mode
        if chat_mode == "auto":
            # Determine if the model needs to use the chat completion API
            # Retrieve the list of models using the OpenAI library
            response = openai.Model.list()

            # Extract model names from the response
            models = response['data']

            # Filter out model names that start with 'gpt'
            gpt_models = [model['id'] for model in models if model['id'].startswith('gpt')]

            if model in gpt_models:
                chat_mode = True
            else:
                chat_mode = False
        
        # fill in default API key value
        if api_key is None: # get from environment variable
            api_key = os.environ.get("OPENAI_API_KEY", getattr(openai, "api_key", None))
        if api_key is not None and not api_key.startswith("sk-") and os.path.exists(os.path.expanduser(api_key)): # get from file
            with open(os.path.expanduser(api_key), 'r') as file:
                api_key = file.read().replace('\n', '')
        if api_key is None: # get from default file location
            try:
                with open(os.path.expanduser('~/.openai_api_key'), 'r') as file:
                    api_key = file.read().replace('\n', '')
            except:
                pass
        if organization is None:
            organization = os.environ.get("OPENAI_ORGANIZATION", None)
        # fill in default endpoint value
        if api_base is None:
            api_base = os.environ.get("OPENAI_API_BASE", None) or os.environ.get("OPENAI_ENDPOINT", None) # ENDPOINT is deprecated

        import tiktoken
        if encoding_name is None:
            encoding_name = tiktoken.encoding_for_model(model).name
        self._tokenizer = tiktoken.get_encoding(encoding_name)
        self.chat_mode = chat_mode
        
        self.allowed_special_tokens = allowed_special_tokens
        self.model_name = model
        self.deployment_id = deployment_id
        self.caching = caching
        self.max_retries = max_retries
        self.max_calls_per_min = max_calls_per_min
        if isinstance(api_key, str):
            api_key = api_key.replace("Bearer ", "")
        self.api_key = api_key
        self.api_type = api_type
        self.api_base = api_base
        self.api_version = api_version
        self.current_time = time.time()
        self.call_history = collections.deque()
        self.temperature = temperature
        self.organization = organization
        self.rest_call = rest_call
        self.endpoint = endpoint

        if not self.rest_call:
            self.caller = self._library_call
        else:
            self.caller = self._rest_call
            self._rest_headers = {
                "Content-Type": "application/json"
            }

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
    
    @classmethod
    async def stream_then_save(cls, gen, key, stop_regex, n):
        list_out = []
        cached_out = None

        # init stop_regex variables
        if stop_regex is not None:
            if isinstance(stop_regex, str):
                stop_patterns = [regex.compile(stop_regex)]
            else:
                stop_patterns = [regex.compile(pattern) for pattern in stop_regex]

            current_strings = ["" for _ in range(n)]
            # last_out_pos = ["" for _ in range(n)]
        
        # iterate through the stream
        all_done = False
        async for curr_out in gen:

            # if we have a cached output, extend it with the current output
            if cached_out is not None:
                out = merge_stream_chunks(cached_out, curr_out)
            else:
                out = curr_out
            
            # check if we have stop_regex matches
            found_partial = False
            if stop_regex is not None:

                # keep track of the generated text so far
                for i,choice in enumerate(curr_out['choices']):
                    current_strings[i] += choice['text']

                # check if all of the strings match a stop string (and hence we can stop the batch inference)
                all_done = True
                for i in range(len(current_strings)):
                    found = False
                    for s in stop_patterns:
                        if s.search(current_strings[i]):
                            found = True
                    if not found:
                        all_done = False
                        break

                # find where trim off the stop regex matches if needed (and look for partial matches)
                stop_pos = [1e10 for _ in range(n)]
                stop_text = [None for _ in range(n)]
                for i in range(len(current_strings)):
                    for s in stop_patterns:
                        m = s.search(current_strings[i], partial=True)
                        if m:
                            span = m.span()
                            if span[1] > span[0]:
                                if m.partial: # we might be starting a stop sequence, so we can't emit anything yet
                                    found_partial = True
                                    break
                                else:
                                    stop_text[i] = current_strings[i][span[0]:span[1]]
                                    stop_pos[i] = min(span[0], stop_pos[i])
                    if stop_pos != 1e10:
                        stop_pos[i] = stop_pos[i] - len(current_strings[i]) # convert to relative position from the end
            
            # if we might be starting a stop sequence, we need to cache the output and continue to wait and see
            if found_partial:
                cached_out = out
                continue
            
            # if we get here, we are not starting a stop sequence, so we can emit the output
            else:
                cached_out = None

                if stop_regex is not None:
                    for i in range(len(out['choices'])):
                        if stop_pos[i] < len(out['choices'][i]['text']):
                            out['choices'][i] = out['choices'][i].to_dict() # because sometimes we might need to set the text to the empty string (and OpenAI's object does not like that)
                            out['choices'][i]['text'] = out['choices'][i]['text'][:stop_pos[i]]
                            out['choices'][i]['stop_text'] = stop_text[i]
                            out['choices'][i]['finish_reason'] = "stop"
            
                list_out.append(out)
                yield out
                if all_done:
                    gen.aclose()
                    break
        
        # if we have a cached output, emit it
        if cached_out is not None:
            list_out.append(cached_out)
            yield out

        cls.cache[key] = list_out
    
    def _stream_completion(self):
        pass

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

    async def _library_call(self, **kwargs):
        """ Call the OpenAI API using the python package.

        Note that is uses the local auth token, and does not rely on the openai one.
        """

        # save the params of the openai library
        prev_key = openai.api_key
        prev_org = openai.organization
        prev_type = openai.api_type
        prev_version = openai.api_version
        prev_base = openai.api_base
        
        # set the params of the openai library if we have them
        if self.api_key is not None:
            openai.api_key = self.api_key
        if self.organization is not None:
            openai.organization = self.organization
        if self.api_type is not None:
            openai.api_type = self.api_type
        if self.api_version is not None:
            openai.api_version = self.api_version
        if self.api_base is not None:
            openai.api_base = self.api_base

        assert openai.api_key is not None, "You must provide an OpenAI API key to use the OpenAI LLM. Either pass it in the constructor, set the OPENAI_API_KEY environment variable, or create the file ~/.openai_api_key with your key in it."
        
        if self.chat_mode:
            kwargs['messages'] = prompt_to_messages(kwargs['prompt'])
            del kwargs['prompt']
            del kwargs['echo']
            del kwargs['logprobs']
            # print(kwargs)
            out = await openai.ChatCompletion.acreate(**kwargs)
            out = add_text_to_chat_mode(out)
        else:
            out = await openai.Completion.acreate(**kwargs)
        
        # restore the params of the openai library
        openai.api_key = prev_key
        openai.organization = prev_org
        openai.api_type = prev_type
        openai.api_version = prev_version
        openai.api_base = prev_base
        
        return out

    async def _rest_call(self, **kwargs):
        """ Call the OpenAI API using the REST API.
        """

        # Define the request headers
        headers = copy.copy(self._rest_headers)
        if self.api_key is not None:
            headers['Authorization'] = f"Bearer {self.api_key}"

        # Define the request data
        stream = kwargs.get("stream", False)
        data = {
            "model": self.model_name,
            "prompt": kwargs["prompt"],
            "max_tokens": kwargs.get("max_tokens", None),
            "temperature": kwargs.get("temperature", 0.0),
            "top_p": kwargs.get("top_p", 1.0),
            "n": kwargs.get("n", 1),
            "stream": stream,
            "logprobs": kwargs.get("logprobs", None),
            'stop': kwargs.get("stop", None),
            "echo": kwargs.get("echo", False)
        }
        if self.chat_mode:
            data['messages'] = prompt_to_messages(data['prompt'])
            del data['prompt']
            del data['echo']
            del data['logprobs']

        # Send a POST request and get the response
        # An exception for timeout is raised if the server has not issued a response for 10 seconds
        try:
            if stream:
                session = aiohttp.ClientSession()
                response = await session.post(self.endpoint, json=data, headers=headers, timeout=60)
                status = response.status
            else:
                response = requests.post(self.endpoint, headers=headers, json=data, timeout=60)
                status = response.status_code
                text = response.text
            if status != 200:
                if stream:
                    text = await response.text()
                raise Exception("Response is not 200: " + text)
            if stream:
                response = self._rest_stream_handler(response, session)
            else:
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

    async def _rest_stream_handler(self, response, session):
        # async for line in response.iter_lines():
        async for line in response.content:
            text = line.decode('utf-8')
            if text.startswith('data: '):
                text = text[6:]
                if text.strip() == '[DONE]':
                    await self._close_response_and_session(response, session)
                    break
                else:
                    yield json.loads(text)
    
    def encode(self, string):
        # note that is_fragment is not used used for this tokenizer
        return self._tokenizer.encode(string, allowed_special=self.allowed_special_tokens)
    
    def decode(self, tokens):
        return self._tokenizer.decode(tokens)


def merge_stream_chunks(first_chunk, second_chunk):
    """ This merges two stream responses together.
    """

    out = copy.deepcopy(first_chunk)

    # merge the choices
    for i in range(len(out['choices'])):
        out_choice = out['choices'][i]
        second_choice = second_chunk['choices'][i]
        out_choice['text'] += second_choice['text']
        if 'index' in second_choice:
            out_choice['index'] = second_choice['index']
        if 'finish_reason' in second_choice:
            out_choice['finish_reason'] = second_choice['finish_reason']
        if out_choice.get('logprobs', None) is not None:
            out_choice['logprobs']['token_logprobs'] += second_choice['logprobs']['token_logprobs']
            out_choice['logprobs']['top_logprobs'] += second_choice['logprobs']['top_logprobs']
            out_choice['logprobs']['text_offset'] = second_choice['logprobs']['text_offset']
    
    return out


class OpenAIStreamer():
    def __init__(self, stop_regex, n):
        self.stop_regex = stop_regex
        self.n = n
        self.current_strings = ["" for _ in range(n)]
        self.current_length = 0

class RegexStopChecker():
    def __init__(self, stop_pattern, decode, prefix_length):
        if isinstance(stop_pattern, str):
            self.stop_patterns = [regex.compile(stop_pattern)]
        else:
            self.stop_patterns = [regex.compile(pattern) for pattern in stop_pattern]
        self.prefix_length = prefix_length
        self.decode = decode
        self.current_strings = None
        self.current_length = 0

    def __call__(self, input_ids, scores, **kwargs):

        # extend our current strings
        if self.current_strings is None:
            self.current_strings = ["" for _ in range(len(input_ids))]
        for i in range(len(self.current_strings)):
            self.current_strings[i] += self.decode(input_ids[i][self.current_length:])
        
        # trim off the prefix string so we don't look for stop matches in the prompt
        if self.current_length == 0:
            for i in range(len(self.current_strings)):
                self.current_strings[i] = self.current_strings[i][self.prefix_length:]
        
        self.current_length = len(input_ids[0])
        
        # check if all of the strings match a stop string (and hence we can stop the batch inference)
        all_done = True
        for i in range(len(self.current_strings)):
            found = False
            for s in self.stop_patterns:
                if s.search(self.current_strings[i]):
                    found = True
            if not found:
                all_done = False
                break
        
        return all_done

# define the syntax for the function definitions
import pyparsing as pp
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
    async def __call__(self, prompt, stop=None, stop_regex=None, temperature=None, n=1, max_tokens=1000, logprobs=None,
                       top_p=1.0, echo=False, logit_bias=None, token_healing=None, pattern=None, stream=None,
                       cache_seed=0, caching=None, **completion_kwargs):
        """ Generate a completion of the given prompt.
        """

        # we need to stream in order to support stop_regex
        if stream is None:
            stream = stop_regex is not None
        assert stop_regex is None or stream, "We can only support stop_regex for the OpenAI API when stream=True!"
        assert stop_regex is None or n == 1, "We don't yet support stop_regex combined with n > 1 with the OpenAI API!"

        assert token_healing is None or token_healing is False, "The OpenAI API does not yet support token healing! Please either switch to an endpoint that does, or don't use the `token_healing` argument to `gen`."

        # set defaults
        if temperature is None:
            temperature = self.llm.temperature

        # get the arguments as dictionary for cache key generation
        args = locals().copy()

        assert not pattern, "The OpenAI API does not support Guidance pattern controls! Please either switch to an endpoint that does, or don't use the `pattern` argument to `gen`."
        # assert not stop_regex, "The OpenAI API does not support Guidance stop_regex controls! Please either switch to an endpoint that does, or don't use the `stop_regex` argument to `gen`."

        # define the key for the cache
        cache_params = self._cache_params(args)
        llm_cache = self.llm.cache
        key = llm_cache.create_key(self.llm.llm_name, **cache_params)
        
        # allow streaming to use non-streaming cache (the reverse is not true)
        if key not in llm_cache and stream:
            cache_params["stream"] = False
            key1 = llm_cache.create_key(self.llm.llm_name, **cache_params)
            if key1 in llm_cache:
                key = key1
        
        # check the cache
        if key not in llm_cache or caching is False or (caching is not True and not self.llm.caching):

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
                        "deployment_id": self.llm.deployment_id,
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "n": n,
                        "stop": stop,
                        "logprobs": logprobs,
                        "echo": echo,
                        "stream": stream,
                        **completion_kwargs
                    }
                    if functions is None:
                        if "function_call" in call_args:
                            del call_args["function_call"]
                    else:
                        call_args["functions"] = functions
                    if logit_bias is not None:
                        call_args["logit_bias"] = {str(k): v for k,v in logit_bias.items()} # convert keys to strings since that's the open ai api's format
                    out = await self.llm.caller(**call_args)

                except (openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.APIError, openai.error.Timeout):
                    await asyncio.sleep(3)
                    try_again = True
                    fail_count += 1
                
                if not try_again:
                    break

                if fail_count > self.llm.max_retries:
                    raise Exception(f"Too many (more than {self.llm.max_retries}) OpenAI API errors in a row!")

            if stream:
                return self.llm.stream_then_save(out, key, stop_regex, n)
            else:
                llm_cache[key] = out
        
        # wrap as a list if needed
        if stream:
            if isinstance(llm_cache[key], list):
                return llm_cache[key]
            return [llm_cache[key]]
        
        return llm_cache[key]


import os
import json
import platformdirs
from ._openai import OpenAI

class AzureOpenAI(OpenAI):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("The AzureOpenAI class has been merged with the OpenAI class for Azure usage. Please use the OpenAI class instead: https://guidance.readthedocs.io/en/latest/example_notebooks/api_examples/llms/OpenAI.html")

class MSALOpenAI(OpenAI):
    """ Microsoft Authentication Library (MSAL) OpenAI style integration.

    Warning: This class is not finalized and may change in the future.
    """

    llm_name: str = "azure_openai"

    def __init__(self, model=None, client_id=None, authority=None, caching=True, max_retries=5, max_calls_per_min=60, token=None,
                 endpoint=None, scopes=None, temperature=0.0, chat_mode="auto", rest_call=False):
        

        assert endpoint is not None, "An endpoint must be specified!"
        
        # build a standard OpenAI LLM object
        super().__init__(
            model=model, caching=caching, max_retries=max_retries, max_calls_per_min=max_calls_per_min,
            token=token, endpoint=endpoint, temperature=temperature, chat_mode=chat_mode, rest_call=rest_call
        )

        self.client_id = client_id
        self.authority = authority
        self.scopes = scopes

        from msal import PublicClientApplication, SerializableTokenCache
        self._token_cache = SerializableTokenCache()
        self._token_cache_path = os.path.join(platformdirs.user_cache_dir("guidance"), "_azure_openai.token")
        self._app = PublicClientApplication(client_id=self.client_id, authority=self.authority, token_cache=self._token_cache)
        if os.path.exists(self._token_cache_path):
            self._token_cache.deserialize(open(self._token_cache_path, 'r').read())

        if( rest_call ):
            self._rest_headers["X-ModelType"] = self.model_name

    @property
    def api_key(self):
        return self._get_token()
    @api_key.setter
    def api_key(self, value):
        pass # ignored for now

    def _get_token(self):
        accounts = self._app.get_accounts()
        result = None

        if accounts:
            # Assuming the end user chose this one
            chosen = accounts[0]

            # Now let's try to find a token in cache for this account
            result = self._app.acquire_token_silent(self.scopes, account=chosen)
    
        if not result:
            # So no suitable token exists in cache. Let's get a new one from AAD.
            flow = self._app.initiate_device_flow(scopes=self.scopes)

            if "user_code" not in flow:
                raise ValueError(
                    "Fail to create device flow. Err: %s" % json.dumps(flow, indent=4))

            print(flow["message"])

            result = self._app.acquire_token_by_device_flow(flow)

            # save the aquired token
            with open(self._token_cache_path, "w") as f:
                f.write(self._token_cache.serialize())

        return result["access_token"]
