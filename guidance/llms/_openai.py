import openai
import os
import time
import requests
import copy
import time
import types
import collections
import json
import re
from ._llm import LLM


class MalformedPromptException(Exception):
    pass
def prompt_to_messages(prompt):
    messages = []
    start_tags = re.findall(r'<\|im_start\|>', prompt)
    end_tags = re.findall(r'<\|im_end\|>', prompt)
    # if len(start_tags) != len(end_tags):
    #     raise MalformedPromptException("Malformed prompt: start and end tags are not properly paired")

    pattern = r'<\|im_start\|>(\w+)(.*?)(?=<\|im_end\|>)'
    matches = re.findall(pattern, prompt, re.DOTALL)

    if not matches:
        return [{'role': 'user', 'content': prompt.strip()}]

    for match in matches:
        role, content = match
        content = content.strip()
        messages.append({'role': role, 'content': content})

    return messages

def add_text_to_chat_mode_generator(chat_mode):
    for resp in chat_mode:
        if "choices" in resp:
            for c in resp['choices']:
                if "content" in c['delta']:
                    c['text'] = c['delta']['content']
                else:
                    break # the role markers are outside the generation in chat mode right now TODO: consider how this changes for uncontrained generation
            else:
                yield resp
        else:
            yield resp

def add_text_to_chat_mode(chat_mode):
    if isinstance(chat_mode, types.GeneratorType):
        return add_text_to_chat_mode_generator(chat_mode)
    else:
        for c in chat_mode['choices']:
            c['text'] = c['message']['content']
        return chat_mode
        

        # c['text'] = f'<|im_start|>{c["message"]["role"]}\n{c["message"]["content"]}<|im_end|>'

# model that need to use the chat completion API
chat_models = [
    "gpt-4",
    "gpt-3.5-turbo",
    "gpt-4-0314",
    "gpt-3.5-turbo-0301"
]

class OpenAI(LLM):
    cache = LLM._open_cache("_openai.diskcache")

    def __init__(self, model=None, caching=True, max_retries=5, max_calls_per_min=60, token=None, endpoint=None, temperature=0.0, chat_mode="auto"):

        # fill in default model value
        if model is None:
            model = os.environ.get("OPENAI_MODEL", None)
        if model is None:
            try:
                with open(os.path.expanduser('~/.openai_model'), 'r') as file:
                    model = file.read().replace('\n', '')
            except:
                pass

        # auto detect chat completion mode
        if chat_mode == "auto":
            if model in chat_models:
                chat_mode = True
            else:
                chat_mode = False
        
        # fill in default API key value
        if token is None: # get from environment variable
            token = os.environ.get("OPENAI_API_KEY", openai.api_key)
        if token is not None and not token.startswith("sk-") and os.path.exists(os.path.expanduser(token)): # get from file
            with open(os.path.expanduser(token), 'r') as file:
                token = file.read().replace('\n', '')
        if token is None: # get from default file location
            try:
                with open(os.path.expanduser('~/.openai_api_key'), 'r') as file:
                    token = file.read().replace('\n', '')
            except:
                pass
        
        # fill in default endpoint value
        if endpoint is None:
            endpoint = os.environ.get("OPENAI_ENDPOINT", None)

        import tiktoken
        self._tokenizer = tiktoken.get_encoding("cl100k_base")
        self.chat_mode = chat_mode
        
        self.model_name = model
        self.caching = caching
        self.max_retries = max_retries
        self.max_calls_per_min = max_calls_per_min
        if isinstance(token, str):
            self.token = token.replace("Bearer ", "")
        self.endpoint = endpoint
        self.current_time = time.time()
        self.call_history = collections.deque()
        self.temperature = temperature

        if self.endpoint is None:
            self.caller = self._library_call
        else:
            self.caller = self._rest_call
            self._rest_headers = {
                "Content-Type": "application/json"
            }

    def role_start(self, role):
        assert self.chat_mode, "role_start() can only be used in chat mode"
        return "<|im_start|>"+role+"\n"
    
    def role_end(self, role=None):
        assert self.chat_mode, "role_end() can only be used in chat mode"
        return "<|im_end|>"
    
    def __call__(self, prompt, stop=None, stop_regex=None, temperature=None, n=1, max_tokens=1000, logprobs=None, top_p=1.0, echo=False, logit_bias=None, pattern=None, stream=False, cache_seed=0):
        """ Generate a completion of the given prompt.
        """
        args = locals().copy()

        assert not pattern, "The OpenAI API does not support Guidance pattern controls! Please either switch to an endpoint that does, or don't use the `pattern` argument to `gen`."
        assert not stop_regex, "The OpenAI API does not support Guidance stop_regex controls! Please either switch to an endpoint that does, or don't use the `stop_regex` argument to `gen`."

        if temperature is None:
            temperature = self.temperature

        # define the key for the cache
        key = self._cache_key(args)
        
        # allow streaming to use non-streaming cache (the reverse is not true)
        if key not in self.__class__.cache and stream:
            args["stream"] = False
            key1 = self._cache_key(args)
            if key1 in self.__class__.cache:
                key = key1
        
        # check the cache
        if key not in self.__class__.cache or not self.caching:

            # ensure we don't exceed the rate limit
            if self.count_calls() > self.max_calls_per_min:
                time.sleep(1)        

            fail_count = 0
            while True:
                try_again = False
                try:
                    self.add_call()
                    call_args = {
                        "model": self.model_name,
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "n": n,
                        "stop": stop,
                        "logprobs": logprobs,
                        "echo": echo,
                        "stream": stream
                    }
                    if logit_bias is not None:
                        call_args["logit_bias"] = logit_bias
                    out = self.caller(**call_args)

                except openai.error.RateLimitError:
                    time.sleep(3)
                    try_again = True
                    fail_count += 1
                
                if not try_again:
                    break

                if fail_count > self.max_retries:
                    raise Exception(f"Too many (more than {self.max_retries}) OpenAI API RateLimitError's in a row!")

            if stream:
                return self.stream_then_save(out, key)
            else:
                self.__class__.cache[key] = out
        
        # wrap as a list if needed
        if stream:
            if isinstance(self.__class__.cache[key], list):
                return self.__class__.cache[key]
            return [self.__class__.cache[key]]
        
        return self.__class__.cache[key]
    
    def stream_then_save(self, gen, key):
        list_out = []
        for out in gen:
            list_out.append(out)
            yield out
        self.__class__.cache[key] = list_out
    
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

    def _library_call(self, **kwargs):
        """ Call the OpenAI API using the python package.

        Note that is uses the local auth token, and does not rely on the openai one.
        """
        prev_key = openai.api_key
        openai.api_key = self.token
        if self.chat_mode:
            kwargs['messages'] = prompt_to_messages(kwargs['prompt'])
            del kwargs['prompt']
            del kwargs['echo']
            del kwargs['logprobs']
            # print(kwargs)
            out = openai.ChatCompletion.create(**kwargs)
            out = add_text_to_chat_mode(out)
        else:
            out = openai.Completion.create(**kwargs)
        openai.api_key = prev_key
        return out

    def _rest_call(self, **kwargs):
        """ Call the OpenAI API using the REST API.
        """

        # Define the request headers
        headers = copy.copy(self._rest_headers)
        if self.token is not None:
            headers['Authorization'] = f"Bearer {self.token}"

        # Define the request data
        stream = kwargs.get("stream", False)
        data = {
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
            del data['stream']

        # Send a POST request and get the response
        response = requests.post(self.endpoint, headers=headers, json=data, stream=stream)
        if response.status_code != 200:
            raise Exception("Response is not 200: " + response.text)
        if stream:
            return self._rest_stream_handler(response)
        else:
            response = response.json()
        if self.chat_mode:
            response = add_text_to_chat_mode(response)
        return response
        
    def _rest_stream_handler(self, response):
        for line in response.iter_lines():
            text = line.decode('utf-8')
            if text.startswith('data: '):
                text = text[6:]
                if text == '[DONE]':
                    break
                else:
                    yield json.loads(text)
    
    def encode(self, string, is_suffix=False):
        # note that is_suffix is not used used for this tokenizer
        return self._tokenizer.encode(string)
    
    def decode(self, tokens, is_suffix=False):
        return self._tokenizer.decode(tokens)

    # def tokenize(self, strings):
    #     fail_count = 0
    #     while True:
    #         try_again = False
    #         try:
    #             out = self.caller(
    #                 model=self.model, prompt=strings, max_tokens=1, temperature=0, logprobs=0, echo=True
    #             )

    #         except openai.error.RateLimitError:
    #             time.sleep(3)
    #             try_again = True
    #             fail_count += 1
            
    #         if not try_again:
    #             break

    #         if fail_count > self.max_retries:
    #             raise Exception(f"Too many (more than {self.max_retries}) OpenAI API RateLimitError's in a row!")
        
    #     if isinstance(strings, str):
    #         return out["choices"][0]["logprobs"]["tokens"][:-1]
    #     else:
    #         return [choice["logprobs"]["tokens"][:-1] for choice in out["choices"]]


# Define a deque to store the timestamps of the calls


