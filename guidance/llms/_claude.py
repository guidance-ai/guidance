import anthropic
import os
import time
import requests
import copy
import time
import asyncio
import types
import collections
import json
import re
import regex
from ._llm import LLM, LLMSession, SyncSession

# example from Claude
# https://github.com/anthropics/anthropic-sdk-python/blob/main/examples/basic_async.py
async def main(max_tokens_to_sample: int = 100):
    c = anthropic.Client(os.environ["ANTHROPIC_API_KEY"])
    resp = await c.acompletion(
        prompt=f"{anthropic.HUMAN_PROMPT} How many toes do dogs have?{anthropic.AI_PROMPT}",
        stop_sequences=[anthropic.HUMAN_PROMPT],
        model="claude-v1-100k",
        max_tokens_to_sample=max_tokens_to_sample,
    )
    print(resp)


if __name__ == "__main__":
    asyncio.run(main())


class MalformedPromptException(Exception):
    pass


def prompt_to_messages(prompt):
    messages = []

    assert prompt.endswith("<|im_start|>nAssistant\n"), ""

    pattern = r'<\|im_start\|>(\w+)(.*?)(?=<\|im_end\|>|$)'
    matches = re.findall(pattern, prompt, re.DOTALL)

    if not matches:
        return [{'role': 'user', 'content': prompt.strip()}]

    for match in matches:
        role, content = match
        content = content.strip()  # should we do this?
        messages.append({'role': role, 'content': content})

    return messages


# Should we do this?
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

# From here this is just a copy paste
class Claude(LLM):
    cache = LLM._open_cache("_claude.diskcache")  # not really sure what I am doing here

    def __init__(self, model=None, caching=True, max_retries=5, max_calls_per_min=60, token=None, endpoint=None,
                 temperature=0.0, chat_mode="auto", organization=None, allowed_special_tokens={"<|endoftext|>", "<|endofprompt|>"}):
        super().__init__()

        # fill in default model value
        if model is None:
            model = os.environ.get("CLAUDE_MODEL", None)
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
            token = os.environ.get("OPENAI_API_KEY", getattr(openai, "api_key", None))
        if token is not None and not token.startswith("sk-") and os.path.exists(os.path.expanduser(token)): # get from file
            with open(os.path.expanduser(token), 'r') as file:
                token = file.read().replace('\n', '')
        if token is None: # get from default file location
            try:
                with open(os.path.expanduser('~/.openai_api_key'), 'r') as file:
                    token = file.read().replace('\n', '')
            except:
                pass
        if organization is None:
            organization = os.environ.get("OPENAI_ORGANIZATION", None)
        # fill in default endpoint value
        if endpoint is None:
            endpoint = os.environ.get("OPENAI_ENDPOINT", None)

        import tiktoken
        self._tokenizer = tiktoken.get_encoding(tiktoken.encoding_for_model(model).name)
        self.chat_mode = chat_mode
        
        self.allowed_special_tokens = allowed_special_tokens
        self.model_name = model
        self.caching = caching
        self.max_retries = max_retries
        self.max_calls_per_min = max_calls_per_min
        if isinstance(token, str):
            token = token.replace("Bearer ", "")
        self.token = token
        self.endpoint = endpoint
        self.current_time = time.time()
        self.call_history = collections.deque()
        self.temperature = temperature
        self.organization = organization

        if self.endpoint is None:
            self.caller = self._library_call
        else:
            self.caller = self._rest_call
            self._rest_headers = {
                "Content-Type": "application/json"
            }

