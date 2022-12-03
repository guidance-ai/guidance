import openai
import pathlib
import diskcache
import os
import time

curr_dir = pathlib.Path(__file__).parent.resolve()
_file_cache = diskcache.Cache(f"{curr_dir}/../lm.diskcache")

try:
    with open(os.path.expanduser('~/.openai_api_key'), 'r') as file:
        openai.api_key = file.read().replace('\n', '')
except:
    raise Exception("Warning: No OpenAI api key found, you need to put your OpenAI key in the file ~/.openai_api_key")

try:
    with open(os.path.expanduser('~/.openai_lm_model_name'), 'r') as file:
        gpt_model = file.read().replace('\n', '')
except:
    raise Exception("Warning: No OpenAI model name found, you need to put your OpenAI model name in the file ~/.openai_lm_model_name")

class OpenAI():
    def __init__(self, model=None, caching=True):

        # default to the model specified in the environment variable
        if model is None:
            model = os.environ.get("OPENAI_MODEL", gpt_model)
        
        self.model = model
        self.caching = caching
    
    def __call__(self, prompt, stop=None, temperature=0.0, n=1, max_tokens=1000, logprobs=None):
        key = "_---_".join([str(v) for v in (self.model, prompt, stop, temperature, n, max_tokens, logprobs)])
        if key not in _file_cache or not self.caching:
            # print("CALLING LM")

            while True:
                try_again = False
                try:
                    out = openai.Completion.create(
                        model=self.model, prompt=prompt, max_tokens=max_tokens,
                        temperature=temperature, top_p=1.0, n=n, stop=stop, logprobs=logprobs#, stream=True
                    )
                except openai.error.RateLimitError:
                    time.sleep(5)
                    try_again = True
                
                if not try_again:
                    break

            _file_cache[key] = out
        return _file_cache[key]

    def tokenize(self, strings):
        out = openai.Completion.create(
            model=self.model, prompt=strings, max_tokens=1, temperature=0, logprobs=0, echo=True
        )
        return [choice["logprobs"]["tokens"][:-1] for choice in out["choices"]]

