import os
from pathlib import Path
import multiprocessing
from itertools import takewhile
import operator
import threading
import numpy as np
import queue
import time
import tiktoken

from ._model import Chat
from ._local import Local

try:
    # TODO: can we eliminate the torch requirement for llama.cpp by using numpy in the caller instead?
    import torch
    is_torch = True
except ImportError:
    is_torch = False

try:
    # TODO: can we eliminate the torch requirement for llama.cpp by using numpy in the caller instead?
    from vertexai.preview.language_models import TextGenerationModel
    is_vertexai = True
except ImportError:
    is_vertexai = False

class VertexAI(Local):
    def __init__(self, model, tokenizer=None, echo=True, caching=True, temperature=0.0, max_streaming_tokens=500, **kwargs):
        if not is_vertexai:
            raise Exception("Please install the vertexai package using `pip install google-cloud-aiplatform` in order to use guidance.models.VertexAI!")

        if isinstance(model, str):
            self.model_name = model
            self.model_obj = TextGenerationModel.from_pretrained(self.model_name)

        self.caching = caching
        self.temperature = temperature
        self.max_streaming_tokens = max_streaming_tokens

        # tokens = [bytes([i]) for i in range(256)] + [b'<|endoftext|>']

        # Vertex AI models don't have public tokenizations, so we simulate pretend they tokenize like gpt2...
        encoding = tiktoken.get_encoding("gpt2")
        tokens = [encoding.decode_single_token_bytes(i) for i in range(encoding.n_vocab)] + [b"<|endofprompt|>"]
        super().__init__(
            tokens,
            None, # we have no BOS token
            len(tokens)-2, # EOS token id
            echo=echo
        )

        self._shared_state = {
            "data_queue": queue.Queue(), # this is where the streaming thread puts results
            "data": b'', # these are the bytes we are ready to use in the main thread
            "not_running_stream": threading.Event(), # this is phrased negatively so we can wait for the stop event
            "last_call": 0,
            "num_calls_made": 0
        }
        self._shared_state["not_running_stream"].set()

        # self._data_queue = queue.Queue() # this is where the streaming thread put results
        # self._data = b'' # these are the bytes we are ready to use in the main thread
        # self._data_pos = 0 # how much of the data has b
        # self._running = False # is the streaming thread running or not?
        # self._generator = None
        # self._running_stream = False
        # self._stop_running_event = threading.Event()
        # self._shared_state["not_running_stream"] = threading.Event() # this is phrased negatively so we can wait for the stop event
        # self._shared_state["not_running_stream"].set()
        self.max_calls = 10
        # self._shared_state["num_calls_made"] = 0
        self.timeout = 5.1
        # self._shared_state["last_call"] = 0

    def _running_stream(self):
        return not self._shared_state["not_running_stream"].is_set() # wrap double negation (which)

    def _start_generator_stream(self, prompt):
        dqueue = self._shared_state["data_queue"]
        generator = self.model_obj.predict_streaming(prompt.decode("utf8"), max_output_tokens=self.max_streaming_tokens)
        first_iteration = True
        for chunk in generator:
            bchunk = chunk.text.encode("utf8")
            if len(bchunk) > 0:
                dqueue.put(bchunk)
            if self._shared_state["not_running_stream"].is_set() or not first_iteration and time.time() - self._shared_state["last_call"] > self.timeout:
                break
            first_iteration = False

        if self._running_stream():
            dqueue.put(b'<|endoftext|>')
        self._shared_state["not_running_stream"].set()
        dqueue.put(b'') # so we never get stuck waiting for a running stream to return something

    def get_role_start(self, name):
        return ""
    
    def get_role_end(self, name):
        if name == "instruction":
            return "<|endofprompt|>"
        else:
            raise Exception(f"The VertexAI model does not know about the {name} role type!")
        
    def _start_new_stream(self, prompt):

        if self._shared_state["num_calls_made"] > self.max_calls:
            raise Exception(f"We have exceeded the maximum number of calls! {self.max_calls}")


        # stop any running stream
        if self._running_stream():
            self._shared_state["not_running_stream"].set() # stop the generator
            self._thread.join() # wait for the thread to finish

        # clear the data queue
        while not self._shared_state["data_queue"].empty(): 
            self._shared_state["data_queue"].get()

        # start the new stream
        prompt_end = prompt.find(b'<|endofprompt|>')
        if prompt_end >= 0:
            stripped_prompt = prompt[:prompt_end]
        else:
            raise Exception("This model cannot handle prompts that don't match the instruct format!")
        self._shared_state["not_running_stream"].clear() # so we know we are running
        # self._stop_event.clear()
        self._shared_state["data"] = stripped_prompt + b'<|endofprompt|>'# we start with this data
        self._shared_state["num_calls_made"] += 1
        self._thread = threading.Thread(target=self._start_generator_stream, args=(stripped_prompt,))
        self._thread.start()
        # self._start_generator_stream(stripped_prompt)
    
    def _get_logits(self, token_ids):
        '''Computes the logits for the given token state.
        
        This overrides a method from the Local class that is used to get
        inference results from the model.
        '''

        if len(token_ids) == 0:
            raise ValueError("token_ids must contain some tokens.")
        
        # compute the prompt bytes
        prompt = b''.join([self.tokens[i] for i in token_ids])

        self._shared_state["last_call"] = time.time()

        # keep looping until we have at least one more byte past our prompt
        token_id = None
        while True:

            # try and get the next token id
            token_id = self._get_next_token(len(prompt))
            if token_id is not None:
                break
            # if self._shared_state["data"].startswith(prompt):
            #     token_id = self._get_next_token(len(prompt))#or len(self._shared_state["data"]) <= len(prompt):
            #     continue

            # we need to restart if extending our data will never lead to matching our prompt
            if len(self._shared_state["data"]) <= len(prompt) and not prompt.startswith(self._shared_state["data"]):
                self._start_new_stream(prompt)

            # extend our data with a chunk from the model stream
            if not self._shared_state["data_queue"].empty():
                new_bytes = self._shared_state["data_queue"].get_nowait()
                
                # if we are at the end of the generation then we try again allowing for early token stopping
                if len(new_bytes) == 0:
                    token_id = self._get_next_token(len(prompt), allow_early_stop=True)
                    if token_id is not None:
                        break
                self._shared_state["data"] += new_bytes
            
            # but if there is nothing and we are not running then we start a stream
            elif self._shared_state["not_running_stream"].is_set():
                self._start_new_stream(prompt)

            # we wait for the running stream to put something in the queue
            else:
                self._shared_state["data"] += self._shared_state["data_queue"].get()
        
        # # if we don't have the next byte of data yet then we wait for it (from the streaming thread)
        # if len(self._shared_state["data"]) == len(prompt):
        #     self._shared_state["data"] += self._shared_state["data"]_queue.get() 

        # token_id = self._get_next_token(len(prompt))

        # set the logits to the next byte the model picked
        logits = np.ones(len(self.tokens)) * np.nan
        logits[token_id] = 100
        
        self._shared_state["last_call"] = time.time()
        return torch.tensor(logits) # TODO: the caller need to know to NOT use the 0 entries, but to fail out if that happens
    
    def _get_next_token(self, pos, allow_early_stop=False):
        data = self._shared_state["data"]
        trie = self._token_trie
        token_id = None
        while True:
            
            # see if we have run out of data
            if pos >= len(data):
                if allow_early_stop:
                    return token_id
                else:
                    return None
            
            # try and walk down the trie
            next_byte = data[pos:pos+1]
            if next_byte in trie.children:
                trie = trie.children[next_byte]
                pos += 1
                if trie.value is not None:
                    token_id = trie.value
            else:
                return token_id # this is the longest greedy token match we can make

    # def _call_end(self):
    #     if 
    #     self._stop_event.set() # stop the generator
    #     self._generator = None
    #     self._thread.join() # stop the streaming thread
    
class VertexAIChat(VertexAI, Chat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
