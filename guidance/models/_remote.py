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
import re

from ._model import Chat, Instruct
from ._local import Local


# try:
#     # TODO: can we eliminate the torch requirement for llama.cpp by using numpy in the caller instead?
#     import vertexai
#     is_vertexai = True
# except ImportError:
#     is_vertexai = False

class Remote(Local):
    def __init__(self, model, tokenizer=None, echo=True, caching=True, temperature=0.0, max_streaming_tokens=500, **kwargs):
        self.caching = caching
        self.temperature = temperature
        self.max_streaming_tokens = max_streaming_tokens

        # Remote models don't always have public tokenizations, so when not provided we pretend they tokenize like gpt2...
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("gpt2")

        # tiktoken tokenizer was given
        if hasattr(tokenizer, "decode_single_token_bytes"):
            special_map = {v: k for k,v in tokenizer._special_tokens.items()}
            byte_tokens = [b'<|invalid_special_token|>']
            for i in range(tokenizer.n_vocab):
                try:
                    bval = tokenizer.decode_single_token_bytes(i)
                except KeyError:
                    bval = special_map.get(i, b'<|invalid_special_token|>')
                byte_tokens.append(bval)

            bos_token_id = None
            eos_token_id = tokenizer._special_tokens["<|endoftext|>"]
        
        # a transformer tokenizer was given that has a byte_decoder
        elif hasattr(tokenizer, "byte_decoder"):
            byte_tokens = []
            for i in range(tokenizer.vocab_size):
                byte_coded = bytes([tokenizer.byte_decoder[c] for c in tokenizer.convert_ids_to_tokens(i)])
                byte_tokens.append(byte_coded)
            bos_token_id = tokenizer.bos_token_id
            eos_token_id = tokenizer.eos_token_id
        
        # a transformer tokenizer was given with byte_decoder
        elif hasattr(tokenizer, "convert_ids_to_tokens"):
            byte_tokens = [bytes(tokenizer.convert_tokens_to_string(['a', tokenizer.convert_ids_to_tokens(i)])[1:], encoding="utf8") for i in range(tokenizer.vocab_size)]
            bos_token_id = tokenizer.bos_token_id
            eos_token_id = tokenizer.eos_token_id

        else:
            raise Exception("The tokenizer given was not of a recognized type!")

        # build the 
        super().__init__(
            byte_tokens,
            bos_token_id,
            eos_token_id,
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

        self.max_calls = 10
        self.timeout = 50.1

    def _running_stream(self):
        return not self._shared_state["not_running_stream"].is_set() # wrap double negation (which)

    def _start_generator_stream(self, generator):
        dqueue = self._shared_state["data_queue"]
        first_iteration = True
        for chunk in generator:
            if len(chunk) > 0:
                dqueue.put(chunk)
            if self._shared_state["not_running_stream"].is_set() or not first_iteration and time.time() - self._shared_state["last_call"] > self.timeout:
                break
            first_iteration = False

        if self._running_stream():
            dqueue.put(self.tokens[self.eos_token_id])
        self._shared_state["not_running_stream"].set()
        dqueue.put(b'') # so we never get stuck waiting for a running stream to return something

    def _start_new_stream(self, prompt):

        if self._shared_state["num_calls_made"] > self.max_calls:
            raise Exception(f"We have exceeded the maximum number of calls! {self.max_calls}")

        # stop any running stream
        if self._running_stream():
            self._shared_state["not_running_stream"].set() # stop the generator
            self._shared_state["remote_thread"].join() # wait for the thread to finish

        # clear the data queue
        while not self._shared_state["data_queue"].empty(): 
            self._shared_state["data_queue"].get()

        # start the new stream
        self._shared_state["last_call"] = time.time()
        generator = self._generator(prompt)
        self._shared_state["not_running_stream"].clear() # so we know we are running
        self._shared_state["num_calls_made"] += 1

        self._shared_state["remote_thread"] = threading.Thread(target=self._start_generator_stream, args=(generator,))
        self._shared_state["remote_thread"].start()
        # self._start_generator_stream(stripped_prompt)
    
    def _get_logits(self, token_ids, forced_bytes):
        '''Computes the logits for the given token state.
        
        This overrides a method from the Local class that is used to get
        inference results from the model.
        '''

        if len(token_ids) == 0:
            raise ValueError("token_ids must contain some tokens.")
        
        # compute the prompt bytes
        prompt = b''.join([self.tokens[i] for i in token_ids]) + forced_bytes

        self._shared_state["last_call"] = time.time()

        # keep looping until we have at least one more byte past our prompt
        token_id = None
        while True:

            # try and get the next token id
            if self._shared_state["data"].startswith(prompt):
                token_id = self._get_next_token(len(prompt)-len(forced_bytes))
                if token_id is not None:
                    break

            # restart if extending our data will never lead to matching our prompt
            elif len(self._shared_state["data"]) >= len(prompt) or not prompt.startswith(self._shared_state["data"]):
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
        
        return logits
    
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