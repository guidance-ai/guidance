import threading
import numpy as np
import queue
import time
import tiktoken
import re
import logging
from ._model import Model, format_pattern, ConstraintException

logger = logging.getLogger(__name__)


class Remote(Model):
    def __init__(self, model, tokenizer=None, echo=True, caching=True, temperature=0.0, top_p=1.0, max_streaming_tokens=None, timeout=0.5, **kwargs):
        logger.debug(f"start Remote.__init__(model=\"{model}\")")
        self.caching = caching
        self.temperature = temperature
        self.top_p = top_p
        self.max_streaming_tokens = max_streaming_tokens
        self.timeout = timeout

        # Remote models don't always have public tokenizations, so when not provided we pretend they tokenize like gpt2...
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("gpt2")

        # tiktoken tokenizer was given
        if hasattr(tokenizer, "decode_single_token_bytes"):
            special_map = {v: k for k,v in tokenizer._special_tokens.items()}
            byte_tokens = [] # b'<|invalid_special_token|>'
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

        # a HuggingFace tokenizers tokenizer was given with id_to_token
        elif hasattr(tokenizer, "id_to_token"):
            a_token_ids = tokenizer.encode("a").ids
            if len(a_token_ids) == 3:
                bos_token_id = a_token_ids[0]
                a_id = a_token_ids[1]
                eos_token_id = a_token_ids[2]
            else:
                raise Exception("This tokenizer does not seem to have a BOS and EOS, support for this need to be implemented still.")

            byte_tokens = [bytes(tokenizer.decode([a_id, i])[1:], encoding="utf8") for i in range(tokenizer.get_vocab_size())]
            for i,b in enumerate(byte_tokens):
                if b == b'':
                    byte_tokens[i] = bytes(tokenizer.id_to_token(i), encoding="utf8")

        else:
            raise Exception("The tokenizer given was not of a recognized type!")

        # build the 
        super().__init__(
            tokens=byte_tokens,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
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

        self.max_repeated_calls = 10
        self.timeout = timeout
        logger.debug(f"finish Remote.__init__")

    def __call__(self, grammar, max_tokens=1000000, n=1, top_p=1, temperature=0.0, ensure_bos_token=True):
        self._shared_state["num_calls_made"] = 0 # reset the number of calls count so we only limit the number of calls within a single grammar execution
        return super().__call__(grammar, max_tokens=max_tokens, n=n, top_p=top_p, temperature=temperature, ensure_bos_token=ensure_bos_token)

    def _running_stream(self):
        return not self._shared_state["not_running_stream"].is_set() # wrap double negation (which)

    def _start_generator_stream(self, generator):
        logger.debug(f"start Remote._start_generator_stream")
        dqueue = self._shared_state["data_queue"]
        first_iteration = True
        try: 
            for chunk in generator:
                logger.debug(f"Got chunk: " + str(chunk))
                if len(chunk) > 0:
                    dqueue.put(chunk)
                if self._shared_state["not_running_stream"].is_set() or not first_iteration and time.time() - self._shared_state["last_call"] > self.timeout:
                    break
                first_iteration = False
        
        # we pass any exceptions back to the main thread
        except Exception as e:
            self._shared_state["not_running_stream"].set()
            while not dqueue.empty(): 
                dqueue.get()
            dqueue.put(e)

        if self._running_stream():
            dqueue.put(self.tokens[self.eos_token_id])
        self._shared_state["not_running_stream"].set()
        dqueue.put(b'') # so we never get stuck waiting for a running stream to return something

    def _start_new_stream(self, prompt):

        if self._shared_state["num_calls_made"] > self.max_repeated_calls:
            raise Exception(f"We have exceeded the maximum number of repeat calls ({self.max_repeated_calls}) per grammar execution!")

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
        self._shared_state["data"] = prompt # we reset out current data state to be this prompt
        self._shared_state["remote_thread"] = threading.Thread(target=self._start_generator_stream, args=(generator,))
        self._shared_state["remote_thread"].start()
    
    def _get_logits(self, token_ids, forced_bytes):
        '''Computes the logits for the given token state.
        
        This overrides a method from the Local class that is used to get
        inference results from the model.
        '''

        logger.debug(f"start Remote._get_logits(token_ids={token_ids})")

        if len(token_ids) == 0:
            raise ValueError("token_ids must contain some tokens.")
        
        # compute the prompt bytes
        prompt = b''.join([self.tokens[i] for i in token_ids]) + forced_bytes

        self._shared_state["last_call"] = time.time()

        # keep looping until we have at least one more byte past our prompt
        token_id = None
        restarted = False # track if we have restarted the data stream during this call
        while True:

            # try and get the next token id
            if self._shared_state["data"].startswith(prompt):
                token_id = self._get_next_token(len(prompt)-len(forced_bytes))
                if token_id is not None:
                    break

            # restart if extending our data will never lead to matching our prompt
            elif not self._shared_state["data"].startswith(prompt) and len(self._shared_state["data"]) >= len(prompt): #not prompt.startswith(self._shared_state["data"]): # len(self._shared_state["data"]) >= len(prompt) or 

                # check if we have already restarted once and so retrying by default is not likely to be helpful
                if restarted:
                    raise self._report_failed_match(prompt)

                # check the length of the prefix match
                match_len = 0
                found_mismatch = False
                data = self._shared_state["data"]
                for match_len,v in enumerate(prompt):
                    if v != data[match_len]:
                        found_mismatch = True
                        break
                if not found_mismatch:
                    match_len = len(prompt)
                leftover = prompt[match_len:]

                # record any active non-empty role ends. Ignore role ends that are spaces
                parts = []
                for _,role_end_str in self.opened_blocks.values():
                    role_end_str = format_pattern.sub("", role_end_str)
                    if len(role_end_str) > 0 and not re.fullmatch(r'\s+', role_end_str):
                        parts.append(role_end_str.encode("utf8"))

                # record the eos token
                parts.append(self.eos_token)

                # see if adding an end token would work here (if so we avoid recalling the server and just produce an end token)
                found_match = False
                for p in parts:
                    if p.startswith(leftover):
                        self._shared_state["data"] = self._shared_state["data"][:match_len] + p
                        logger.debug(f'automatically adding an end token since it fits the forcing of the grammar')
                        found_match = True
                        break
                if found_match:
                    continue # start our loop over again

                logger.debug(f'restarting a stream because the data we have does not match the ids. We have {str(self._shared_state["data"])} but the prompt is {str(prompt)}')
                restarted = True
                self._start_new_stream(prompt)

            # extend our data with a chunk from the model stream
            if not self._shared_state["data_queue"].empty():
                new_bytes = self._shared_state["data_queue"].get_nowait()
                if isinstance(new_bytes, Exception):
                    raise new_bytes
                
                # if we are at the end of the generation then we try again allowing for early token stopping
                if len(new_bytes) == 0:
                    token_id = self._get_next_token(len(prompt), allow_early_stop=True)
                    if token_id is not None:
                        break
                self._shared_state["data"] += new_bytes
            
            # but if there is nothing and we are not running then we start a stream
            elif self._shared_state["not_running_stream"].is_set():
                logger.debug("starting a new stream because there is no data to read and no stream running...")
                restarted = True
                self._start_new_stream(prompt)

            # we wait for the running stream to put something in the queue
            else:
                self._shared_state["last_call"] = 10e9 # set to essentialy infinity so we don't stop the data stream while we are waiting for it
                new_bytes = self._shared_state["data_queue"].get()
                if isinstance(new_bytes, Exception):
                    raise new_bytes
                self._shared_state["data"] += new_bytes
                self._shared_state["last_call"] = time.time() # reset out call time to allow the data stream to time out if we happen to be done with it
        
        # # if we don't have the next byte of data yet then we wait for it (from the streaming thread)
        # if len(self._shared_state["data"]) == len(prompt):
        #     self._shared_state["data"] += self._shared_state["data"]_queue.get() 

        # token_id = self._get_next_token(len(prompt))

        # set the logits to the next byte the model picked
        logits = np.ones(len(self.tokens)) * -np.inf
        logits[token_id] = 100
        
        return logits
    
    def _report_failed_match(self, prompt):

        # check the length of the prefix match
        match_len = 0
        found_mismatch = False
        data = self._shared_state["data"]
        for match_len,v in enumerate(prompt):
            if v != data[match_len]:
                found_mismatch = True
                break
        if not found_mismatch:
            match_len = len(prompt)
        leftover = prompt[match_len:]

        # compute the mismatch parts
        data_after_prompt = self._shared_state["data"][match_len:]
        if len(data_after_prompt) > 40:
            data_after_prompt = data_after_prompt[:40] + b"..."
        prompt_tail = prompt[:match_len]
        if len(prompt_tail) > 40:
            prompt_tail = b"..." + prompt_tail[-40:]

        # show in the model output where and how we diverged from the grammar
        try:
            # just for display when echo is on
            already_shown = len(self._current_prompt().encode())
            self += self._shared_state["data"][already_shown:match_len].decode() + f"<||_html:<span style='color: rgba(165,0,0,1);' title='{leftover}'><span style='text-decoration: underline;'>{data_after_prompt.decode()}</span></span>_||>"
        except:
            pass # could not decode the data the model generated into a string...
        
        # create an exception for users to deal with (that our caller can throw)
        return ConstraintException(
            f"The model attempted to generate {str(data_after_prompt)} after the prompt `{prompt_tail}`, but that does\n" +
            "not match the given grammar constraints! Since your model is a remote API that does not support full guidance\n" +
            "integration we cannot force the model to follow the grammar, only flag an error when it fails to match.\n" +
            "You can try to address this by improving the prompt, making your grammar more flexible, rerunning with\n" +
            "a non-zero temperature, or using a model that supports full guidance grammar constraints."
        )
    
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
            if trie.has_child(next_byte):
                trie = trie.child(next_byte)
                pos += 1
                if trie.value >= 0:
                    token_id = trie.value
            else:
                return token_id # this is the longest greedy token match we can make