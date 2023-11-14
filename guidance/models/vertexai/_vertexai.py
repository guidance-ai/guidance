from .._model import Chat, Instruct
# from ._local import Local
from .._remote import Remote


try:
    # TODO: can we eliminate the torch requirement for llama.cpp by using numpy in the caller instead?
    import torch
    is_torch = True
except ImportError:
    is_torch = False

try:
    # TODO: can we eliminate the torch requirement for llama.cpp by using numpy in the caller instead?
    import vertexai
    is_vertexai = True
except ImportError:
    is_vertexai = False

class VertexAI(Remote):
    def __init__(self, model, tokenizer=None, echo=True, caching=True, temperature=0.0, max_streaming_tokens=500, **kwargs):
        if not is_vertexai:
            raise Exception("Please install the vertexai package using `pip install google-cloud-aiplatform` in order to use guidance.models.VertexAI!")
        
        # if we are called directly (as opposed to through super()) then we convert ourselves to a more specific subclass if possible
        if self.__class__ is VertexAI:
            found_subclass = None
            from .. import vertexai

            if isinstance(model, str):
                model_name = model
            else:
                model_name = self.model_obj._model_id

            # PaLM2Instruct
            if model_name == "text-bison@001":
                found_subclass = vertexai.PaLM2Instruct

            # PaLM2Chat
            elif model_name == "chat-bison@001":
                found_subclass = vertexai.PaLM2Chat
            
            # convert to any found subclass
            if found_subclass is not None:
                self.__class__ = found_subclass
                found_subclass.__init__(self, model, tokenizer=tokenizer, echo=echo, caching=caching, temperature=temperature, max_streaming_tokens=max_streaming_tokens, **kwargs)
                return # we return since we just ran init above and don't need to run again
        
            # make sure we have a valid model object
            if isinstance(model, str):
                raise Exception("The model ID you passed, `{model}`, does not match any known subclasses!")

        super().__init__(
            model, tokenizer=tokenizer, echo=echo,
            caching=caching, temperature=temperature,
            max_streaming_tokens=max_streaming_tokens, **kwargs
        )
        # self.caching = caching
        # self.temperature = temperature
        # self.max_streaming_tokens = max_streaming_tokens

        # # Vertex AI models don't always have public tokenizations, so when not provided we pretend they tokenize like gpt2...
        # if tokenizer is None:
        #     tokenizer = tiktoken.get_encoding("gpt2") # TODO: auto detect some common models in the model garden

        # # tiktoken tokenizer was given
        # if hasattr(tokenizer, "decode_single_token_bytes"):
        #     byte_tokens = [tokenizer.decode_single_token_bytes(i) for i in range(tokenizer.n_vocab)] + [b"<|endofprompt|>"]
        #     bos_token_id = None
        #     eos_token_id = len(byte_tokens)-2
        
        # # a transformer tokenizer was given that has a byte_decoder
        # elif hasattr(tokenizer, "byte_decoder"):
        #     byte_tokens = []
        #     for i in range(tokenizer.vocab_size):
        #         byte_coded = bytes([tokenizer.byte_decoder[c] for c in tokenizer.convert_ids_to_tokens(i)])
        #         byte_tokens.append(byte_coded)
        #     bos_token_id = tokenizer.bos_token_id
        #     eos_token_id = tokenizer.eos_token_id
        
        # # a transformer tokenizer was given with byte_decoder
        # elif hasattr(tokenizer, "convert_ids_to_tokens"):
        #     byte_tokens = [bytes(tokenizer.convert_tokens_to_string(['a', tokenizer.convert_ids_to_tokens(i)])[1:], encoding="utf8") for i in range(tokenizer.vocab_size)]
        #     bos_token_id = tokenizer.bos_token_id
        #     eos_token_id = tokenizer.eos_token_id

        # else:
        #     raise Exception("The tokenizer given was not of a recognized type!")

        # # build the 
        # super().__init__(
        #     byte_tokens,
        #     bos_token_id,
        #     eos_token_id,
        #     echo=echo
        # )

        # self._shared_state = {
        #     "data_queue": queue.Queue(), # this is where the streaming thread puts results
        #     "data": b'', # these are the bytes we are ready to use in the main thread
        #     "not_running_stream": threading.Event(), # this is phrased negatively so we can wait for the stop event
        #     "last_call": 0,
        #     "num_calls_made": 0
        # }
        # self._shared_state["not_running_stream"].set()

        # self.max_calls = 10
        # self.timeout = 5.1

    # def _running_stream(self):
    #     return not self._shared_state["not_running_stream"].is_set() # wrap double negation (which)

    # def _start_generator_stream(self, generator):
    #     dqueue = self._shared_state["data_queue"]
    #     first_iteration = True
    #     for chunk in generator:
    #         bchunk = chunk.text.encode("utf8")
    #         if len(bchunk) > 0:
    #             dqueue.put(bchunk)
    #         if self._shared_state["not_running_stream"].is_set() or not first_iteration and time.time() - self._shared_state["last_call"] > self.timeout:
    #             break
    #         first_iteration = False

    #     if self._running_stream():
    #         dqueue.put(self.tokens[self.eos_token_id])
    #     self._shared_state["not_running_stream"].set()
    #     dqueue.put(b'') # so we never get stuck waiting for a running stream to return something
        
    # def _start_new_stream(self, prompt):

    #     if self._shared_state["num_calls_made"] > self.max_calls:
    #         raise Exception(f"We have exceeded the maximum number of calls! {self.max_calls}")


    #     # stop any running stream
    #     if self._running_stream():
    #         self._shared_state["not_running_stream"].set() # stop the generator
    #         self._thread.join() # wait for the thread to finish

    #     # clear the data queue
    #     while not self._shared_state["data_queue"].empty(): 
    #         self._shared_state["data_queue"].get()

    #     # start the new stream
    #     prompt_end = prompt.find(b'<|endofprompt|>')
    #     if prompt_end >= 0:
    #         stripped_prompt = prompt[:prompt_end]
    #     else:
    #         raise Exception("This model cannot handle prompts that don't match the instruct format!")
    #     self._shared_state["not_running_stream"].clear() # so we know we are running
    #     # self._stop_event.clear()
    #     self._shared_state["data"] = stripped_prompt + b'<|endofprompt|>'# we start with this data
    #     self._shared_state["num_calls_made"] += 1
    #     generator = self.model_obj.predict_streaming(prompt.decode("utf8"), max_output_tokens=self.max_streaming_tokens)
    #     self._thread = threading.Thread(target=self._start_generator_stream, args=(generator,))
    #     self._thread.start()
    #     # self._start_generator_stream(stripped_prompt)

    # def _start_new_stream(self, prompt):

    #     if self._shared_state["num_calls_made"] > self.max_calls:
    #         raise Exception(f"We have exceeded the maximum number of calls! {self.max_calls}")


    #     # stop any running stream
    #     if self._running_stream():
    #         self._shared_state["not_running_stream"].set() # stop the generator
    #         self._thread.join() # wait for the thread to finish

    #     # clear the data queue
    #     while not self._shared_state["data_queue"].empty(): 
    #         self._shared_state["data_queue"].get()

    #     # start the new stream
    #     generator = self._generator(prompt)
    #     self._shared_state["not_running_stream"].clear() # so we know we are running
    #     self._shared_state["num_calls_made"] += 1

    #     self._thread = threading.Thread(target=self._start_generator_stream, args=(generator,))
    #     self._thread.start()
    #     # self._start_generator_stream(stripped_prompt)
    
    # def _get_logits(self, token_ids, forced_bytes):
    #     '''Computes the logits for the given token state.
        
    #     This overrides a method from the Local class that is used to get
    #     inference results from the model.
    #     '''

    #     if len(token_ids) == 0:
    #         raise ValueError("token_ids must contain some tokens.")
        
    #     # compute the prompt bytes
    #     prompt = b''.join([self.tokens[i] for i in token_ids]) + forced_bytes

    #     self._shared_state["last_call"] = time.time()

    #     # keep looping until we have at least one more byte past our prompt
    #     token_id = None
    #     while True:

    #         # try and get the next token id
    #         if self._shared_state["data"].startswith(prompt):
    #             token_id = self._get_next_token(len(prompt)-len(forced_bytes))
    #             if token_id is not None:
    #                 break
    #         # if self._shared_state["data"].startswith(prompt):
    #         #     token_id = self._get_next_token(len(prompt))#or len(self._shared_state["data"]) <= len(prompt):
    #         #     continue

    #         # we need to restart if extending our data will never lead to matching our prompt
    #         if len(self._shared_state["data"]) <= len(prompt) and not prompt.startswith(self._shared_state["data"]):
    #             self._start_new_stream(prompt)

    #         # extend our data with a chunk from the model stream
    #         if not self._shared_state["data_queue"].empty():
    #             new_bytes = self._shared_state["data_queue"].get_nowait()
                
    #             # if we are at the end of the generation then we try again allowing for early token stopping
    #             if len(new_bytes) == 0:
    #                 token_id = self._get_next_token(len(prompt), allow_early_stop=True)
    #                 if token_id is not None:
    #                     break
    #             self._shared_state["data"] += new_bytes
            
    #         # but if there is nothing and we are not running then we start a stream
    #         elif self._shared_state["not_running_stream"].is_set():
    #             self._start_new_stream(prompt)

    #         # we wait for the running stream to put something in the queue
    #         else:
    #             self._shared_state["data"] += self._shared_state["data_queue"].get()
        
    #     # # if we don't have the next byte of data yet then we wait for it (from the streaming thread)
    #     # if len(self._shared_state["data"]) == len(prompt):
    #     #     self._shared_state["data"] += self._shared_state["data"]_queue.get() 

    #     # token_id = self._get_next_token(len(prompt))

    #     # set the logits to the next byte the model picked
    #     logits = np.ones(len(self.tokens)) * np.nan
    #     logits[token_id] = 100
        
    #     self._shared_state["last_call"] = time.time()
    #     return torch.tensor(logits) # TODO: the caller need to know to NOT use the 0 entries, but to fail out if that happens
    
    # def _get_next_token(self, pos, allow_early_stop=False):
    #     data = self._shared_state["data"]
    #     trie = self._token_trie
    #     token_id = None
    #     while True:
            
    #         # see if we have run out of data
    #         if pos >= len(data):
    #             if allow_early_stop:
    #                 return token_id
    #             else:
    #                 return None
            
    #         # try and walk down the trie
    #         next_byte = data[pos:pos+1]
    #         if next_byte in trie.children:
    #             trie = trie.children[next_byte]
    #             pos += 1
    #             if trie.value is not None:
    #                 token_id = trie.value
    #         else:
    #             return token_id # this is the longest greedy token match we can make

    # def _call_end(self):
    #     if 
    #     self._stop_event.set() # stop the generator
    #     self._generator = None
    #     self._thread.join() # stop the streaming thread

class VertexAIInstruct(VertexAI, Instruct):

    def get_role_start(self, name):
        return ""
    
    def get_role_end(self, name):
        if name == "instruction":
            return "<|endofprompt|>"
        else:
            raise Exception(f"The VertexAIInstruct model does not know about the {name} role type!")

    def _generator(self, prompt):
        # start the new stream
        prompt_end = prompt.find(b'<|endofprompt|>')
        if prompt_end >= 0:
            stripped_prompt = prompt[:prompt_end]
        else:
            raise Exception("This model cannot handle prompts that don't match the instruct format!")
        self._shared_state["not_running_stream"].clear() # so we know we are running
        self._shared_state["data"] = stripped_prompt + b'<|endofprompt|>'# we start with this data
        for chunk in self.model_obj.predict_streaming(self._shared_state["data"].decode("utf8"), max_output_tokens=self.max_streaming_tokens):
            yield chunk.text.encode("utf8")
    
    # def _get_next_token(self, pos, allow_early_stop=False):
    #     data = self._shared_state["data"]
    #     trie = self._token_trie
    #     token_id = None
    #     while True:
            
    #         # see if we have run out of data
    #         if pos >= len(data):
    #             if allow_early_stop:
    #                 return token_id
    #             else:
    #                 return None
            
    #         # try and walk down the trie
    #         next_byte = data[pos:pos+1]
    #         if next_byte in trie.children:
    #             trie = trie.children[next_byte]
    #             pos += 1
    #             if trie.value is not None:
    #                 token_id = trie.value
    #         else:
    #             return token_id # this is the longest greedy token match we can make

    # def _call_end(self):
    #     if 
    #     self._stop_event.set() # stop the generator
    #     self._generator = None
    #     self._thread.join() # stop the streaming thread

class VertexAIChat(VertexAI, Chat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _generator(self, prompt):
        
        # find the system text
        pos = 0
        system_start = b'<|im_start|>system\n'
        user_start = b'<|im_start|>user\n'
        assistant_start = b'<|im_start|>assistant\n'
        role_end = b'<|im_end|>'
        # system_start_pos = prompt.startswith(system_start)
        
        # find the system text
        system_text = b''
        if prompt.startswith(system_start):
            pos += len(system_start)
            system_end_pos = prompt.find(role_end)
            system_text = prompt[pos:system_end_pos]
            pos = system_end_pos + len(role_end)

        # find the user/assistant pairs
        user_assistant_pairs = []
        last_user_text = b''
        while True:

            # find the user text
            if prompt[pos:].startswith(user_start):
                pos += len(user_start)
                user_end_pos = prompt[pos:].find(role_end)
                user_text = prompt[pos:pos+user_end_pos]
                pos += user_end_pos + len(role_end)
            else:
                raise Exception(f"Bad chat format! Expected a user start tag where there was none at prompt byte position: {pos}")
            
            # and the assistant text
            if prompt[pos:].startswith(assistant_start):
                pos += len(assistant_start)
                assistant_end_pos = prompt[pos:].find(role_end)
                if assistant_end_pos < 0:
                    last_user_text = user_text
                    break
                else:
                    assistant_text = prompt[pos:pos+assistant_end_pos]
                    pos += assistant_end_pos + len(role_end)
                    user_assistant_pairs.append((user_text, assistant_text))
            else:
                raise Exception(f"Bad chat format! Expected a assistant start tag where there was none at prompt byte position: {pos}")
            
        self._shared_state["data"] = prompt[:pos]
        
        chat = self.model_obj.start_chat(
            context=system_text.decode("utf8"),
            examples=[
                vertexai.language_models.InputOutputTextPair(
                    input_text=pair[0].decode("utf8"),
                    output_text=pair[1].decode("utf8"),
                )
            for pair in user_assistant_pairs],
        )

        generator = chat.send_message_streaming(
            last_user_text.decode("utf8"),
            max_output_tokens=self.max_streaming_tokens
        )

        for chunk in generator:
            yield chunk.text.encode("utf8")