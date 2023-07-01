import time
import collections
import regex
import pygtrie
import queue
import collections.abc
from ._llm import LLM, LLMSession, SyncSession

import torch
from exllama_lib.model import ExLlama as ExLlamaModel, ExLlamaConfig
from exllama_lib.tokenizer import ExLlamaTokenizer
from exllama_lib.generator import ExLlamaGenerator
from exllama_lib.tokenizer import SentencePieceProcessor

import warnings

class ExLlama(LLM):
    """ A HuggingFace transformers language model with Guidance support.
    """

    llm_name: str = "exllama"

    def __init__(self, model=None, generator=None, tokenizer=None, model_config=None, caching=False, token_healing=False, acceleration=False, \
                 temperature=0.01, **kwargs):
        super().__init__()

        self.model: ExLlamaModel = model
        self.model_obj: ExLlamaGenerator = generator
        self.tokenizer: ExLlamaTokenizer = tokenizer
        self._sentence_piece_processor: SentencePieceProcessor = self.tokenizer.tokenizer
        self.model_config: ExLlamaConfig = model_config

        # TODO:
        # Use temperature somehow

        self.caching = caching
        self.current_time = time.time()
        self.call_history = collections.deque()
        self.temperature = temperature
        self.token_healing = token_healing
        self.acceleration = acceleration

        self._token_prefix_map = self._build_token_prefix_map()

    def prefix_matches(self, prefix):
        """ Return the list of tokens that match the given prefix.
        """
        return [v for arr in self._token_prefix_map.values(prefix=prefix) for v in arr]

    def encode(self, string, **kwargs):
        return self.tokenizer.encode(string, **kwargs)
        
    def decode(self, tokens, **kwargs):
        return self.tokenizer.decode(tokens, **kwargs)
    
    def id_to_token(self, id):
        token = self._sentence_piece_processor.Decode(id)
        return token

    def token_to_id(self, token):
        print("Encoding...", token)
        return self._sentence_piece_processor.Encode(token)[-1]

    def end_of_text(self):
        return ""
        # return self.model_config.eos_token_id

    @staticmethod
    def role_start(role):
        raise NotImplementedError("In order to use chat role tags you need to use a chat-specific subclass of Transformers for your LLM from guidance.transformers.*!")

    def _build_token_prefix_map(self):
        """ Build a map from token to index."""
        token_map = pygtrie.CharTrie()

        for i in range(self.model_config.vocab_size):
            s = self.id_to_token(i)
            if s in token_map:
                token_map[s].append(i) # handle duplicate token encodings... (GPT2 BPE has this oddly enough)
            else:
                token_map[s] = [i]

        return token_map

    def autocomplete_options(self, prompt, options):
        def generate_bias_from_valid_characters(model_config, chars):
            encoded_tokens = [] 
            for char in list(set(chars)):
                encoded_tokens.append(self.tokenizer.encode(char))

            import torch
            logit_bias = torch.zeros([1, 1, model_config.vocab_size])

            for encoded_token in encoded_tokens:
                logit_bias[:, :, encoded_token] += 1000.0

            return logit_bias

        import pygtrie
        tree = pygtrie.CharTrie()

        # Fill tree with options paths
        for option in options:
            for idx in range(len(option)):
                    key = option[:idx]
                    if tree.has_key(key):
                        tree[key].append(option[idx:])
                    else:
                        tree[key] = [option[idx:]]

        first_char_options = []

        for option in options:
            first_char_options.append(option[0])

        logit_bias = generate_bias_from_valid_characters(self.model_config, first_char_options)
        prefix = ""
        option_fulfilled = False
        max_tokens = 10

        i = 0
        while not option_fulfilled and i < max_tokens:
            prefix += self.model_obj.generate_token_with_bias(prompt + prefix, logit_bias=logit_bias)
            suffixes_to_explore = tree[prefix]
            if len(suffixes_to_explore) == 1:
                prefix += suffixes_to_explore[0]
                option_fulfilled = True         
            else:
                valid_chars = []
                for suffix in suffixes_to_explore:
                    valid_chars.append(suffix[0])
                logit_bias = generate_bias_from_valid_characters(self.model_config, valid_chars)

            i += 1
        return prefix



    def session(self, asynchronous=False):
        if asynchronous:
            return ExLlamaSession(self)
        else:
            return SyncSession(ExLlamaSession(self))


class ExLlamaSession(LLMSession):
    def __init__(self, llm):
        super().__init__(llm)
        
        self._past_key_values = None
        self._prefix_cache = []
    
    def __enter__(self):

        # we only need decorators if we are using token acceleration
        if self.llm.acceleration:

            # decorate the prep step to preserve the initial past key values we have passed
            def prep_step_decorator(method):
                def decorate_prep_step(input_ids, **kwargs):

                    # if we are extending the input ids with the cached tokens then
                    # don't pass past key values to the input prep step, otherwise it
                    # would delete all but the last input_ids, and we have already removed
                    # the correct prefix from the input_ids (which is not always all but the last one)
                    if len(self._prefix_cache) > 0:
                        
                        kwargs["past"] = None
                        input_ids = input_ids[:,len(self._prefix_cache):]
                        # if "attention_mask" in kwargs:
                        #     kwargs["attention_mask"] = kwargs["attention_mask"][:,len(self._prefix_cache):]
                        model_kwargs = method(input_ids, **kwargs)

                        # provide the past key values for the actual model call
                        model_kwargs["past_key_values"] = self._past_key_values
                        if "position_ids" in model_kwargs: # models like OPT update the position ids internally
                            model_kwargs["position_ids"] = model_kwargs["position_ids"][:,len(self._prefix_cache):] # and update position ids

                        # we only need to do this first time, after that the past key values will
                        # be up until the last token, just like transformer models normally expect
                        # so we can clear our cache and let transformers cache like normal
                        self._prefix_cache = [] # this will get refilled once the generate call is done
                    
                        return model_kwargs
                    else:
                        return method(input_ids, **kwargs)
                decorate_prep_step.__func__ = method.__func__ # make us still look like a bound method
                return decorate_prep_step
            if getattr(self.llm.model_obj, "_orig_prepare_method", None) is None:
                self.llm.model_obj._orig_prepare_method = self.llm.model_obj.prepare_inputs_for_generation
            self.llm.model_obj.prepare_inputs_for_generation = prep_step_decorator(self.llm.model_obj._orig_prepare_method)

            # decorate the update step to save the past key values
            def update_step_decorator(method):
                def decorate_update_step(outputs, *args, **kwargs):

                    # save the past key values
                    self._past_key_values = getattr(outputs, "past_key_values", None)

                    return method(outputs, *args, **kwargs)
                return decorate_update_step
            if getattr(self.llm.model_obj, "_orig_update_method", None) is None:
                self.llm.model_obj._orig_update_method = self.llm.model_obj._update_model_kwargs_for_generation
            self.llm.model_obj._update_model_kwargs_for_generation = update_step_decorator(self.llm.model_obj._orig_update_method)

        return self
    
    async def __call__(self, prompt, stop=None, stop_regex=None, temperature=None, n=1, max_tokens=1000, logprobs=None,
                       top_p=1.0, echo=False, logit_bias=None, token_healing=None, pattern=None, stream=False,
                       cache_seed=0, caching=None, **generate_kwargs):
        """ Generate a completion of the given prompt.
        """
        
        # fill in defaults
        if temperature is None:
            temperature = self.llm.temperature
        if token_healing is None:
            token_healing = self.llm.token_healing

        self.llm: ExLlama
        # generate the cache key
        cache_params = self._cache_params(locals().copy())
        llm_cache = self.llm.cache
        key = llm_cache.create_key(self.llm.llm_name, **cache_params)

        # set the stop patterns
        if stop is not None:
            if isinstance(stop, str):
                stop_regex = [regex.escape(stop)]
            else:
                stop_regex = [regex.escape(s) for s in stop]
        if isinstance(stop_regex, str):
            stop_regex = [stop_regex]
        if stop_regex is None:
            stop_regex = []
        stop_regex.append(regex.escape(
            self.llm.id_to_token(self.llm.model_config.eos_token_id)
        )) # make sure the end of sequence token is always included

        # handle caching
        in_cache = key in llm_cache
        not_caching = (caching is not True and not self.llm.caching) or caching is False
        if not in_cache or not_caching:
            assert prompt != "", "You must provide a non-zero length prompt to the ExLlama language model!"

            # encode the prompt
            import torch
            # encoded2 = self.llm.encode([prompt for _ in range(n)], return_tensors="pt")
            encoded = self.llm.encode(prompt)
            encoded = torch.tensor(encoded)

            input_ids = encoded#["input_ids"]
            # attention_mask = encoded["attention_mask"]
            model_config: ExLlamaConfig = self.llm.model_config

            # ensure that we are extending a common sequence batch (our token healing assumes this right now)
            assert (input_ids[0,-1] == input_ids[:,-1]).all(), "The current token healing implementation assumes that batches are reps of the same sequence!"

            healed_token_ids = []
            stoppers = []

            # save what the prompt looks like when coded and then decoded (this captures added start tokens, etc.)
            coded_prompt = self.llm.decode(input_ids[0])

            # setup token healing
            if token_healing:
                raise TypeError("Token healing is not supported for ExLlama in this version")
                # healer = TokenHealingLogitsProcessor(self.llm, model_config.vocab_size, input_ids[0])
                # healed_token_ids = healer.healed_token_ids
                # if len(healed_token_ids) > 0:
                #     input_ids = input_ids[:,:-len(healed_token_ids)]
                #     # attention_mask = attention_mask[:,:-len(healed_token_ids)]
                #     max_tokens += len(healed_token_ids) # increase to account for the tokens we regen for token healing
                #     processors.append(healer)


            # find the max context length
            possible_attributes = ["max_sequence_length", "max_seq_len", "model_max_length", "n_positions", "max_position_embeddings"]
            max_context = None
            for obj in [model_config, self.llm.tokenizer]:
                for attr in possible_attributes:
                    if max_context is None:
                        max_context = getattr(obj, attr, None)
                    else:
                        break
            assert max_context is not None, "Could not find a max context length for the model! Tried: "+", ".join(possible_attributes)

            # make sure we don't run off the end of the model
            if max_tokens + len(input_ids[0]) > max_context:
                max_tokens = max_context - len(input_ids[0])

            # find how much of the prompt is cached
            prefix_match_len = 0
            for token in input_ids[0]:
                if prefix_match_len >= len(self._prefix_cache) or token != self._prefix_cache[prefix_match_len]:
                    break
                else:
                    prefix_match_len += 1

            # we always need to run the model on at least one token so transformers is happy
            if prefix_match_len == len(input_ids[0]):
                prefix_match_len -= 1

            # trim the cache to what we can use
            if prefix_match_len < len(self._prefix_cache): # prefix_match_len > 0 and 
                self._past_key_values = tuple((key[:,:,:prefix_match_len,:],value[:,:,:prefix_match_len,:]) for key,value in self._past_key_values) # TODO: this is specific to the GPT2 tensor layout
                self._prefix_cache = self._prefix_cache[:prefix_match_len]

            # add support for pattern guidance
            if pattern is not None:
                raise TypeError("Pattern is not supported by ExLlama in this version")
            
            if stop_regex is not None:
                # TODO: fix dependency on this
                warnings.warn("Using RegexStoppingCriteria, but it's not supported by ExLlama!")
                stoppers.append(RegexStoppingCriteria(stop_regex, self.llm, len(coded_prompt)))

            select_options=generate_kwargs.get("exllama_select_options")

            # a streamer to handle potentially partial output
            streamer = ExLlamaStreamer(
                input_ids=input_ids,
                stop_regex=stop_regex,
                healed_token_ids=healed_token_ids,
                prefix_length=len(coded_prompt),
                llm=self.llm,
                max_new_tokens=max_tokens,
                logprobs=logprobs,
                select_options=select_options,
            )

            # the args for the transformers generate call
            generate_args = dict(
                prefix=input_ids,
                logit_bias=logit_bias,
                max_new_tokens=max_tokens,

                # TODO: try to support more parameters below
                # Specially important are:
                # 1. stopping_criteria
                # 2. temperature

                # temperature=temperature,
                # top_p=top_p,
                # pad_token_id=model_config.pad_token_id if model_config.pad_token_id is not None else self.llm.tokenizer.eos_token_id,
                # stopping_criteria=transformers.StoppingCriteriaList(stoppers),
                # past_key_values=self._past_key_values,
                # output_scores=logprobs is not None and logprobs > 0,
                # return_dict_in_generate=True,
            )

            # TODO: support stream argument
            if select_options:
                token = self.llm.autocomplete_options(prompt=prompt, options=select_options)
                streamer.put(token)
                self.llm.cache[key] = streamer.__next__()
                self._update_prefix_cache(streamer)
            else:
                for token in self.llm.model_obj.generate_raw_stream_with_bias(**generate_args):
                    streamer.put(token)
                    self.llm.cache[key] = streamer.__next__()
                    self._update_prefix_cache(streamer)

        return llm_cache[key]
    
    def _update_prefix_cache(self, streamer):
        # note what we now have cached and ready for our next call in this session
        if self._past_key_values and len(streamer.generated_sequence) == 1:
            self._prefix_cache = streamer.generated_sequence[0][:self._past_key_values[0][0].shape[-2]] # self._past_key_values is already saved, this just aligns with it

    def _stream_then_save(self, streamer, key, thread):
        list_out = []
        for out in streamer:
            list_out.append(out)
            yield out
        thread.join() # clean up the thread
        self.llm.cache[key] = list_out
        self._update_prefix_cache(streamer)
        self._last_computed_key = key

    def __exit__(self, exc_type, exc_value, traceback):
        """ Restore the model to its original state by removing monkey patches.
        """
        if getattr(self.llm.model_obj, "_orig_prepare_method", None) is not None:
            self.llm.model_obj.prepare_inputs_for_generation = self.llm.model_obj._orig_prepare_method
            del self.llm.model_obj._orig_prepare_method
        if getattr(self.llm.model_obj, "_orig_update_method", None) is not None:
            self.llm.model_obj._update_model_kwargs_for_generation = self.llm.model_obj._orig_update_method
            del self.llm.model_obj._orig_update_method
        return False



class ExLlamaStreamer():
    def __init__(self, input_ids, stop_regex, healed_token_ids, prefix_length, llm, max_new_tokens, logprobs, select_options=None, timeout=None):

        self.input_ids = input_ids
        self.stop_regex = stop_regex
        self.healed_token_ids = healed_token_ids
        self.logprobs = logprobs
        self.llm: ExLlama = llm
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout
        self.str_pos = [prefix_length for i in range(len(self.input_ids))]
        self.out_queue = queue.Queue()
        self.input_length = len(input_ids[0])
        self.generated_sequence = []
        self.generated_string = [""]
        self.prefix_cache = []
        self.input_string = self.llm.tokenizer.decode(input_ids[0])
        self.select_options = select_options

    def put(self, token_obj):
        if self.select_options and isinstance(token_obj, str):
            for choice in self.select_options:
                if choice.lower() == token_obj:
                    self.out_queue.put({
                        "choices": [{
                            "text": token_obj,
                            "finish_reason": "match choice",
                            "stop_text": choice,
                            # TODO: fix these logprobs
                            "logprobs": {
                                "top_logprobs": [{token_obj: choice}]
                            },
                            "early_select_quit": True
                        }]
                    })
                    return

        new_tokens = None
        if isinstance(token_obj, torch.Tensor):
            new_tokens = token_obj.cpu()
        
        # if we are given a single sequence, then make it a batch of size 1
        if len(new_tokens.shape) == 1:
            new_tokens = new_tokens.unsqueeze(0)

        if len(self.input_ids) > 1:
            raise NotImplementedError("Batch size > 1 is not handled in ExLlamaStreamer")

        # print(new_tokens)
        out = {"choices": [None for i in range(len(self.input_ids))]}


        put_data = True

        # print("new_tokens", new_tokens)
        # print("input ids", self.input_ids)
        finish_reason = None
        self.generated_sequence.extend(new_tokens)
        self.generated_string.extend(self.llm.tokenizer.decode(new_tokens))
        
        # Add max tokens check
        # print("in here")
        # print("str pos", self.str_pos)
        # print("generated sequence", self.generated_sequence)
        # print("generated_string", self.generated_string)

        if len(self.generated_sequence) >= self.max_new_tokens:
            finish_reason = "length"

        elif self.generated_sequence[-1] == self.llm.tokenizer.eos_token_id:
            finish_reason = "endoftext"
            self.generated_string.pop() # remove the end of text token

        # print("generated_string", self.generated_string)
        output_string = " ".join(self.generated_string)
        # print("output_string", output_string)

        # trim off the stop regex matches if needed
        found_partial = False
        stop_text = None
        stop_pos = None
        if self.stop_regex is not None:# and (finish_reason is None or len(self.input_ids) > 1):
            stop_regex_obj = [regex.compile(s) for s in self.stop_regex]
            for s in stop_regex_obj:
                print("stop", s)
                print("output_string", output_string)
                m = s.search(output_string, partial=True)
                print("m", m)
                if m:
                    span = m.span()
                    if span[1] > span[0]:
                        if m.partial: # we might be starting a stop sequence, so we can't emit anything yet
                            # found_partial = True
                            break
                        else:
                            stop_text = output_string[span[0]:span[1]]
                            stop_pos = min(span[0], stop_pos)
                            break

        print("found partial", found_partial)
        print("stop_text", stop_text)
        print("stop_pos", stop_pos)
        # record the reason we stopped (if we have stopped)
        if stop_pos and stop_pos <= len(self.generated_string):
            finish_reason = "stop"
            # Remove text after stop pos
            output_string = output_string[:stop_pos]
        
        output_string = self.input_string + output_string
        if finish_reason is not None and not found_partial:
            out["choices"][0] = {
                "text": output_string,
                "finish_reason": finish_reason,
                "stop_text": stop_text
            }
            # Consider whether this boolean is still needed
            # put_data = True

        if put_data:
            self.out_queue.put(out)


    def end(self):

        # make sure we have flushed all of the data
        for i in range(len(self.input_ids)):
            assert self.str_pos[i] >= len(self.generated_string[i]), "Not all data was flushed, this means generation stopped for an unknown reason!"

        self.out_queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.out_queue.get(timeout=self.timeout)
        if value is None:
            raise StopIteration()
        else:
            return value


class RegexStoppingCriteria():
    def __init__(self, stop_pattern, llm, prefix_length):
        if isinstance(stop_pattern, str):
            self.stop_patterns = [regex.compile(stop_pattern)]
        else:
            self.stop_patterns = [regex.compile(pattern) for pattern in stop_pattern]
        self.prefix_length = prefix_length
        self.llm = llm
        self.current_strings = None
        self.current_length = 0

    def __call__(self, input_ids, scores, **kwargs):

        # handle 1D inputs
        if not isinstance(input_ids[0], collections.abc.Sequence) and not (hasattr(input_ids[0], "shape") and len(input_ids[0].shape) > 0):
            input_ids = [input_ids]

        # extend our current strings
        if self.current_strings is None:
            self.current_strings = [self.llm.new_string_builder() for _ in range(len(input_ids))]
        for i in range(len(self.current_strings)):
            self.current_strings[i].extend(input_ids[i][self.current_length:])
        
        self.current_length = len(input_ids[0])
        
        # check if all of the strings match a stop string (and hence we can stop the batch inference)
        all_done = True
        for i in range(len(self.current_strings)):
            found = False
            for s in self.stop_patterns:
                if s.search(str(self.current_strings[i])[self.prefix_length:]):
                    found = True
            if not found:
                all_done = False
                break
        
        return all_done