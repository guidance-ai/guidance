import openai
import pathlib
import diskcache
import os
import time
import requests
import warnings
import time
import collections
import tiktoken
import asyncio
import re
import pygtrie
curr_dir = pathlib.Path(__file__).parent.resolve()
_file_cache = diskcache.Cache(f"{curr_dir}/_transfomers.diskcache")


class TransformersSession():
    def __init__(self, model):
        import torch
        self._model = model
        self._past_key_values = None
        self._cached_prefix = ""
        self._cached_tokens = torch.zeros(0)
        self._past_key_values = None
    
    def __enter__(self):
        return self

    def __call__(self, prompt, stop=None, temperature=None, n=1, max_tokens=1000, logprobs=None, top_p=1.0, echo=False, logit_bias=None, stream=False, cache_seed=0):
        """ Generate a completion of the given prompt.
        """

        # fill in defaults
        if temperature is None:
            temperature = self._model.temperature

        # handle caching
        key = "_---_".join([str(v) for v in (self._model.model_name, prompt, stop, temperature, n, max_tokens, logprobs, echo, logit_bias, cache_seed)])
        if key not in _file_cache or not self.caching:

            # actually run the model
            out = self.sample(prompt, stop=None, temperature=temperature, n=1, max_tokens=max_tokens, logprobs=None, top_p=top_p, echo=False, logit_bias=None, stream=stream)

            if stream:
                return self.stream_then_save(out, key)
            else:
                _file_cache[key] = out
        return _file_cache[key]
    
    def __exit__(self, exc_type, exc_value, traceback):
        return False
    
    def sample(self, prompt, stop, temperature, n, max_tokens, logprobs, top_p, echo, logit_bias, stream,
               pad_token_id=None,
               eos_token_id=None,
               synced_gpus=False,
               **model_kwargs):
        import torch
        import torch.distributed as dist
        import transformers

        # encode the prompt
        input_ids = self._model._encoding.encode(prompt, return_tensors="pt")
        max_tokens += len(input_ids[0])

        # when we tokenize the prompt the last token we get is not the last token
        last_token_str = self._model._encoding.decode(input_ids[0][-1])
        allowed_first_tokens = self._model.token_prefix_map.values(prefix=last_token_str)

        # find how much of the prompt is cached
        for prefix_match_len, token in enumerate(input_ids):
            if prefix_match_len >= len(self._cached_tokens) or token != self._cached_tokens[prefix_match_len]:
                break

        # Create logits processor list
        logits_processor_list = []
        if top_p != 1:
            logits_processor_list.append(transformers.TopPLogitsWarper(top_p))
        if temperature != 0:
            logits_processor_list.append(transformers.TemperatureLogitsWarper(temperature))
        logits_processor = transformers.LogitsProcessorList(logits_processor_list)

        # Create stopping criteria list
        stopping_criteria = transformers.StoppingCriteriaList([
            transformers.MaxLengthCriteria(max_tokens)
        ])

        # init values
        model_config = self._model.model_obj.config
        pad_token_id = pad_token_id if pad_token_id is not None else model_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else model_config.eos_token_id
        if pad_token_id is None and eos_token_id is not None:
            # if model_kwargs.get("attention_mask", None) is None:
            #     logger.warning(
            #         "The attention mask and the pad token id were not set. As a consequence, you may observe "
            #         "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
            #     )
            # logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            pad_token_id = eos_token_id
        # output_scores = output_scores if output_scores is not None else model_config.output_scores
        # output_attentions = output_attentions if output_attentions is not None else model_config.output_attentions
        # output_hidden_states = (
        #     output_hidden_states if output_hidden_states is not None else model_config.output_hidden_states
        # )
        # return_dict_in_generate = (
        #     return_dict_in_generate if return_dict_in_generate is not None else model_config.return_dict_in_generate
        # )

        # init attention / hidden states / scores tuples
        scores = None
        # scores = () if (return_dict_in_generate and output_scores) else None
        # decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        # cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        # decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        # if return_dict_in_generate and model_config.is_encoder_decoder:
        #     encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        #     encoder_hidden_states = (
        #         model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        #     )

        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        cur_len = input_ids.shape[-1]

        this_peer_finished = False  # used by synced_gpus only
        # auto-regressive generation
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            # model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self._model.model_obj(
                **{"input_ids": input_ids},
                return_dict=True,
                # output_attentions=output_attentions,
                # output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            # next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            # if return_dict_in_generate:
            #     if output_scores:
            #         scores += (next_token_scores,)
            #     if output_attentions:
            #         decoder_attentions += (
            #             (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
            #         )
            #         if self.config.is_encoder_decoder:
            #             cross_attentions += (outputs.cross_attentions,)

            #     if output_hidden_states:
            #         decoder_hidden_states += (
            #             (outputs.decoder_hidden_states,)
            #             if self.config.is_encoder_decoder
            #             else (outputs.hidden_states,)
            #         )

            if temperature == 0.0: # greedy sampling
                next_tokens = torch.argmax(next_token_scores, dim=-1)
            else: # sample
                probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=model_config.is_encoder_decoder
            )
            cur_len = cur_len + 1

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        # save the key values in our cache
        self._past_key_values = outputs.past_key_values
        self._cached_tokens = input_ids[0][:-1] # we don't have the KVs for the last token (it's not generated yet)


        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return SampleEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return SampleDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids
    
    @staticmethod
    def _update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder = False):
        """ From transformers.
        
        (we have to reimplement because https://github.com/huggingface/transformers/pull/17574 never got merged)
        """
        import torch
        # update past
        if "past_key_values" in outputs:
            model_kwargs["past"] = outputs.past_key_values
        elif "mems" in outputs:
            model_kwargs["past"] = outputs.mems
        elif "past_buckets_states" in outputs:
            model_kwargs["past"] = outputs.past_buckets_states
        else:
            model_kwargs["past"] = None

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        # update attention mask
        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        return model_kwargs

class Transformers():
    """ A HuggingFace transformers language model with Guidance support.
    """

    def __init__(self, model=None, caching=True, temperature=0.0):

        try:
            import transformers
        except:
            raise Exception("Please install transformers with `pip install transformers` in order to use guidance.llms.Transformers!")

        # fill in default model value
        if model is None:
            model = os.environ.get("TRANSFORMERS_MODEL", None)
        if model is None:
            try:
                with open(os.path.expanduser('~/.transformers_model'), 'r') as file:
                    model = file.read().replace('\n', '')
            except:
                pass

        self._encoding = transformers.AutoTokenizer.from_pretrained(model)
        
        self.model_name = model
        self.model_obj = transformers.AutoModelForCausalLM.from_pretrained(model)
        self.caching = caching
        self.current_time = time.time()
        self.call_history = collections.deque()
        self.temperature = temperature

        self.token_prefix_map = self._build_token_prefix_map(model)

    def _build_token_prefix_map(self, model_name):
        """ Build a map from token to index.
        """
        token_map = pygtrie.CharTrie()
        for i in range(self._encoding.vocab_size):
            token_map[self._encoding.decode([i])] = i

        return token_map

    def session(self):
        return TransformersSession(self)
    
    # def __call__(self, prompt, stop=None, temperature=None, n=1, max_tokens=1000, logprobs=None, top_p=1.0, echo=False, logit_bias=None, stream=False, cache_seed=0):
    #     """ Generate a completion of the given prompt.
    #     """

    #     if temperature is None:
    #         temperature = self.temperature

    #     key = "_---_".join([str(v) for v in (self.model, prompt, stop, temperature, n, max_tokens, logprobs, echo, logit_bias, cache_seed)])
    #     if key not in _file_cache or not self.caching:

    #         # ensure we don't exceed the rate limit
    #         if self.count_calls() > self.max_calls_per_min:
    #             time.sleep(1)        

    #         fail_count = 0
    #         while True:
    #             try_again = False
    #             try:
    #                 self.add_call()
    #                 call_args = {
    #                     "model": self.model,
    #                     "prompt": prompt,
    #                     "max_tokens": max_tokens,
    #                     "temperature": temperature,
    #                     "top_p": top_p,
    #                     "n": n,
    #                     "stop": stop,
    #                     "logprobs": logprobs,
    #                     "echo": echo,
    #                     "stream": stream
    #                 }
    #                 if logit_bias is not None:
    #                     call_args["logit_bias"] = logit_bias
    #                 out = self.caller(**call_args)

    #             except openai.error.RateLimitError:
    #                 time.sleep(3)
    #                 try_again = True
    #                 fail_count += 1
                
    #             if not try_again:
    #                 break

    #             if fail_count > self.max_retries:
    #                 raise Exception(f"Too many (more than {self.max_retries}) OpenAI API RateLimitError's in a row!")

    #         if stream:
    #             return self.stream_then_save(out, key)
    #         else:
    #             _file_cache[key] = out
    #     return _file_cache[key]
    
    def stream_then_save(self, gen, key):
        list_out = []
        for out in gen:
            list_out.append(out)
            yield out
        _file_cache[key] = list_out
    
    # def _stream_completion(self):
    #     pass

    # # Define a function to add a call to the deque
    # def add_call(self):
    #     # Get the current timestamp in seconds
    #     now = time.time()
    #     # Append the timestamp to the right of the deque
    #     self.call_history.append(now)

    # # Define a function to count the calls in the last 60 seconds
    # def count_calls(self):
    #     # Get the current timestamp in seconds
    #     now = time.time()
    #     # Remove the timestamps that are older than 60 seconds from the left of the deque
    #     while self.call_history and self.call_history[0] < now - 60:
    #         self.call_history.popleft()
    #     # Return the length of the deque as the number of calls
    #     return len(self.call_history)

    # def _library_call(self, **kwargs):
    #     """ Call the OpenAI API using the python package.

    #     Note that is uses the local auth token, and does not rely on the openai one.
    #     """
    #     prev_key = openai.api_key
    #     openai.api_key = self.token
    #     if self.chat_completion:
    #         kwargs['messages'] = prompt_to_messages(kwargs['prompt'])
    #         del kwargs['prompt']
    #         del kwargs['echo']
    #         del kwargs['logprobs']
    #         print(kwargs)
    #         out = openai.ChatCompletion.create(**kwargs)
    #         add_text_to_chat_completion(out)
    #     else:
    #         out = openai.Completion.create(**kwargs)
    #     openai.api_key = prev_key
    #     return out

    # def _rest_call(self, **kwargs):
    #     """ Call the OpenAI API using the REST API.
    #     """

    #     # Define the request headers
    #     headers = {
    #         'Content-Type': 'application/json'
    #     }
    #     if self.token is not None:
    #         headers['Authorization'] = f"Bearer {self.token}"

    #     # Define the request data
    #     data = {
    #         "prompt": kwargs["prompt"],
    #         "max_tokens": kwargs.get("max_tokens", None),
    #         "temperature": kwargs.get("temperature", 0.0),
    #         "top_p": kwargs.get("top_p", 1.0),
    #         "n": kwargs.get("n", 1),
    #         "stream": False,
    #         "logprobs": kwargs.get("logprobs", None),
    #         'stop': kwargs.get("stop", None),
    #         "echo": kwargs.get("echo", False)
    #     }
    #     if self.chat_completion:
    #         data['messages'] = prompt_to_messages(data['prompt'])
    #         del data['prompt']
    #         del data['echo']
    #         del data['stream']

    #     # Send a POST request and get the response
    #     response = requests.post(self.endpoint, headers=headers, json=data)
    #     if response.status_code != 200:
    #         raise Exception("Response is not 200: " + response.text)
    #     response = response.json()
    #     if self.chat_completion:
    #         add_text_to_chat_completion(response)
    #     return response
    
    def encode(self, string):
        return self._encoding.encode(string)
    
    def decode(self, tokens):
        return self._encoding.decode(tokens)

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


