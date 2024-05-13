import os
import typing

import diskcache as dc
import hashlib
import platformdirs
import tiktoken


from ._model import Chat, Instruct
from ._grammarless import GrammarlessEngine, Grammarless

try:
    import openai

    client_class: typing.Optional[typing.Type[openai.OpenAI]] = openai.OpenAI
except ImportError:
    client_class = None


class OpenAIEngine(GrammarlessEngine):
    def __init__(
        self,
        tokenizer,
        max_streaming_tokens,
        timeout,
        compute_log_probs,
        model,
        client_class=client_class,
        **kwargs,
    ):

        if client_class is None:
            raise Exception(
                "Please install the openai package version >= 1 using `pip install openai -U` in order to use guidance.models.OpenAI!"
            )

        self.client = client_class(**kwargs)
        self.model_name = model

        if tokenizer is None:
            tokenizer = tiktoken.encoding_for_model(model)

        super().__init__(tokenizer, max_streaming_tokens, timeout, compute_log_probs)


class OpenAI(Grammarless):
    def __init__(
        self,
        model,
        tokenizer=None,
        echo=True,
        api_key=None,
        max_streaming_tokens=1000,
        timeout=0.5,
        compute_log_probs=False,
        engine_class=None,
        **kwargs,
    ):
        """Build a new OpenAI model object that represents a model in a given state.

        This class automatically subclasses itself into the appropriate OpenAIChat, OpenAIInstruct,
        or OpenAICompletion subclass based on the model name.

        Parameters
        ----------
        model : str
            The name of the OpenAI model to use (e.g. gpt-3.5-turbo).
        tokenizer : None or tiktoken.Encoding
            The tokenizer to use for the given model. If set to None we use `tiktoken.encoding_for_model(model)`.
        echo : bool
            If true the final result of creating this model state will be displayed (as HTML in a notebook).
        api_key : None or str
            The OpenAI API key to use for remote requests, passed directly to the `openai.OpenAI` constructor.
        max_streaming_tokens : int
            The maximum number of tokens we allow this model to generate in a single stream. Normally this is set very
            high and we rely either on early stopping on the remote side, or on the grammar terminating causing the
            stream loop to break on the local side. This number needs to be longer than the longest stream you want
            to generate.
        **kwargs :
            All extra keyword arguments are passed directly to the `openai.OpenAI` constructor. Commonly used argument
            names include `base_url` and `organization`
        """

        if client_class is None:
            raise Exception(
                "Please install the openai package version >= 1 using `pip install openai -U` in order to use guidance.models.OpenAI!"
            )

        # if we are called directly (as opposed to through super()) then we convert ourselves to a more specific subclass if possible
        if self.__class__ is OpenAI:

            # instruct
            # elif "instruct" in model: # All current OpenAI instruct models behave as Completion models.
            #     found_subclass = OpenAIInstruct

            found_subclass: typing.Type[OpenAI] = (
                OpenAICompletion if model.endswith("-instruct") else OpenAIChat
            )

            # convert to any found subclass
            self.__class__ = found_subclass
            found_subclass.__init__(
                self,
                model,
                tokenizer=tokenizer,
                echo=echo,
                api_key=api_key,
                max_streaming_tokens=max_streaming_tokens,
                **kwargs,
            )
            return  # we return since we just ran init above and don't need to run again

        # this allows us to use a single constructor for all our subclasses
        if engine_class is None:
            engine_map = {
                OpenAICompletion: OpenAICompletionEngine,
                OpenAIInstruct: OpenAIInstructEngine,
                OpenAIChat: OpenAIChatEngine,
            }
            for k in engine_map:
                if issubclass(self.__class__, k):
                    engine_class = engine_map[k]
                    break

        super().__init__(
            engine_class(
                tokenizer=tokenizer,
                api_key=api_key,
                max_streaming_tokens=max_streaming_tokens,
                timeout=timeout,
                compute_log_probs=compute_log_probs,
                model=model,
                **kwargs,
            ),
            echo=echo,
        )


class OpenAICompletion(OpenAI):
    pass


class OpenAICompletionEngine(OpenAIEngine):
    def _generator(self, prompt, temperature):

        self._reset_shared_data(prompt, temperature)  # update our shared data state

        try:
            prompt_string = prompt.decode("utf8")
            generator = self.client.completions.create(
                model=self.model_name,
                prompt=prompt_string,
                max_tokens=self.max_streaming_tokens,
                n=1,
                top_p=1.0,  # TODO: this should be controllable like temp (from the grammar)
                temperature=temperature,
                stream=True,
            )
        except Exception as e:  # TODO: add retry logic
            raise e

        for part in generator:
            if len(part.choices) > 0:
                chunk = part.choices[0].text or ""
            else:
                chunk = ""
            self.metrics.engine_input_tokens += len(self.tokenizer(prompt_string))
            self.metrics.engine_output_tokens += len(self.tokenizer(chunk))
            yield chunk.encode("utf8")


class OpenAIInstruct(OpenAI, Instruct):
    def get_role_start(self, name):
        return ""

    def get_role_end(self, name):
        if name == "instruction":
            return "<|endofprompt|>"
        else:
            raise Exception(
                f"The OpenAIInstruct model does not know about the {name} role type!"
            )


class OpenAIInstructEngine(OpenAIEngine):
    def _generator(self, prompt, temperature):
        # start the new stream
        eop_count = prompt.count(b"<|endofprompt|>")
        if eop_count > 1:
            raise Exception(
                "This model has been given multiple instruct blocks or <|endofprompt|> tokens, but this is not allowed!"
            )
        updated_prompt = prompt + b"<|endofprompt|>" if eop_count == 0 else prompt

        self._reset_shared_data(updated_prompt, temperature)

        try:
            generator = self.client.completions.create(
                model=self.model_name,
                prompt=self._shared_state["data"].decode("utf8"),
                max_tokens=self.max_streaming_tokens,
                n=1,
                top_p=1.0,  # TODO: this should be controllable like temp (from the grammar)
                temperature=temperature,
                stream=True,
            )
        except Exception as e:  # TODO: add retry logic
            raise e

        for part in generator:
            if len(part.choices) > 0:
                chunk = part.choices[0].text or ""
            else:
                chunk = ""

            yield chunk.encode("utf8")


class OpenAIChat(OpenAI, Chat):
    pass


class OpenAIChatEngine(OpenAIEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        path = os.path.join(platformdirs.user_cache_dir("guidance"), "openai.tokens")
        self.cache = dc.Cache(path)

    def _hash_prompt(self, prompt):
        return hashlib.sha256(f"{prompt}".encode()).hexdigest()

    def _generator(self, prompt, temperature):

        # find the role tags
        pos = 0
        role_end = b"<|im_end|>"
        messages = []
        found = True
        input_token_count = 0
        while found:

            # find the role text blocks
            found = False
            for role_name, start_bytes in (
                ("system", b"<|im_start|>system\n"),
                ("user", b"<|im_start|>user\n"),
                ("assistant", b"<|im_start|>assistant\n"),
            ):
                if prompt[pos:].startswith(start_bytes):
                    pos += len(start_bytes)
                    end_pos = prompt[pos:].find(role_end)
                    if end_pos < 0:
                        assert (
                            role_name == "assistant"
                        ), "Bad chat format! Last role before gen needs to be assistant!"
                        break
                    btext = prompt[pos : pos + end_pos]
                    pos += end_pos + len(role_end)
                    message_content = btext.decode("utf8")
                    input_token_count += len(self.tokenizer(message_content))
                    messages.append({"role": role_name, "content": message_content})
                    found = True
                    break

        # Add nice exception if no role tags were used in the prompt.
        # TODO: Move this somewhere more general for all chat models?
        if messages == []:
            raise ValueError(
                f"The OpenAI model {self.model_name} is a Chat-based model and requires role tags in the prompt! \
            Make sure you are using guidance context managers like `with system():`, `with user():` and `with assistant():` \
            to appropriately format your guidance program for this type of model."
            )

        # Update shared data state
        self._reset_shared_data(prompt[:pos], temperature)

        # Use cache only when temperature is 0
        if temperature == 0:
            cache_key = self._hash_prompt(prompt)

            # Check if the result is already in the cache
            if cache_key in self.cache:
                for chunk in self.cache[cache_key]:
                    yield chunk
                return

        # API call and response handling
        try:
            # Ideally, for the metrics we would use those returned by the
            # OpenAI API. Unfortunately, it appears that AzureAI hosted
            # models do not support returning metrics when streaming yet
            generator = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_streaming_tokens,
                n=1,
                top_p=1.0,  # TODO: this should be controllable like temp (from the grammar)
                temperature=temperature,
                stream=True,
            )
            self.metrics.engine_input_tokens += input_token_count

            if temperature == 0:
                cached_results = []

            for part in generator:
                if len(part.choices) > 0:
                    chunk = part.choices[0].delta.content or ""
                else:
                    chunk = ""
                encoded_chunk = chunk.encode("utf8")
                self.metrics.engine_output_tokens += len(self.tokenizer(chunk))
                yield encoded_chunk

                if temperature == 0:
                    cached_results.append(encoded_chunk)

            # Cache the results after the generator is exhausted
            if temperature == 0:
                self.cache[cache_key] = cached_results

        except Exception as e:  # TODO: add retry logic
            raise e
