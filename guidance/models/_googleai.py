import re
from ._model import Chat, Instruct
from ._grammarless import Grammarless, GrammarlessEngine
import tiktoken
import os

_image_token_pattern = re.compile(r"<\|_image:(.*)\|>")


class GoogleAIEngine(GrammarlessEngine):
    def __init__(
        self,
        model,
        tokenizer,
        api_key,
        max_streaming_tokens,
        timeout,
        compute_log_probs,
        **kwargs,
    ):
        try:
            import google.generativeai as genai
        except ModuleNotFoundError:
            raise Exception(
                "Please install the Google AI Studio(makersuite.google.com) package using `pip install google-generativeai google-ai-generativelanguage` in order to use guidance.models.GoogleAI!"
            )

        assert (
            not compute_log_probs
        ), "We don't support compute_log_probs=True yet for GoogleAIEngine!"

        if api_key is None:
            api_key = os.environ.get("GOOGLEAI_API_KEY")

        genai.configure(api_key=api_key)

        # Gemini does not have a public tokenizer, so we pretend it tokenizes like gpt2...
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("gpt2")
        self.model_name = model

        self.model_obj = genai.GenerativeModel(self.model_name, **kwargs)

        super().__init__(tokenizer, max_streaming_tokens, timeout, compute_log_probs)


class GoogleAI(Grammarless):
    def __init__(
        self,
        model,
        tokenizer=None,
        echo=True,
        api_key=None,
        max_streaming_tokens=None,
        timeout=0.5,
        compute_log_probs=False,
        **kwargs,
    ):
        """Build a new GoogleAI model object that represents a model in a given state."""

        # if we are called directly (as opposed to through super()) then we convert ourselves to a more specific subclass if possible
        if self.__class__ is GoogleAI:
            found_subclass = None

            # chat
            found_subclass = GoogleAIChat  # we assume all models are chat right now

            # instruct
            # elif "instruct" in model:
            #     found_subclass = GoogleAIInstruct

            # # regular completion
            # else:
            #     found_subclass = GoogleAICompletion

            # convert to any found subclass
            self.__class__ = found_subclass
            found_subclass.__init__(
                self,
                model,
                tokenizer=None,
                echo=True,
                api_key=api_key,
                max_streaming_tokens=max_streaming_tokens,
                timeout=timeout,
                compute_log_probs=False,
                **kwargs,
            )
            return  # we return since we just ran init above and don't need to run again

        # this allows us to use a single constructor for all our subclasses
        engine_map = {GoogleAIChat: GoogleAIChatEngine}

        super().__init__(
            engine=engine_map[self.__class__](
                model=model,
                tokenizer=tokenizer,
                api_key=api_key,
                max_streaming_tokens=max_streaming_tokens,
                timeout=timeout,
                compute_log_probs=compute_log_probs,
                **kwargs,
            ),
            echo=echo,
        )


class GoogleAIChatEngine(GoogleAIEngine):
    def _generator(self, prompt, temperature):

        # find the system text
        pos = 0
        system_start = b"<|im_start|>system\n"
        user_start = b"<|im_start|>user\n"
        assistant_start = b"<|im_start|>assistant\n"
        role_end = b"<|im_end|>"
        # system_start_pos = prompt.startswith(system_start)

        # find the system text
        system_text = b""
        if prompt.startswith(system_start):
            pos += len(system_start)
            system_end_pos = prompt.find(role_end)
            system_text = prompt[pos:system_end_pos]
            pos = system_end_pos + len(role_end)

        # find the user/assistant pairs
        messages = []
        valid_end = False
        while True:

            # find the user text
            if prompt[pos:].startswith(user_start):
                pos += len(user_start)
                end_pos = prompt[pos:].find(role_end)
                if end_pos < 0:
                    break
                messages.append(
                    dict(
                        role="user",
                        content=prompt[pos : pos + end_pos].decode("utf8"),
                    )
                )
                pos += end_pos + len(role_end)
            elif prompt[pos:].startswith(assistant_start):
                pos += len(assistant_start)
                end_pos = prompt[pos:].find(role_end)
                if end_pos < 0:
                    valid_end = True
                    break
                messages.append(
                    dict(
                        role="model",
                        content=prompt[pos : pos + end_pos].decode("utf8"),
                    )
                )
                pos += end_pos + len(role_end)
            else:
                raise Exception(
                    "It looks like your prompt is not a well formed chat prompt! Please enclose all model state appends inside chat role blocks like `user()` or `assistant()`."
                )

        self._data = prompt[:pos]

        assert len(messages) > 0, "Bad chat format! No chat blocks were defined."
        assert (
            messages[-1]["role"] == "user"
        ), "Bad chat format! There must be a user() role before the last assistant() role."
        assert valid_end, "Bad chat format! You must generate inside assistant() roles."

        # TODO: don't make a new session on every call
        # last_user_text = messages.pop().content

        return self._start_generator(system_text.decode("utf8"), messages, temperature)

        # kwargs = {}
        # if self.max_streaming_tokens is not None:
        #     kwargs["max_output_tokens"] = self.max_streaming_tokens
        # generator = chat_session.send_message_streaming(last_user_text, temperature=temperature, **kwargs)

        # for chunk in generator:
        #     yield chunk.text.encode("utf8")

    def _start_chat(self, system_text, messages):
        assert (
            system_text == ""
        ), "We don't support passing system text to Gemini models (yet?)!"
        out = self.model_obj.start_chat(history=messages)
        return out

    def _start_generator(self, system_text, messages, temperature):
        from google.ai.generativelanguage import Content, Part, Blob

        # last_user_text = messages[-1]["content"]
        formated_messages = []
        for m in messages:
            raw_parts = _image_token_pattern.split(m["content"])
            parts = []
            for i in range(0, len(raw_parts), 2):

                # append the text portion
                if len(raw_parts[i]) > 0:
                    parts.append(Part(text=raw_parts[i]))

                # append any image
                if i + 1 < len(raw_parts):
                    # parts.append(Part.from_image(Image.from_bytes(self[raw_parts[i+1]])))
                    parts.append(
                        Part(
                            inline_data=Blob(
                                mime_type="image/jpeg", data=self[raw_parts[i + 1]]
                            )
                        )
                    )
            formated_messages.append(Content(role=m["role"], parts=parts))
        last_user_parts = (
            formated_messages.pop()
        )  # remove the last user stuff that goes in send_message (and not history)

        chat_session = self.model_obj.start_chat(
            history=formated_messages,
        )

        generation_config = {"temperature": temperature}
        if self.max_streaming_tokens is not None:
            generation_config["max_output_tokens"] = self.max_streaming_tokens
        generator = chat_session.send_message(
            last_user_parts, generation_config=generation_config, stream=True
        )

        for chunk in generator:
            yield chunk.candidates[0].content.parts[0].text.encode("utf8")


class GoogleAIChat(GoogleAI, Chat):
    pass
