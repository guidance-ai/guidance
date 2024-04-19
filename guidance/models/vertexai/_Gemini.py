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

from ._vertexai import (
    VertexAICompletion,
    VertexAIInstruct,
    VertexAIChat,
    VertexAIChatEngine,
)

_image_token_pattern = re.compile(r"<\|_image:(.*)\|>")

try:
    from vertexai.language_models import (
        TextGenerationModel,
        ChatModel,
        InputOutputTextPair,
    )
    from vertexai.preview.generative_models import GenerativeModel, Content, Part, Image
    import vertexai

    # def get_chat_response(message):
    #     vertexai.init(project="PROJECT_ID", location="us-central1")
    #     model = GenerativeModel("gemini-pro")
    #     chat = model.start_chat()
    #     response = chat.send_message(message)
    #     return response.text

    # print(get_chat_response("Hello"))
    # print(get_chat_response("What are all the colors in a rainbow?"))
    # print(get_chat_response("Why does it appear when it rains?"))
    is_vertexai = True
except ModuleNotFoundError:
    is_vertexai = False

# class GeminiCompletion(VertexAICompletion):
#     def __init__(self, model, tokenizer=None, echo=True, caching=True, temperature=0.0, max_streaming_tokens=None, **kwargs):

#         if isinstance(model, str):
#             self.model_name = model
#             self.model_obj = TextGenerationModel.from_pretrained(self.model_name)

#         # Gemini does not have a public tokenizer, so we pretend it tokenizes like gpt2...
#         if tokenizer is None:
#             tokenizer = tiktoken.get_encoding("gpt2")

#         # the superclass does all the work
#         super().__init__(
#             model,
#             tokenizer=tokenizer,
#             echo=echo,
#             caching=caching,
#             temperature=temperature,
#             max_streaming_tokens=max_streaming_tokens,
#             **kwargs
#         )

# class GeminiInstruct(VertexAIInstruct):
#     def __init__(self, model, tokenizer=None, echo=True, caching=True, temperature=0.0, max_streaming_tokens=None, **kwargs):

#         if isinstance(model, str):
#             self.model_name = model
#             self.model_obj = TextGenerationModel.from_pretrained(self.model_name)

#         # Gemini does not have a public tokenizer, so we pretend it tokenizes like gpt2...
#         if tokenizer is None:
#             tokenizer = tiktoken.get_encoding("gpt2")

#         # the superclass does all the work
#         super().__init__(
#             model,
#             tokenizer=tokenizer,
#             echo=echo,
#             caching=caching,
#             temperature=temperature,
#             max_streaming_tokens=max_streaming_tokens,
#             **kwargs
#         )


class GeminiChat(VertexAIChat):
    def __init__(
        self, model, tokenizer=None, echo=True, max_streaming_tokens=None, **kwargs
    ):
        if isinstance(model, str):
            model = GenerativeModel(model)

        # Gemini does not have a public tokenizer, so we pretend it tokenizes like gpt2...
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("gpt2")

        # the superclass does all the work
        super().__init__(
            model,
            tokenizer=tokenizer,
            echo=echo,
            max_streaming_tokens=max_streaming_tokens,
            engine_class=GeminiChatEngine,
            **kwargs,
        )


class GeminiChatEngine(VertexAIChatEngine):
    def _start_chat(self, system_text, messages):
        assert (
            system_text == ""
        ), "We don't support passing system text to Gemini models (yet?)!"
        out = self.model_obj.start_chat(history=messages)
        return out

    def _start_generator(self, system_text, messages, temperature):
        # last_user_text = messages[-1]["content"]
        formated_messages = []
        for m in messages:
            raw_parts = _image_token_pattern.split(m["content"])
            parts = []
            for i in range(0, len(raw_parts), 2):

                # append the text portion
                if len(raw_parts[i]) > 0:
                    parts.append(Part.from_text(raw_parts[i]))

                # append any image
                if i + 1 < len(raw_parts):
                    parts.append(
                        Part.from_image(Image.from_bytes(self[raw_parts[i + 1]]))
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
