import copy
import re
from ._model import Chat
from ._grammarless import Grammarless, GrammarlessEngine

_image_token_pattern = re.compile(r"<\|_image:(.*)\|>")


class ClassUnavailableException(Exception):
    pass


class QianfanAI(Grammarless):
    def __init__(
        self,
        model=None,
        echo=True,
        max_streaming_tokens=None,
        timeout=0.5,
        compute_log_probs=False,
        **kwargs,
    ):
        """Build a new QianfanAI model object that represents a model in a given state."""

        # if we are called directly (as opposed to through super()) then we convert ourselves to a more specific subclass if possible
        if self.__class__ is QianfanAI:
            raise ClassUnavailableException("Cannot use `QianfanAI` directly, please use `QianfanAIChat` or `QianfanAICompletion` instead")

        engine_map = {
            QianfanAIChat: QianfanAIChatEngine,
            QianfanAICompletion: QianfanAICompletionEngine,
        }

        super().__init__(
            engine=engine_map[self.__class__](
                model=model,
                max_streaming_tokens=max_streaming_tokens,
                timeout=timeout,
                compute_log_probs=compute_log_probs,
                **kwargs,
            ),
            echo=echo,
        )


class QianfanAIEngine(GrammarlessEngine):

    def __init__(
        self,
        model,
        max_streaming_tokens,
        timeout,
        compute_log_probs,
        **kwargs,
    ):
        try:
            from qianfan import ChatCompletion, Completion
        except ModuleNotFoundError:
            raise Exception(
                "Please install the Baidu Qianfan package using `pip install qianfan` "
                "in order to use guidance.models.QianfanAI!"
            )

        assert (
            not compute_log_probs
        ), "We don't support compute_log_probs=True yet for QianfanAIEngine!"
        self.model_name = model

        self.model_obj = ChatCompletion(model=model, **kwargs) if self.__class__ is QianfanAIChatEngine else Completion(model=model, **kwargs)

        self.extra_arguments = copy.deepcopy(kwargs)
        self.extra_arguments.pop("endpoint") if "endpoint" in kwargs else None

        super().__init__(None, max_streaming_tokens, timeout, compute_log_probs)


class QianfanAIChat(QianfanAI, Chat):
    pass


class QianfanAIChatEngine(QianfanAIEngine):
    def _generator(self, prompt, temperature):

        # find the system text
        pos = 0

        system_start = b"<|im_start|>system\n"
        user_start = b"<|im_start|>user\n"
        assistant_start = b"<|im_start|>assistant\n"
        role_end = b"<|im_end|>"

        # find the system text
        system_text = ""
        if prompt.startswith(system_start):
            pos += len(system_start)
            system_end_pos = prompt.find(role_end)
            system_text = prompt[pos:system_end_pos].decode("utf8")
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
                        content=prompt[pos: pos + end_pos].decode("utf8"),
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
                        role="assistant",
                        content=prompt[pos: pos + end_pos].decode("utf8"),
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

        if temperature == 0.0:
            temperature = 0.0001

        input_kwargs = {"temperature": temperature}
        input_kwargs.update(self.extra_arguments)

        if system_text:
            input_kwargs["system"] = system_text

        input_kwargs["stream"] = True

        result_iter = self.model_obj.do(messages, **input_kwargs)
        for response in result_iter:
            yield response.body["result"].encode("utf8")


class QianfanAICompletion(QianfanAI):
    pass


class QianfanAICompletionEngine(QianfanAIEngine):
    def _generator(self, prompt, temperature):
        if temperature == 0.0:
            temperature = 0.0001

        input_kwargs = {"temperature": temperature}
        input_kwargs.update(self.extra_arguments)
        input_kwargs["stream"] = True

        self._data = prompt

        result_iter = self.model_obj.do(prompt.decode("utf8"), **input_kwargs)
        for response in result_iter:
            yield response.body["result"].encode("utf8")