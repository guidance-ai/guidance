from typing import Iterator
from pydantic import TypeAdapter

from ..._ast import GrammarNode, RuleNode, RegexNode, JsonNode, LarkNode
from ...trace import OutputAttr, TextOutput
from .._base import Model
from .._openai_base import (
    Message,
    BaseOpenAIInterpreter,
)


class LiteLLMInterpreter(BaseOpenAIInterpreter):
    SUPPORTED_ENDPOINT_TYPES = ["openai", "azure", "azure_ai", "gemini", "hosted_vllm"]

    def __init__(self, model_description: dict, **kwargs):
        try:
            import litellm
        except ImportError:
            raise Exception(
                "Please install the litellm package version >= 1.71.0 using `pip install litellm -U` in order to use guidance.models.LiteLLM!"
            )

        self.ep_type = self._check_model(model_description)
        # set default model to the first one in the list
        self.model = model_description["model_name"]
        self.router = litellm.Router(model_list=[model_description])

        super().__init__(model=self.model, client=None, **kwargs)

    def _check_model(self, model_desc: dict) -> str:
        """Check if the model description is valid."""        
        for ep_type in self.SUPPORTED_ENDPOINT_TYPES:
            if model_desc["litellm_params"]["model"].startswith(ep_type):
                return ep_type

        raise Exception("Cannot parse endpoint type. " 
                        "Please use this 'model' format in 'litellm_params': endpoint_type/model_name (e.g., openai/gpt-3.5-turbo). "
                        "Supported endpoints are: " + ", ".join(self.SUPPORTED_ENDPOINT_TYPES))


    def _run(self, **kwargs) -> Iterator[OutputAttr]:
        if self.state.active_role is None:
            # Should never happen?
            raise ValueError(
                "LiteLLM models require chat blocks (e.g. use `with assistant(): ...`)"
            )
        if self.state.active_role != "assistant":
            raise ValueError(
                "LiteLLM models can only generate as the assistant (i.e. inside of `with assistant(): ...`)"
            )
        if self.state.content:
            raise ValueError(
                f"LiteLLM models do not support pre-filled assistant messages: got data {self.state.content}."
            )
        
        chunks = self.router.completion(
            model=self.model, # pass the model name
            messages=TypeAdapter(list[Message]).dump_python(self.state.messages),  # type: ignore[arg-type]
            stream=True,
            **kwargs,
            )
        
        yield from self._handle_stream(chunks)

    def rule(self, node: RuleNode, **kwargs) -> Iterator[OutputAttr]:
        # if node.stop:
        #     raise ValueError("Stop condition not yet supported for OpenAI")
        # if node.suffix:
        #     raise ValueError("Suffix not yet supported for OpenAI")
        # if node.stop_capture:
        #     raise ValueError("Save stop text not yet supported for OpenAI")

        kwargs = kwargs.copy()
        if node.temperature:
            kwargs["temperature"] = node.temperature
        if node.max_tokens:
            kwargs["max_tokens"] = node.max_tokens

        chunks = self.run(node.value, **kwargs)
        if node.capture:
            buffered_text = ""
            for chunk in chunks:
                # TODO: this isinstance check is pretty darn fragile.
                # ~there must be a better way~
                if isinstance(chunk, TextOutput):
                    buffered_text += chunk.value
                yield chunk
            yield self.state.apply_capture(
                name=node.capture,
                value=buffered_text,
                log_prob=1,  # TODO
                is_append=node.list_append,
            )
        else:
            yield from chunks

    def regex(self, node: RegexNode, **kwargs) -> Iterator[OutputAttr]:
        if node.regex is not None and self.ep_type not in ["hosted_vllm"]:
            raise ValueError(f"Regex is not yet supported for ep {self.ep_type}")
        
        if self.ep_type == "hosted_vllm":
            return self._regex_vllm(node, **kwargs)

        # We're in unconstrained mode now.
        return self._run(**kwargs)
    
    def _regex_vllm(self, node: RegexNode, **kwargs):
        """Run the regex node using vLLM."""
        buffer: str = ""
        for attr in self._run(
            extra_body=dict(
                guided_decoding_backend="guidance",
                guided_regex=node.regex
            )
        ):
            if isinstance(attr, TextOutput):
                buffer += attr.value
            yield attr
    
    def json(self, node: JsonNode, **kwargs) -> Iterator[OutputAttr]:
        if self.ep_type in ["openai"] and \
            (self.model in ["gpt-3.5-turbo"] or self.model.startswith("gpt-4-")):
            raise ValueError(f"Json is not supported for ep {self.ep_type}/{self.model}")

        schema = {k: v for k,v in node.schema.items() if k != "x-guidance"}
        schema["additionalProperties"] = False  # Ensure no additional properties are allowed
        return self._run(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "json_schema",  # TODO?
                    "schema": schema,
                    "strict": True,                    
                }
            },
            **kwargs,
        )
    
    def grammar(self, node: GrammarNode, **kwargs) -> Iterator[OutputAttr]:
        if self.ep_type not in ["hosted_vllm"]:
            raise ValueError(f"Grammar is not yet supported for ep {self.ep_type}")        

        if self.ep_type == "hosted_vllm":
            return self._grammar_vllm(node, **kwargs)

        return self._run(**kwargs)
    
    def _grammar_vllm(self, node: GrammarNode, **kwargs) -> Iterator[OutputAttr]:
        """Run the grammar node using vLLM."""
        buffer: str = ""
        for attr in self._run(
            extra_body=dict(
                guided_decoding_backend="guidance",
                guided_grammar=node.ll_grammar()
            )
        ):
            if isinstance(attr, TextOutput):
                buffer += attr.value
            yield attr
        matches = node.match(
            buffer,
            raise_exceptions=False,
            # Turn of max_tokens since we don't have access to the tokenizer
            enforce_max_tokens=False,
        )
        if matches is None:
            # TODO: should probably raise...
            # raise ValueError("vLLM failed to constrain the grammar")
            pass
        else:
            for name, value in matches.captures.items():
                log_probs = matches.log_probs[name]
                if isinstance(value, list):
                    assert isinstance(log_probs, list)
                    assert len(value) == len(log_probs)
                    for v, l in zip(value, log_probs):
                        yield self.state.apply_capture(
                            name=name, value=v, log_prob=l, is_append=True
                        )
                else:
                    yield self.state.apply_capture(
                        name=name, value=value, log_prob=log_probs, is_append=False
                    )

    def lark(self, node: LarkNode, **kwargs):
        if self.ep_type not in ["hosted_vllm"]:
            raise ValueError(f"LarkGrammar is not yet supported for ep {self.ep_type}")
        
        if self.ep_type == "hosted_vllm":
            return self._lark_vllm(node, **kwargs)
        
        return self._run(**kwargs)

    def _lark_vllm(self, node: LarkNode, **kwargs):
        """Run the lark grammar node using vLLM."""
        buffer: str = ""
        for attr in self._run(
            extra_body=dict(
                guided_decoding_backend="guidance",
                guided_grammar=node.lark_grammar
            )
        ):
            if isinstance(attr, TextOutput):
                buffer += attr.value
            yield attr

class LiteLLM(Model):
    def __init__(self, model_description: dict, echo=True, **kwargs):
        interpreter = LiteLLMInterpreter(model_description=model_description, **kwargs)
        super().__init__(
            interpreter=interpreter,
            echo=echo,
        )
