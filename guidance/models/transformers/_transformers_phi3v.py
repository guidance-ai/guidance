import logging
import json
import re
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import llguidance
from transformers import AutoModelForCausalLM, AutoProcessor

from guidance._parser import TokenParser, process_grammar, process_prompt
from guidance._schema import EngineCallResponse, GuidanceEngineMetrics
from guidance.models._model import (
    ConstraintException,
    Engine,
    Modality,
    Model,
    PromptPart,
)
# from guidance.models.transformers._transformers import TransformersTokenizer
from guidance.chat import ChatMLTemplate
from guidance.models.transformers._transformers import TransformersTokenizer

logger = logging.getLogger(__name__)


class TransformersPhi3VisionEngine(Engine):
    def __init__(
        self,
        model="microsoft/Phi-3-vision-128k-instruct",
        compute_log_probs=False,
        **kwargs,
    ):
        self.model_name = model
        # Initialize the underlying Phi 3 Vision model
        self.model_obj = AutoModelForCausalLM.from_pretrained(model, **kwargs)
        self.device = self.model_obj.device

        # Processor handles tokenization and image processing
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        super().__init__(self.processor.tokenizer, compute_log_probs)
        self.tokenizer = TransformersTokenizer(model, self.processor.tokenizer)

        # Cache for past key values
        self._past_key_values = None
        self._cached_token_ids: list[int] = []

        # Track last image token position for cache invalidation
        self._last_image_token_position = -1


    def start(self, prompt, grammar, media: dict, ensure_bos_token=True) -> TokenParser:
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")
        elif isinstance(prompt, str):
            prompt = prompt
        elif isinstance(prompt, TokenParser):
            raise NotImplementedError(
                "Still need to implement support for extending a full Parser state."
            )
        else:
            raise Exception("The passed prompt is of an unknown type!")

        # Map Guidance placeholders to Phi 3 Vision format
        # and make list of images for processing
        image_counter = 1
        images = []
        processed_prompt = prompt
        for image_id in re.findall(r"<\|image:([^\|]+)\|>", prompt):
            processed_prompt = processed_prompt.replace(
                f"<|image:{image_id}|>", f"<|image_{image_counter}|>"
            )
            images.append(media[image_id])
            image_counter += 1
        logger.debug("Transformed prompt: %s -> ", prompt, processed_prompt)

        model_inputs = self.processor(
            text=processed_prompt,
            images=images,
            return_tensors="pt",
        ).to(self.device)
        tokens = model_inputs["input_ids"][0].tolist()
        # HACK - Filter out negative image placeholder tokens, replacing with token ID 6
        # ID 6 is a no-op?
        tokens = [t if t >= 0 else 6 for t in tokens]

        serialized_grammar = process_grammar(grammar)
        ll_tokenizer = llguidance.LLTokenizer(
            llguidance.TokenizerWrapper(self.tokenizer)
        )
        ll_interpreter = llguidance.LLInterpreter(
            ll_tokenizer,
            serialized_grammar,
            log_level=int(os.environ.get("LLGUIDANCE_LOG_LEVEL", "1")),
        )
        if ensure_bos_token and self.tokenizer.bos_token_id is not None:
            bos_token_id = self.tokenizer.bos_token_id
        else:
            bos_token_id = None
        prompt_tokens = process_prompt(tokens, ll_interpreter, bos_token_id)
        return TokenParser(ll_interpreter, prompt_tokens)


    def get_next_token(
        self,
        tokens: list[int],
        mask: Optional[bytes],
        temperature: float,
        tokenization_output,
    ) -> Tuple[int, Optional[float]]:
        """Get the next token from the model."""
        logger.debug(
            f"Start TransformersPhi3Engine.get_next_token({tokens=}, {mask=}, {temperature=})"
        )

        # Invalidate cache if a new image token is encountered
        current_image_token_position = self._find_last_image_token_position(tokens)
        if current_image_token_position != self._last_image_token_position:
            self._past_key_values = None
            self._cached_token_ids = []
            self._last_image_token_position = current_image_token_position

        # Filter out negative image tokens (might need adjustment based on llguidance's handling)
        filtered_tokens = [t for t in tokens if t >= 0]

        # Get logits and log probabilities from the Phi 3 Vision model
        logits, logprobs = self.get_logits(filtered_tokens, tokenization_output)

        # Apply temperature and mask for sampling
        token_id = self.sample_with_temperature(logits, mask, temperature)

        # If log probabilities are requested, retrieve the log prob of the sampled token
        token_logprob = logprobs[token_id] if self.compute_log_probs else None

        return token_id, token_logprob

    def get_logits(self, token_ids, media):
        """Computes the logits for the given token state.

        This overrides a method from the LocalEngine class that is used to get
        inference results from the model.
        """

        # make sure we don't run off the end of the model
        if len(token_ids) >= getattr(self.model_obj.config, "max_position_embeddings", 1e10):
            raise Exception(
                f"Attempted to run a transformers model past its maximum context window size of {self.model_obj.config.max_position_embeddings}!"
            )

        # get the number of cache positions we are using
        cache_token_ids = self._cached_token_ids
        num_cached = 0
        for id in cache_token_ids:
            if (
                num_cached >= len(cache_token_ids)
                or num_cached >= len(token_ids)
                or token_ids[num_cached] != id
            ):
                break
            num_cached += 1

        # reset the cache length according to that number of positions
        past_key_values = self._past_key_values
        past_length = past_key_values[0][0].size(-2) if past_key_values is not None else 0
        if past_length > num_cached:
            # note we recompute the last token because we don't bother to handle the special case of just computing logits
            past_length = max(0, num_cached - 1)
            self._past_key_values = tuple(
                tuple(p[..., :past_length, :] for p in v) for v in past_key_values
            )
        cache_token_ids[past_length:] = []

        # call the model
        new_token_ids = token_ids[past_length:]
        if len(new_token_ids) > 0:
            with torch.no_grad():
                # Not all models support batched tokens for some reason
                try:
                    model_out = self.model_obj(
                        input_ids=torch.tensor(new_token_ids).unsqueeze(0).to(self.device),
                        past_key_values=self._past_key_values,
                        use_cache=True,
                        position_ids=torch.arange(past_length, past_length + len(new_token_ids))
                        .unsqueeze(0)
                        .to(self.device),
                        attention_mask=torch.ones(1, past_length + len(new_token_ids)).to(
                            self.device
                        ),
                        return_dict=True,
                        output_attentions=False,
                        output_hidden_states=False,
                    )
                except AssertionError:
                    for i, new_token_id in enumerate(new_token_ids):
                        input_ids = torch.tensor([new_token_id]).unsqueeze(0).to(self.device)

                        model_out = self.model_obj(
                            input_ids=input_ids,
                            past_key_values=self._past_key_values,
                            use_cache=True,
                            position_ids=torch.arange(past_length, past_length + 1)
                            .unsqueeze(0)
                            .to(self.device),
                            attention_mask=torch.ones(1, past_length + 1).to(self.device),
                            return_dict=True,
                            output_attentions=False,
                            output_hidden_states=False,
                        )

                        self._past_key_values = model_out.past_key_values
                        past_length += 1

            # save the results
            self._past_key_values = model_out.past_key_values
            cache_token_ids.extend(new_token_ids)
            # Need to add special truncating logic here for weird models that have a different output size than tokenizer vocab
            self._cached_logits = (
                model_out.logits[0, -1, : len(self.tokenizer.tokens)].cpu().numpy()
            )
            self.metrics.engine_input_tokens += len(new_token_ids)
            self.metrics.engine_output_tokens += 1

        return self._cached_logits

    def _find_last_image_token_position(self, tokens: list[int]) -> int:
        """Find the position of the last negative token (image placeholder)."""
        for i, token in enumerate(reversed(tokens)):
            if token < 0:
                return len(tokens) - i - 1
        return -1


class TransformersPhi3VisionModel(Model):
    def __init__(
        self,
        model=None,
        echo=True,
        compute_log_probs=False,
        **kwargs,
    ):
        """Build a new TransformersPhi3Model object."""
        if model is None or len(model) == 0:
            model = "microsoft/Phi-3-vision-128k-instruct"
        super().__init__(
            TransformersPhi3VisionEngine(
                model,
                compute_log_probs,
                **kwargs,
            ),
            echo=echo,
        )