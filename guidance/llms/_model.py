import os
import logging

from text_generation import AsyncClient
from transformers.configuration_utils import PretrainedConfig

endpoint_url = os.getenv("TGI_ENDPOINT_URL", "http://127.0.0.1:8080/")

logger = logging.getLogger("__guidance__display__")


# TODO: tgi model
class TGIModel:
    # TODO: tgi only can use the api to access.
    # install and import the text generate client

    def __init__(self, config: PretrainedConfig) -> None:
        self.config = config
        self.client = AsyncClient(endpoint_url)

    async def generate(self, prompt: str, **kwargs):
        logger.debug(f"input prompt :{prompt},kwargs:{kwargs}")
        text = await self.client.generate(prompt, **kwargs)

        res_dict: dict = text.model_dump()    
        res_text =  res_dict["generated_text"]

        logger.debug(f"res_text: {res_text}")
        for stop_word in kwargs.get("stop_sequences", []):
            if stop_word in res_text:
                res_dict["generated_text"] = res_text[: res_text.index(stop_word)]

        return res_dict


# TODO: vllm model


class Vllm:
    pass
