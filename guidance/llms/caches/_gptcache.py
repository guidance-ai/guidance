import os
from typing import Dict, Any

import platformdirs

from guidance.llms.caches import Cache
import gptcache
from gptcache.adapter.api import get, put, init_similar_cache


class GPTCache(Cache):
    """GPTCache is a semantic cache that uses GPTCache lib."""

    def __init__(self, cache_obj: gptcache.Cache):
        self._cache_obj = cache_obj

    def __getitem__(self, key: str) -> str:
        return get(key)

    def __setitem__(self, key: str, value: str) -> None:
        put(key, value)

    def __contains__(self, key: str) -> bool:
        return get(key) is not None

    def default_cache_state(self) -> bool:
        return False

    def create_key(self, llm: str, **kwargs: Dict[str, Any]) -> str:
        return str(kwargs["prompt"])

    @staticmethod
    def default_cache_creator(llm_name: str) -> Cache:
        cache_obj = Cache()
        init_similar_cache(
            data_dir=os.path.join(
                platformdirs.user_cache_dir("guidance"), f"_{llm_name}.gptcache"
            ),
            cache_obj=cache_obj,
        )
        return GPTCache(cache_obj=cache_obj)
