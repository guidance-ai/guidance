import os
from typing import Dict, Any

import platformdirs

from guidance.llms.caches import Cache
try:
    from gptcache.adapter.api import get, put, init_similar_cache
except ImportError:
    pass


class GPTCache(Cache):
    """GPTCache is a semantic cache that uses GPTCache lib."""

    def __init__(self, cache):
        """Build or wrap a gptcache object."""

        if isinstance(cache, str):
            cache_obj = Cache()
            init_similar_cache(
                data_dir=os.path.join(
                    platformdirs.user_cache_dir("guidance"), f"_{cache}.gptcache"
                ),
                cache_obj=cache_obj,
            )
        else:
            cache_obj = cache

        self._cache_obj = cache_obj

    def __getitem__(self, key: str) -> str:
        return get(key)

    def __setitem__(self, key: str, value: str) -> None:
        put(key, value)

    def __contains__(self, key: str) -> bool:
        return get(key) is not None

    def create_key(self, llm: str, **kwargs: Dict[str, Any]) -> str:
        if "cache_key" in kwargs:
            return str(kwargs["cache_key"])
        else:
            return str(kwargs["prompt"])
