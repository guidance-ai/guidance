import os

import diskcache
import platformdirs

from guidance.llms.caches import Cache

def user_cache_dir(appname):
    return "\\tmp"

class DiskCache(Cache):
    """DiskCache is a cache that uses diskcache lib."""
    def __init__(self, llm_name: str):
        print("Inside my guidance")
        self._diskcache = diskcache.Cache(
            os.path.join(
                user_cache_dir("guidance"), f"{llm_name}.diskcache"
            )
        )

    def __getitem__(self, key: str) -> str:
        return self._diskcache[key]

    def __setitem__(self, key: str, value: str) -> None:
        self._diskcache[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._diskcache
    
    def clear(self):
        self._diskcache.clear()
