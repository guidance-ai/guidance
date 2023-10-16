from guidance.llms.caches import Cache

class MemCache(Cache):
    """MemoryCache is a cache that uses in-memory dictionary."""
    def __init__(self, *args, **kwargs):
        self._cache = dict()

    def __getitem__(self, key: str) -> str:
        return self._cache[key]

    def __setitem__(self, key: str, value: str) -> None:
        self._cache[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def clear(self) -> None:
        self._cache.clear()
