from ._cache import Cache
from ._diskcache import DiskCache

cache_creator = DiskCache.default_cache_creator
