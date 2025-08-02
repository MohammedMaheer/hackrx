import functools
import time
from typing import Callable, Any, Tuple, Dict

# Simple in-memory cache for hackathon leaderboard (reset on restart)
_CACHE: Dict[str, Tuple[float, Any]] = {}
CACHE_TTL = 600  # 10 minutes

def cache_result(ttl: int = CACHE_TTL):
    def decorator(fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = str((fn.__name__, args, frozenset(kwargs.items())))
            now = time.time()
            if key in _CACHE:
                t, v = _CACHE[key]
                if now - t < ttl:
                    return v
            result = fn(*args, **kwargs)
            _CACHE[key] = (now, result)
            return result
        return wrapper
    return decorator

# Async version for LLM calls
import asyncio
async def acache_result(fn):
    cache = {}
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        key = str((fn.__name__, args, frozenset(kwargs.items())))
        if key in cache:
            return cache[key]
        result = await fn(*args, **kwargs)
        cache[key] = result
        return result
    return wrapper
