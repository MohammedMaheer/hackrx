import functools
import time
import hashlib
import pickle
import logging
from typing import Callable, Any, Tuple, Dict, Optional

logger = logging.getLogger(__name__)

# Simple in-memory cache for hackathon leaderboard (reset on restart)
_CACHE: Dict[str, Tuple[float, Any]] = {}
_ASYNC_CACHE: Dict[str, Any] = {}
CACHE_TTL = 600  # 10 minutes

def _make_hashable(obj):
    """Recursively convert lists/dicts/sets to tuples for hashing"""
    if isinstance(obj, (tuple, list)):
        return tuple(_make_hashable(e) for e in obj)
    elif isinstance(obj, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, set):
        return tuple(sorted(_make_hashable(e) for e in obj))
    return obj

def generate_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Generate a stable cache key from function name and arguments, handling unhashable types"""
    try:
        hashable_args = _make_hashable(args)
        hashable_kwargs = _make_hashable(kwargs)
        args_str = str(hashable_args)
        kwargs_str = str(hashable_kwargs)
        combined = f"{func_name}:{args_str}:{kwargs_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    except Exception:
        # Fallback to simple string conversion (may still error if unhashable)
        return f"{func_name}:{hash((str(args), str(kwargs)))}"


def cache_result(ttl: int = CACHE_TTL):
    """Decorator for caching synchronous function results"""
    def decorator(fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = generate_cache_key(fn.__name__, args, kwargs)
            
            # Check cache
            now = time.time()
            if cache_key in _CACHE:
                timestamp, value = _CACHE[cache_key]
                if now - timestamp < ttl:
                    logger.debug(f"Cache hit for {fn.__name__}")
                    return value
            
            # Execute function and cache result
            try:
                result = fn(*args, **kwargs)
                _CACHE[cache_key] = (now, result)
                logger.debug(f"Cached result for {fn.__name__}")
                return result
            except Exception as e:
                logger.error(f"Function {fn.__name__} failed: {e}")
                raise
                
        return wrapper
    return decorator

def acache_result(fn: Callable) -> Callable:
    """Decorator for caching async function results"""
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        # Generate cache key
        cache_key = generate_cache_key(fn.__name__, args, kwargs)
        
        # Check cache
        if cache_key in _ASYNC_CACHE:
            logger.debug(f"Async cache hit for {fn.__name__}")
            return _ASYNC_CACHE[cache_key]
        
        # Execute function and cache result
        try:
            result = await fn(*args, **kwargs)
            _ASYNC_CACHE[cache_key] = result
            logger.debug(f"Async cached result for {fn.__name__}")
            
            # Limit cache size to prevent memory issues
            if len(_ASYNC_CACHE) > 1000:
                # Remove oldest entries (simple FIFO)
                keys_to_remove = list(_ASYNC_CACHE.keys())[:100]
                for key in keys_to_remove:
                    del _ASYNC_CACHE[key]
                logger.info("Async cache cleaned up")
            
            return result
        except Exception as e:
            logger.error(f"Async function {fn.__name__} failed: {e}")
            raise
    
    return wrapper

def clear_cache():
    """Clear all cached results"""
    global _CACHE, _ASYNC_CACHE
    _CACHE.clear()
    _ASYNC_CACHE.clear()
    logger.info("All caches cleared")

def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics"""
    now = time.time()
    
    # Count valid entries in sync cache
    valid_sync = 0
    for timestamp, _ in _CACHE.values():
        if now - timestamp < CACHE_TTL:
            valid_sync += 1
    
    return {
        "sync_cache_size": len(_CACHE),
        "sync_cache_valid": valid_sync,
        "async_cache_size": len(_ASYNC_CACHE),
        "cache_ttl": CACHE_TTL
    }

class TimedCache:
    """A more sophisticated cache with automatic cleanup"""
    
    def __init__(self, ttl: int = 600, max_size: int = 1000):
        self.ttl = ttl
        self.max_size = max_size
        self.cache: Dict[str, Tuple[float, Any]] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        if key not in self.cache:
            return None
        
        timestamp, value = self.cache[key]
        if time.time() - timestamp > self.ttl:
            del self.cache[key]
            return None
        
        return value
    
    def set(self, key: str, value: Any):
        """Set value in cache with cleanup if needed"""
        now = time.time()
        
        # Clean expired entries
        expired_keys = [
            k for k, (t, _) in self.cache.items() 
            if now - t > self.ttl
        ]
        for k in expired_keys:
            del self.cache[k]
        
        # Remove oldest entries if cache is too large
        if len(self.cache) >= self.max_size:
            oldest_keys = sorted(
                self.cache.keys(), 
                key=lambda k: self.cache[k][0]
            )[:len(self.cache) - self.max_size + 1]
            
            for k in oldest_keys:
                del self.cache[k]
        
        self.cache[key] = (now, value)
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
    
    def size(self) -> int:
        """Get current cache size"""
        return len(self.cache)

# Global timed cache instance
timed_cache = TimedCache()

def cached_async(ttl: int = 600):
    """Advanced async caching decorator with TTL"""
    def decorator(fn: Callable):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            cache_key = generate_cache_key(fn.__name__, args, kwargs)
            
            # Check cache
            cached_value = timed_cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for {fn.__name__}")
                return cached_value
            
            # Execute and cache
            try:
                result = await fn(*args, **kwargs)
                timed_cache.set(cache_key, result)
                logger.debug(f"Cached result for {fn.__name__}")
                return result
            except Exception as e:
                logger.error(f"Cached function {fn.__name__} failed: {e}")
                raise
        
        return wrapper
    return decorator