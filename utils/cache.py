#!/usr/bin/env python
# coding: utf-8

"""
Caching system for database queries to improve performance.
"""

import logging
import time
import hashlib
import pickle
import functools
from typing import Any, Callable, Dict, Optional

# Global cache dictionary
_CACHE = {}
_CACHE_TIMESTAMPS = {}

# Default cache timeout (5 minutes)
DEFAULT_TIMEOUT = 300


def cached(timeout: Optional[int] = None):
    """
    Decorator to cache function results with optional timeout.
    
    Args:
        timeout: Cache timeout in seconds (None for no expiry)
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a unique key from function name and arguments
            key_parts = [func.__name__]
            for arg in args:
                key_parts.append(str(arg))
            for k, v in sorted(kwargs.items()):
                key_parts.append(f"{k}={v}")
            
            key_string = ":".join(key_parts)
            cache_key = hashlib.md5(key_string.encode()).hexdigest()
            
            # Check if result is in cache and not expired
            current_time = time.time()
            if cache_key in _CACHE:
                timestamp = _CACHE_TIMESTAMPS.get(cache_key, 0)
                cache_timeout = timeout or DEFAULT_TIMEOUT
                
                if timeout is None or current_time - timestamp < cache_timeout:
                    logging.debug(f"Cache hit for {func.__name__}")
                    return _CACHE[cache_key]
                else:
                    logging.debug(f"Cache expired for {func.__name__}")
            
            # Call function and cache result
            result = func(*args, **kwargs)
            _CACHE[cache_key] = result
            _CACHE_TIMESTAMPS[cache_key] = current_time
            logging.debug(f"Cached result for {func.__name__}")
            
            return result
        return wrapper
    return decorator


def clear_cache():
    """
    Clear the entire cache.
    """
    global _CACHE, _CACHE_TIMESTAMPS
    logging.info(f"Clearing cache - currently has {len(_CACHE)} entries")
    _CACHE.clear()
    _CACHE_TIMESTAMPS.clear()
    logging.info("Cache cleared successfully")


def clear_cache_for_function(func_name: str):
    """
    Clear cache entries for a specific function.
    
    Args:
        func_name: Function name
    """
    global _CACHE, _CACHE_TIMESTAMPS
    keys_to_delete = []
    
    for key in _CACHE.keys():
        if key.startswith(func_name):
            keys_to_delete.append(key)
    
    for key in keys_to_delete:
        if key in _CACHE:
            del _CACHE[key]
        if key in _CACHE_TIMESTAMPS:
            del _CACHE_TIMESTAMPS[key]
    
    logging.info(f"Cleared {len(keys_to_delete)} cache entries for {func_name}")


def get_cache_stats():
    """
    Get statistics about the cache.
    
    Returns:
        Dict: Cache statistics
    """
    stats = {
        "total_entries": len(_CACHE),
        "by_function": {}
    }
    
    for key in _CACHE:
        func_name = key.split(':')[0]
        if func_name not in stats["by_function"]:
            stats["by_function"][func_name] = 0
        stats["by_function"][func_name] += 1
    
    return stats