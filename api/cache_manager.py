"""
Cost Optimization Cache Manager
Manages caching for LLM API calls to reduce costs and improve performance.
"""

import os
import logging
import hashlib
import json
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import redis
from redis.exceptions import ConnectionError, RedisError

from config import settings


# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry for storing API responses."""
    key: str
    response: str
    tokens_used: int
    cost_usd: float
    model: str
    created_at: datetime
    expires_at: datetime
    hit_count: int = 0
    last_accessed: Optional[datetime] = None


@dataclass
class CacheStats:
    """Cache statistics."""
    total_requests: int
    cache_hits: int
    cache_misses: int
    total_tokens_saved: int
    total_cost_saved: float
    hit_rate: float
    entries_count: int
    oldest_entry: Optional[datetime]
    newest_entry: Optional[datetime]


class CacheManager:
    """Manages caching for LLM API calls."""
    
    def __init__(self, redis_url: Optional[str] = None, ttl: int = 86400):
        """
        Initialize the cache manager.
        
        Args:
            redis_url: Redis connection URL
            ttl: Default time-to-live in seconds
        """
        self.redis_url = redis_url or settings.redis_url
        self.ttl = ttl
        self.redis_client = None
        self.cache_prefix = "headline_optimizer:"
        self.stats_prefix = "stats:"
        
        # Cost tracking
        self.cost_per_token = {
            'gpt-4o-mini': 0.00003,
            'gpt-4': 0.00006,
            'claude-3-haiku': 0.000015,
            'claude-3-sonnet': 0.00003
        }
        
        # Initialize Redis connection
        self._connect_redis()
    
    def _connect_redis(self):
        """Connect to Redis server."""
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            logger.info("Connected to Redis successfully")
        except (ConnectionError, RedisError) as e:
            logger.warning(f"Could not connect to Redis: {e}")
            self.redis_client = None
    
    def _is_connected(self) -> bool:
        """Check if Redis is connected."""
        if self.redis_client is None:
            return False
        
        try:
            self.redis_client.ping()
            return True
        except (ConnectionError, RedisError):
            return False
    
    def _generate_cache_key(self, prompt: str, model: str, temperature: float = 0.7) -> str:
        """
        Generate a cache key for a prompt.
        
        Args:
            prompt: The prompt text
            model: The model name
            temperature: The temperature setting
            
        Returns:
            Cache key string
        """
        # Create a hash of the prompt and parameters
        content = f"{prompt}:{model}:{temperature}"
        hash_obj = hashlib.md5(content.encode('utf-8'))
        return f"{self.cache_prefix}{hash_obj.hexdigest()}"
    
    def _serialize_entry(self, entry: CacheEntry) -> str:
        """Serialize a cache entry to JSON."""
        entry_dict = asdict(entry)
        # Convert datetime objects to ISO strings
        entry_dict['created_at'] = entry.created_at.isoformat()
        entry_dict['expires_at'] = entry.expires_at.isoformat()
        if entry.last_accessed:
            entry_dict['last_accessed'] = entry.last_accessed.isoformat()
        return json.dumps(entry_dict)
    
    def _deserialize_entry(self, data: str) -> CacheEntry:
        """Deserialize a cache entry from JSON."""
        entry_dict = json.loads(data)
        # Convert ISO strings back to datetime objects
        entry_dict['created_at'] = datetime.fromisoformat(entry_dict['created_at'])
        entry_dict['expires_at'] = datetime.fromisoformat(entry_dict['expires_at'])
        if entry_dict.get('last_accessed'):
            entry_dict['last_accessed'] = datetime.fromisoformat(entry_dict['last_accessed'])
        return CacheEntry(**entry_dict)
    
    def get(self, prompt: str, model: str, temperature: float = 0.7) -> Optional[CacheEntry]:
        """
        Get a cached response for a prompt.
        
        Args:
            prompt: The prompt text
            model: The model name
            temperature: The temperature setting
            
        Returns:
            CacheEntry if found, None otherwise
        """
        if not self._is_connected():
            return None
        
        try:
            cache_key = self._generate_cache_key(prompt, model, temperature)
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                entry = self._deserialize_entry(cached_data)
                
                # Check if entry has expired
                if datetime.now() > entry.expires_at:
                    self.redis_client.delete(cache_key)
                    return None
                
                # Update access statistics
                entry.hit_count += 1
                entry.last_accessed = datetime.now()
                
                # Update cache with new access info
                self.redis_client.setex(
                    cache_key,
                    int((entry.expires_at - datetime.now()).total_seconds()),
                    self._serialize_entry(entry)
                )
                
                # Update stats
                self._update_stats(hit=True, tokens_saved=entry.tokens_used, cost_saved=entry.cost_usd)
                
                logger.info(f"Cache hit for prompt: {prompt[:50]}...")
                return entry
            
            # Cache miss
            self._update_stats(hit=False)
            return None
            
        except (ConnectionError, RedisError) as e:
            logger.error(f"Redis error in get: {e}")
            return None
    
    def set(self, prompt: str, response: str, model: str, tokens_used: int,
            temperature: float = 0.7, ttl: Optional[int] = None) -> bool:
        """
        Cache a response for a prompt.
        
        Args:
            prompt: The prompt text
            response: The response text
            model: The model name
            tokens_used: Number of tokens used
            temperature: The temperature setting
            ttl: Time-to-live in seconds (uses default if None)
            
        Returns:
            True if cached successfully, False otherwise
        """
        if not self._is_connected():
            return False
        
        try:
            cache_key = self._generate_cache_key(prompt, model, temperature)
            ttl = ttl or self.ttl
            
            # Calculate cost
            cost_usd = tokens_used * self.cost_per_token.get(model, 0.00003)
            
            # Create cache entry
            now = datetime.now()
            entry = CacheEntry(
                key=cache_key,
                response=response,
                tokens_used=tokens_used,
                cost_usd=cost_usd,
                model=model,
                created_at=now,
                expires_at=now + timedelta(seconds=ttl),
                hit_count=0,
                last_accessed=None
            )
            
            # Store in Redis
            self.redis_client.setex(
                cache_key,
                ttl,
                self._serialize_entry(entry)
            )
            
            logger.info(f"Cached response for prompt: {prompt[:50]}...")
            return True
            
        except (ConnectionError, RedisError) as e:
            logger.error(f"Redis error in set: {e}")
            return False
    
    def delete(self, prompt: str, model: str, temperature: float = 0.7) -> bool:
        """
        Delete a cached response.
        
        Args:
            prompt: The prompt text
            model: The model name
            temperature: The temperature setting
            
        Returns:
            True if deleted successfully, False otherwise
        """
        if not self._is_connected():
            return False
        
        try:
            cache_key = self._generate_cache_key(prompt, model, temperature)
            result = self.redis_client.delete(cache_key)
            return result > 0
        except (ConnectionError, RedisError) as e:
            logger.error(f"Redis error in delete: {e}")
            return False
    
    def clear(self) -> bool:
        """
        Clear all cached entries.
        
        Returns:
            True if cleared successfully, False otherwise
        """
        if not self._is_connected():
            return False
        
        try:
            # Get all keys with our prefix
            keys = self.redis_client.keys(f"{self.cache_prefix}*")
            if keys:
                self.redis_client.delete(*keys)
            
            # Clear stats
            stats_keys = self.redis_client.keys(f"{self.stats_prefix}*")
            if stats_keys:
                self.redis_client.delete(*stats_keys)
            
            logger.info(f"Cleared {len(keys)} cache entries")
            return True
            
        except (ConnectionError, RedisError) as e:
            logger.error(f"Redis error in clear: {e}")
            return False
    
    def _update_stats(self, hit: bool, tokens_saved: int = 0, cost_saved: float = 0.0):
        """Update cache statistics."""
        if not self._is_connected():
            return
        
        try:
            stats_key = f"{self.stats_prefix}cache_stats"
            
            # Get current stats
            current_stats = self.redis_client.get(stats_key)
            if current_stats:
                stats = json.loads(current_stats)
            else:
                stats = {
                    'total_requests': 0,
                    'cache_hits': 0,
                    'cache_misses': 0,
                    'total_tokens_saved': 0,
                    'total_cost_saved': 0.0
                }
            
            # Update stats
            stats['total_requests'] += 1
            if hit:
                stats['cache_hits'] += 1
                stats['total_tokens_saved'] += tokens_saved
                stats['total_cost_saved'] += cost_saved
            else:
                stats['cache_misses'] += 1
            
            # Calculate hit rate
            if stats['total_requests'] > 0:
                stats['hit_rate'] = stats['cache_hits'] / stats['total_requests']
            else:
                stats['hit_rate'] = 0.0
            
            # Store updated stats
            self.redis_client.setex(stats_key, 86400, json.dumps(stats))
            
        except (ConnectionError, RedisError) as e:
            logger.error(f"Redis error in _update_stats: {e}")
    
    def get_stats(self) -> CacheStats:
        """
        Get cache statistics.
        
        Returns:
            CacheStats object
        """
        if not self._is_connected():
            return CacheStats(
                total_requests=0,
                cache_hits=0,
                cache_misses=0,
                total_tokens_saved=0,
                total_cost_saved=0.0,
                hit_rate=0.0,
                entries_count=0,
                oldest_entry=None,
                newest_entry=None
            )
        
        try:
            # Get basic stats
            stats_key = f"{self.stats_prefix}cache_stats"
            stats_data = self.redis_client.get(stats_key)
            
            if stats_data:
                stats = json.loads(stats_data)
            else:
                stats = {
                    'total_requests': 0,
                    'cache_hits': 0,
                    'cache_misses': 0,
                    'total_tokens_saved': 0,
                    'total_cost_saved': 0.0,
                    'hit_rate': 0.0
                }
            
            # Get cache entries info
            cache_keys = self.redis_client.keys(f"{self.cache_prefix}*")
            entries_count = len(cache_keys)
            
            # Get oldest and newest entries
            oldest_entry = None
            newest_entry = None
            
            if cache_keys:
                # Sample a few entries to get timestamps
                sample_keys = cache_keys[:10] if len(cache_keys) > 10 else cache_keys
                timestamps = []
                
                for key in sample_keys:
                    try:
                        entry_data = self.redis_client.get(key)
                        if entry_data:
                            entry = self._deserialize_entry(entry_data)
                            timestamps.append(entry.created_at)
                    except Exception:
                        continue
                
                if timestamps:
                    oldest_entry = min(timestamps)
                    newest_entry = max(timestamps)
            
            return CacheStats(
                total_requests=stats['total_requests'],
                cache_hits=stats['cache_hits'],
                cache_misses=stats['cache_misses'],
                total_tokens_saved=stats['total_tokens_saved'],
                total_cost_saved=stats['total_cost_saved'],
                hit_rate=stats['hit_rate'],
                entries_count=entries_count,
                oldest_entry=oldest_entry,
                newest_entry=newest_entry
            )
            
        except (ConnectionError, RedisError) as e:
            logger.error(f"Redis error in get_stats: {e}")
            return CacheStats(
                total_requests=0,
                cache_hits=0,
                cache_misses=0,
                total_tokens_saved=0,
                total_cost_saved=0.0,
                hit_rate=0.0,
                entries_count=0,
                oldest_entry=None,
                newest_entry=None
            )
    
    def get_cache_entries(self, limit: int = 100) -> List[CacheEntry]:
        """
        Get a list of cache entries.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of CacheEntry objects
        """
        if not self._is_connected():
            return []
        
        try:
            cache_keys = self.redis_client.keys(f"{self.cache_prefix}*")
            entries = []
            
            for key in cache_keys[:limit]:
                try:
                    entry_data = self.redis_client.get(key)
                    if entry_data:
                        entry = self._deserialize_entry(entry_data)
                        entries.append(entry)
                except Exception as e:
                    logger.warning(f"Error deserializing entry {key}: {e}")
                    continue
            
            # Sort by creation date (newest first)
            entries.sort(key=lambda x: x.created_at, reverse=True)
            
            return entries
            
        except (ConnectionError, RedisError) as e:
            logger.error(f"Redis error in get_cache_entries: {e}")
            return []
    
    def cleanup_expired(self) -> int:
        """
        Clean up expired cache entries.
        
        Returns:
            Number of entries cleaned up
        """
        if not self._is_connected():
            return 0
        
        try:
            cache_keys = self.redis_client.keys(f"{self.cache_prefix}*")
            cleaned_count = 0
            
            for key in cache_keys:
                try:
                    entry_data = self.redis_client.get(key)
                    if entry_data:
                        entry = self._deserialize_entry(entry_data)
                        if datetime.now() > entry.expires_at:
                            self.redis_client.delete(key)
                            cleaned_count += 1
                except Exception as e:
                    logger.warning(f"Error checking expiration for {key}: {e}")
                    continue
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} expired cache entries")
            
            return cleaned_count
            
        except (ConnectionError, RedisError) as e:
            logger.error(f"Redis error in cleanup_expired: {e}")
            return 0
    
    def get_cost_savings(self) -> Dict[str, Any]:
        """
        Get cost savings information.
        
        Returns:
            Dictionary with cost savings data
        """
        stats = self.get_stats()
        
        return {
            'total_cost_saved': stats.total_cost_saved,
            'total_tokens_saved': stats.total_tokens_saved,
            'hit_rate': stats.hit_rate,
            'estimated_monthly_savings': stats.total_cost_saved * 30,  # Rough estimate
            'cache_efficiency': f"{stats.hit_rate:.1%}" if stats.hit_rate > 0 else "0%"
        }


# Global cache manager instance
_cache_manager = None


def get_cache_manager() -> CacheManager:
    """
    Get the global cache manager instance (singleton pattern).
    
    Returns:
        CacheManager instance
    """
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def cache_llm_response(prompt: str, response: str, model: str, tokens_used: int,
                      temperature: float = 0.7, ttl: Optional[int] = None) -> bool:
    """
    Convenience function for caching LLM responses.
    
    Args:
        prompt: The prompt text
        response: The response text
        model: The model name
        tokens_used: Number of tokens used
        temperature: The temperature setting
        ttl: Time-to-live in seconds
        
    Returns:
        True if cached successfully, False otherwise
    """
    manager = get_cache_manager()
    return manager.set(prompt, response, model, tokens_used, temperature, ttl)


def get_cached_response(prompt: str, model: str, temperature: float = 0.7) -> Optional[str]:
    """
    Convenience function for getting cached responses.
    
    Args:
        prompt: The prompt text
        model: The model name
        temperature: The temperature setting
        
    Returns:
        Cached response if found, None otherwise
    """
    manager = get_cache_manager()
    entry = manager.get(prompt, model, temperature)
    return entry.response if entry else None


def main():
    """Test the cache manager."""
    cache_manager = CacheManager()
    
    print("Testing Cache Manager...")
    
    # Test basic operations
    test_prompt = "Generate 3 headlines for a summer sale"
    test_response = "1. Summer Sale - 50% Off Everything!\n2. Hot Deals for Hot Days\n3. Cool Savings This Summer"
    test_model = "gpt-4o-mini"
    test_tokens = 150
    
    # Test caching
    success = cache_manager.set(test_prompt, test_response, test_model, test_tokens)
    print(f"Caching test: {'PASS' if success else 'FAIL'}")
    
    # Test retrieval
    cached_entry = cache_manager.get(test_prompt, test_model)
    if cached_entry:
        print(f"Retrieval test: PASS (hit count: {cached_entry.hit_count})")
    else:
        print("Retrieval test: FAIL")
    
    # Test stats
    stats = cache_manager.get_stats()
    print(f"Stats test: {stats.total_requests} requests, {stats.hit_rate:.1%} hit rate")
    
    # Test cost savings
    savings = cache_manager.get_cost_savings()
    print(f"Cost savings: ${savings['total_cost_saved']:.4f}")
    
    # Test cleanup
    cleaned = cache_manager.cleanup_expired()
    print(f"Cleanup test: {cleaned} entries cleaned")


if __name__ == "__main__":
    main()
