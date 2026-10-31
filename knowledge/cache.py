"""Redis cache.

What we cache:
    qa    final answers that the Evaluator approved
    ret   retrieval results
    gen   Generator answers
    eval  Evaluator decisions
    emb   embeddings          
    doc   audio transcripts 

Old answers must not stay after the knowledge changes. When knowledge is added or deleted the number
goes up, and all the old cached answers are not found anymore.

If Redis is not running the app still works, just without the cache.
"""
import hashlib
import json
import logging

import redis


logger = logging.getLogger(__name__)

redis_client = redis.Redis.from_url(
    "redis://localhost:6379/0",
    decode_responses=True,
    socket_connect_timeout=2,
    socket_timeout=2,
)

redis_working = True  # becomes False after a Redis error
stats = {"hits": 0, "misses": 0, "errors": 0}


def _redis_failed(error):
    global redis_working
    redis_working = False
    stats["errors"] += 1
    logger.warning("Redis error: %s. The cache is off until Redis is back.", error)


def is_available():
    """Ping Redis. If Redis came back, this turns the cache on again."""
    global redis_working
    try:
        redis_client.ping()
        redis_working = True
    except redis.RedisError as error:
        _redis_failed(error)
    return redis_working


def get_stats():
    return dict(stats)


def normalize_question(question):
    """Same question with other spaces or capital letters uses the same cache entry."""
    return " ".join((question or "").lower().split())


# knowledge version
def knowledge_version():
    if not redis_working:
        return "0"
    try:
        return redis_client.get("kb:version") or "0"
    except redis.RedisError as error:
        _redis_failed(error)
        return "0"


def bump_knowledge_version():
    """Call this when knowledge is added or deleted."""
    if not redis_working:
        return
    try:
        version = redis_client.incr("kb:version")
        logger.info("Knowledge version is now %s (old cache is not used anymore)", version)
    except redis.RedisError as error:
        _redis_failed(error)


# cache that depends on the knowledge version
def _hash(parts):
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _make_key(prefix, parts):
    return f"{prefix}:v{knowledge_version()}:{_hash(parts)}"


def cache_get(prefix, *parts):
    """Return the saved value, or None if it is not in the cache."""
    if not redis_working:
        return None
    try:
        saved = redis_client.get(_make_key(prefix, parts))
        if saved is None:
            stats["misses"] += 1
            return None
        stats["hits"] += 1
        logger.info("cache hit [%s]", prefix)
        return json.loads(saved)
    except redis.RedisError as error:
        _redis_failed(error)
    except json.JSONDecodeError:
        logger.warning("Bad cache entry ignored [%s]", prefix)
    return None


def cache_set(prefix, value, *parts):
    if not redis_working:
        return
    try:
        redis_client.set(
            _make_key(prefix, parts),
            json.dumps(value, ensure_ascii=False),
            ex=3600,
        )
    except redis.RedisError as error:
        _redis_failed(error)


# cache that does not depend on the knowledge (embeddings, transcripts)
def content_key(prefix, *parts):
    return f"{prefix}:{_hash(parts)}"


def raw_get_many(keys):
    if not keys or not redis_working:
        return [None] * len(keys)
    try:
        return redis_client.mget(keys)
    except redis.RedisError as error:
        _redis_failed(error)
        return [None] * len(keys)


def raw_set_many(values, seconds):
    if not values or not redis_working:
        return
    try:
        pipe = redis_client.pipeline()
        for key, value in values.items():
            pipe.set(key, value, ex=seconds)
        pipe.execute()
    except redis.RedisError as error:
        _redis_failed(error)
