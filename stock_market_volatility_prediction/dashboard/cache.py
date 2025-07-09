import redis


from flask_caching import Cache
from stock_market_volatility_prediction.utils.config import config


try:
    cache_config = {
        "CACHE_TYPE": "RedisCache",
        "CACHE_REDIS_URL": config.REDIS_URL
    }
    backend = "RedisCache"
except ImportError:
    cache_config = {
        "CACHE_TYPE": "SimpleCache",
        "CACHE_DEFAULT_TIMEOUT": 300
    }
    backend = "SimpleCache"

cache = Cache(config=cache_config)


if backend == "RedisCache":
    print("Using RedisCache for Dash caching")
else:
    print("Redis not found; using SimpleCache for Dash caching")
