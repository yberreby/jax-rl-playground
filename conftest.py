import os

# Enable JAX compilation caching for faster test runs
os.environ["JAX_COMPILATION_CACHE_DIR"] = ".jax_cache"
os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
