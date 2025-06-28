import os
import pytest
import jax

# Enable JAX compilation caching for faster test runs
os.environ["JAX_COMPILATION_CACHE_DIR"] = ".jax_cache"
os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"

@pytest.fixture(autouse=True)
def clear_jax_cache():
    """Clear JAX cache between test modules to avoid memory issues"""
    yield
    jax.clear_caches()