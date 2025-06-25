from jaxtyping import install_import_hook

# All submodules of `src` will get jaxtyping runtime checks.
with install_import_hook("src", "beartype.beartype"):
    from . import core

__all__ = ["core"]
