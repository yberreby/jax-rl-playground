from jaxtyping import install_import_hook

with install_import_hook("core", "beartype.beartype"):
    from . import core

__all__ = ["core"]
