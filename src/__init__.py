# Runtime shape checking.
from jaxtyping import install_import_hook

hook = install_import_hook("src", "beartype.beartype")

from .normalize import normalize_obs, scale_rewards  # noqa: E402
from .init import sparse_init  # noqa: E402
from .reinforce import reinforce_loss  # noqa: E402
from .distributions import gaussian_log_prob  # noqa: E402

__all__ = [
    "normalize_obs",
    "scale_rewards",
    "sparse_init",
    "reinforce_loss",
    "gaussian_log_prob",
]
