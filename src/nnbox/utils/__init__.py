"""Shared helpers (weight init, masking, shape checks).

Re-exported for convenience: `from nnbox.utils import padding_mask_to_bias`.
These are internal plumbing, not part of the package's public layer API, so
they stay out of the top-level `nnbox` namespace.
"""

from nnbox.utils.masking import padding_mask_to_bias

__all__ = ["padding_mask_to_bias"]
