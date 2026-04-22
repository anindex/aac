"""Common type aliases for AAC."""

import torch

# Type aliases for documentation clarity
GraphTensor = torch.Tensor
"""Alias for torch.Tensor used in graph operations."""

AnchorIndices = torch.Tensor
"""Alias for torch.Tensor holding anchor vertex indices (int64)."""
