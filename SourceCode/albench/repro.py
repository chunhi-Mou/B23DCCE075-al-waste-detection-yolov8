"""Determinism + dataset hashing."""
import hashlib
import os
import random


def set_seed(seed: int) -> None:
    """Seed Python/Numpy/Torch + cuDNN determinism guards."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) if hasattr(torch, "cuda") else None
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def sha256_manifest(relpaths: list[str]) -> str:
    """Order-independent hash of a file list (the split manifest)."""
    h = hashlib.sha256()
    for p in sorted(relpaths):
        h.update(p.encode())
        h.update(b"\n")
    return h.hexdigest()
