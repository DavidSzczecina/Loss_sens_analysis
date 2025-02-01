# utils.py
import numpy as np
import torch
import warnings
import builtins

def set_seed(seed: int = 321) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    warnings.filterwarnings("ignore", "Lazy modules are a new feature.*")

_original_print = builtins.print

def disable_print():
    builtins.print = lambda *args, **kwargs: None

def enable_print():
    builtins.print = _original_print
