import torch
from torch import Tensor

def mean_exp(t: Tensor, exp: float) -> Tensor:
    assert exp > 0

    BASE_EXP = 5

    t = t / t.sum()

    while exp >= BASE_EXP:
        t = t ** exp
        t = t / t.sum()
        exp -= BASE_EXP
    
    if exp > 0:
        t = t ** exp

    t = t / t.sum()

    return t



print(mean_exp(Tensor([1, 2, 3]), 1))