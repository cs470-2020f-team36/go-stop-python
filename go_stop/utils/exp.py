"""
exp.py

Define `mean_exp` method.
"""


from torch import Tensor


BASE_EXP = 5


def mean_exp(tensor: Tensor, exp: float) -> Tensor:
    """
    Return `(tensor ** exp) / (tensor ** exp).sum()` correctly,
    especially when exp is a large positive number.
    """

    assert exp > 0

    tensor = tensor / tensor.sum()

    while exp >= BASE_EXP:
        tensor = tensor ** BASE_EXP
        tensor = tensor / tensor.sum()
        exp -= BASE_EXP

    if exp > 0:
        tensor = tensor ** exp

    tensor = tensor / tensor.sum()

    return tensor
