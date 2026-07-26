import torch

from lm7.cache import input_signature


def test_equivalent_signatures_match():
    assert input_signature((torch.zeros(2, 3),), {}) == input_signature((torch.ones(2, 3),), {})


def test_shape_and_dtype_change_signature():
    base = input_signature((torch.zeros(2, 3),), {})
    assert base != input_signature((torch.zeros(3, 3),), {})
    assert base != input_signature((torch.zeros(2, 3, dtype=torch.float64),), {})
