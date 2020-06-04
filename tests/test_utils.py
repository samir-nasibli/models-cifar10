import pytest

from utils.common_utils import merge_two_dicts


def test_merge_two_dicts_return_type():
    a = {}
    b = {}
    c = merge_two_dicts(a,b)
    assert type(c) == dict
