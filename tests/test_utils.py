import pytest

from utils import common_utils


def test_merge_two_dicts():
    a = {'a' : 1, 'b' : 2, 'c' : 3}
    b = {'b' : 2, 'd' : 4}
    expected_c = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    c =  common_utils.merge_two_dicts(a,b)
    assert c == expected_c


def test_merge_two_dicts_return_type():
    a = {}
    b = {}
    c =  common_utils.merge_two_dicts(a,b)
    assert type(c) == dict
