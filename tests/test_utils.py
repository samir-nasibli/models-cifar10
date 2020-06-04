import pytest

from utils import common_utils


def test_merge_two_dicts():
    a = {}
    b = {}
    c =  common_utils.merge_two_dicts(a,b)
    assert type(c) == dict
