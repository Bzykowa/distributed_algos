import random
from multiset import Multiset
from typing import Tuple


def generateMultiset(
    start: int,
    end: int,
    m_range: Tuple[int, int]
) -> Multiset:
    """Generate multiset with values between start and end.
    Their multiplicity will be a random number in m_range."""

    mset = Multiset()
    a, b = m_range
    for i in range(start, end):
        mul = random.randint(a, b) if a < b else a
        mset.add(i, mul)
    return mset


def withinDistance(data, distance=0.1):
    """Returns percentage of values that are in specific distance from 1."""
    if len(data) == 0:
        return False
    count = len(list(filter(lambda x: abs(x - 1) < distance, data)))
    return count / len(data)
