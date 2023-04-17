from matplotlib import pyplot as plt
from multiset import Multiset
import numpy as np
from scipy.integrate import quad
import argparse
import random
from test_hashes import Hash32Bit

MIN_B = 4
MAX_B = 16
N = 10 ** 4

# Helper functions for HyperLogLog


def ro(s: str) -> int:
    """Return leading zeroes number plus 1."""
    idx = 0
    for c in s:
        if c == 0:
            idx = idx + 1
        else:
            break
    return idx + 1


def alpha_m_fu(u, m):
    """Return f(u) value for alpha_m calculations."""
    return np.power(np.log2(np.divide((2 + u), (1 + u))), m)


def calculate_alpha_m(m: int):
    """Return alpha_m value."""
    result, _ = quad(alpha_m_fu, 0, np.inf, args=[m])
    final_result = 1 / (result * m)
    return final_result


def hyperLogLog(M_set: Multiset, b: int, h: Hash32Bit) -> float:
    """
    Returns cardinality of distinct elements from a large multiset.

    Mult - a large multiset
    m - a number of experiments (2^b), b in [4,16]
    h - a FloatHash (of 32 length)
    """
    if b not in range(MIN_B, MAX_B+1) or h.B != 32:
        print("b not in [4,16] or hash is not 32 bit long")
        return -1

    m = 2 ** b
    alpha = calculate_alpha_m(m)

    M = [0 for _ in range(m)]
    elems = sorted(M_set)
    random.shuffle(elems)

    for e in elems:
        h_e = h(str(e).encode())
        # Divide hash to h_1h_2..h_b and h_b+1..h_32
        h_1 = h_e[:b]
        w = h_e[b:]
        j = int(h_1, 2) + 1

        M[j] = max(M[j], ro(w))

    E = alpha * np.power(m, 2) * (1 / sum([2 ** (-j) for j in M]))

    # Corrections (relative error +- 1.04/sqrt(m) from 2007 paper)
    if E <= 2.5*m:
        V = len(filter(lambda zero: zero == 0, M))
        if V > 0:
            return m * np.log(m/V)
        else:
            return E
    elif E <= (2**32) / 30:
        return E
    else:
        return -(2**32) * np.log(1 - (E/(2**32)))

# Tests

def test_different_b() -> None:
    """Test HyperLogLog for different lengths of m."""
    pass


def test_different_h() -> None:
    """Test HyperLogLog for different hash functions."""
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HyperLogLog")
    parser.add_argument(
        "--exp", default=0, type=int,
        help="Experiment to run. Default: 0 \n" +
        "1 - different m lengths \n" +
        "2 - different hash functions \n"
    )
    args = parser.parse_args()

    if args.exp == 1:
        test_different_b()
    elif args.exp == 2:
        test_different_h()
