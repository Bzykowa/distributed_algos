from matplotlib import pyplot as plt
from multiset import Multiset
import numpy as np
from scipy.integrate import quad
import argparse
import random
from multiset_utils import generateMultiset
from test_hashes import Hash32Bit

MIN_B = 4
MAX_B = 16
N = 10 ** 3

# HyperLogLog and helper functions


def ro(s: str) -> int:
    """Return leading zeroes number plus 1."""
    idx = 0
    for c in s:
        if c == 0:
            idx = idx + 1
        else:
            break
    return idx


def alpha_m_fu(u, m):
    """Return f(u) value for alpha_m calculations."""
    return np.power(np.log2(np.divide((2 + u), (1 + u))), m)


def calculate_alpha_m(m: int):
    """Return alpha_m value."""
    result, _ = quad(alpha_m_fu, 0, np.inf, args=[m])
    final_result = np.power(result * m, -1)
    return final_result


def hyperLogLog(M_set: Multiset, b: int, h: Hash32Bit) -> float:
    """
    Returns cardinality of distinct elements from a large multiset.

    Mult - a large multiset
    m - a number of experiments (2^b)
    h - a hash function (of 32 length)
    """
    """
    if b not in range(MIN_B, MAX_B+1):
        print("b not in [4,16]")
        return -1
    """

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
        j = int(h_1, 2)

        M[j] = max(M[j], ro(w))

    E = alpha * np.power(m, 2) * (sum([2 ** (-j) for j in M]) ** (-1))

    # Corrections (relative error +- 1.04/sqrt(m) from 2007 paper)
    if E <= 2.5*m:
        V = len(list(filter(lambda zero: zero == 0, M)))
        if V > 0:
            return m * np.log(m/V)
        else:
            return E
    elif E <= (2**32) / 30:
        return E
    else:
        return -(2**32) * np.log(1 - (E/(2**32)))

# Graphs


def graphBs(x_axis: list, y_axis: list, save: bool = False) -> None:
    plt.scatter(x=x_axis, y=y_axis[0], c='red', label='m=2^4', s=5)
    plt.scatter(x=x_axis, y=y_axis[1], c='yellow', label='m=2^8', s=5)
    plt.scatter(x=x_axis, y=y_axis[2], c='green', label='m=2^12', s=5)
    plt.scatter(x=x_axis, y=y_axis[3], c='cyan', label='m=2^16', s=5)
    plt.xlabel('n')
    plt.ylabel(r'$\frac{\hat{n}}{n}$')
    plt.title(r'$\frac{\hat{n}}{n}$ for different m. (5b)')
    plt.legend()
    if save is True:
        plt.savefig('figures/task8_bs.png')
    plt.show()

# Tests


def test_different_b() -> None:
    """Test HyperLogLog for different lengths of m."""
    start = 1
    h = Hash32Bit("sha256")
    bs = [4, 8, 12, 16]
    plot_x = [i for i in range(1, N+1)]
    plot_y = [[], [], [], []]

    for i, b in enumerate(bs):
        print(f"b = {b}")
        for n in range(1, N+1):
            end = start + n
            mset = generateMultiset(start, end, (1, 2))
            n_hat = hyperLogLog(mset, b, h)
            plot_y[i].append(n_hat/n)
            start = end

    graphBs(plot_x, plot_y, True)


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
    else:
        # Test hash length
        h = Hash32Bit("sha256")
        data = h("2137".encode())
        print(f"h('2137') = {data}, len = {len(data)}")
