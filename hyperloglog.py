from matplotlib import pyplot as plt
from multiset import Multiset
import numpy as np
from scipy.integrate import quad
import argparse
import random
from multiset_utils import generateMultiset, withinDistance
from test_hashes import Hash32Bit

MIN_B = 4
MAX_B = 16
N = 10 ** 3

# HyperLogLog and helper functions


def ro(s: str) -> int:
    """Return leading zeroes number plus 1."""
    idx = 0
    for c in s:
        if c == '0':
            idx = idx + 1
        else:
            break
    return idx + 1


def alpha_m_fu(u, m):
    """Return f(u) value for alpha_m calculations."""
    return np.log2(np.divide((2 + u), (1 + u))) ** m


def calculate_alpha_m(b: int):
    """Return alpha_m value."""
    # integral starts to give wrong results at high m values
    # so using alpha from paper for b >= 4
    if b < 4:
        m = 2 ** b
        result, _ = quad(alpha_m_fu, 0, np.inf, args=[m])
        final_result = (result * m) ** (-1)
        return final_result
    elif b == 4:
        return 0.673
    elif b == 5:
        return 0.697
    elif b == 6:
        return 0.709
    elif b >= 7:
        return 0.7213 / (1 + (1.079 / (2 ** b)))


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
    alpha = calculate_alpha_m(b)

    M = [0 for _ in range(m)]
    elems = sorted(M_set)
    random.shuffle(elems)

    for e in elems:
        h_e = h(str(e).encode())
        # Divide hash to h_1h_2..h_b and h_b+1..h_32
        h_1 = h_e[:b]
        w = h_e[b:]
        j = int(h_1, 2)
        #print(f"w = {w}, ro(w) = {ro(w)}, len(w) = {len(w)}")
        M[j] = max(M[j], ro(w))

    hm_sum = 0
    for i in M:
        hm_sum += 0.5 ** i
    hm_sum = 1 / hm_sum

    E = alpha * m * m * hm_sum
    #print(f"M = {M}")
    #print(f"alpha * m^2 = {alpha * m * m}, sum = {hm_sum}, E = {E}")

    # Corrections (relative error +- 1.04/sqrt(m) from 2007 paper)
    if E <= 2.5*m:
        V = len(list(filter(lambda zero: zero == 0, M)))
        if V > 0:
            return m * np.log(m/V), 1
        else:
            return E, 2
    elif E <= (2 ** 32) / 30:
        return E, 2
    else:
        log = ((2 ** 32) - E)/(2 ** 32)
        return -(2 ** 32) * np.log(log), 3

# Graphs


def graphBs(x_axis: list, y_axis: list, save: bool = False) -> None:
    plt.scatter(x=x_axis, y=y_axis[0], c='red', label='m=2^4', s=2)
    plt.scatter(x=x_axis, y=y_axis[1], c='yellow', label='m=2^8', s=2)
    plt.scatter(x=x_axis, y=y_axis[2], c='green', label='m=2^12', s=2)
    plt.scatter(x=x_axis, y=y_axis[3], c='cyan', label='m=2^16', s=2)
    plt.xlabel('n')
    plt.ylabel(r'$\frac{\hat{n}}{n}$')
    plt.title(r'$\frac{\hat{n}}{n}$ for different m.')
    plt.legend()
    if save is True:
        plt.savefig('figures/task8_bs.png')
    plt.show()


def graphHs(x_axis: list, y_axis: list, save: bool = False) -> None:
    plt.scatter(x=x_axis, y=y_axis[0], c='red', label='awful', s=2)
    plt.scatter(x=x_axis, y=y_axis[1], c='yellow', label='md5', s=2)
    plt.scatter(x=x_axis, y=y_axis[2], c='green', label='sha1', s=2)
    plt.scatter(x=x_axis, y=y_axis[3], c='cyan', label='sha256', s=2)
    plt.xlabel('n')
    plt.ylabel(r'$\frac{\hat{n}}{n}$')
    plt.title(r'$\frac{\hat{n}}{n}$ for different hashes.')
    plt.legend()
    if save is True:
        plt.savefig('figures/task8_hs.png')
    plt.show()


def graphBInDistance(x_axis: list, y_axis: list, b: int, distance: float = 0.1,
                     save: bool = False) -> None:

    plt.axhline(1 + distance, linestyle='--',
                label=f'{distance=}', color='black')
    plt.axhline(1 - distance, linestyle='--', color='black')
    plt.scatter(x=x_axis, y=y_axis, c='cyan', s=2)
    plt.xlabel('n')
    plt.ylabel(r'$\frac{\hat{n}}{n}$')
    plt.title(r'$|\frac{\hat{n}}{n} - 1| < 0.1$ for m='+f'2^{b}')
    plt.legend()
    if save is True:
        plt.savefig('figures/task8_b10percent.png')
    plt.show()

# Tests


def test_different_b() -> None:
    """Test HyperLogLog for different lengths of m."""
    h = Hash32Bit("sha256")
    bs = [4, 8, 12, 16]
    plot_x = [i for i in range(1, N+1)]
    plot_y = [[], [], [], []]
    ranges = [0, 0, 0]

    for i, b in enumerate(bs):
        print(f"b = {b}")
        start = 1
        for n in range(1, N+1):
            end = start + n
            mset = generateMultiset(start, end, (1, 1))
            #print(f"n = {n}")
            n_hat, logrange = hyperLogLog(mset, b, h)
            #print(f"n_hat = {n_hat}")
            ranges[logrange-1] += 1
            plot_y[i].append(n_hat/n)
            start = end

    graphBs(plot_x, plot_y, True)
    print(ranges)


def test_different_h() -> None:
    """Test HyperLogLog for different hash functions."""
    b = 12
    hs = [Hash32Bit("awful"), Hash32Bit("md5"),
          Hash32Bit("sha1"), Hash32Bit("sha256")]
    plot_x = [i for i in range(1, N+1)]
    plot_y = [[], [], [], []]
    ranges = [0, 0, 0]

    for i, h in enumerate(hs):
        print(f"h = {h.name}")
        start = 1
        for n in range(1, N+1):
            end = start + n
            mset = generateMultiset(start, end, (1, 1))
            n_hat, logrange = hyperLogLog(mset, b, h)
            ranges[logrange-1] += 1
            plot_y[i].append(n_hat/n)
            start = end

    graphHs(plot_x, plot_y, True)
    print(ranges)


def find_b_below_10_p_err() -> int:
    b = 3
    h = Hash32Bit("sha256")
    within = 0
    results = []
    plot_x = [i for i in range(1, N+1)]
    while within < 0.95:
        b += 1
        start = 1
        results.clear()
        for n in range(1, N+1):
            end = start + n
            mset = generateMultiset(start, end, (1, 1))
            n_hat, _ = hyperLogLog(mset, b, h)
            results.append(n_hat/n)
            start = end
        within = withinDistance(results)
    graphBInDistance(plot_x, results, b, save=True)
    return b


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
    elif args.exp == 3:
        b = find_b_below_10_p_err()
        # b = 9
        print(
            f"For m = 2^{b} there is at least 95% chance that" +
            " |hat{n}/n - 1| < 10%"
        )
    else:
        # Test hash length
        h = Hash32Bit("sha256")
        data = h("2137".encode())
        print(f"h('2137') = {data}, len = {len(data)}")
