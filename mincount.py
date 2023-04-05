from matplotlib import pyplot as plt
from multiset import Multiset
from typing import Tuple
import argparse
import hashlib
import random
import numpy as np


class AwfulHash:
    """A bad hash for task 6."""

    def __init__(self, data):
        self.data = data

    def hexdigest(self):
        hash_value = 0
        for byte in self.data:
            hash_value += byte
            hash_value = hash_value
        hash_value = (hash_value + 2 ** 8) % 2 ** 256
        # Sum of bytes + 2^8, padded with 01, cut to 256 len and reversed
        hash = int((format(hash_value, '02b')+"01"*128)[:256][::-1], 2)
        return hex(hash)


# hash functions for test purposes
HASH_FUNS = {
    "awful": AwfulHash,
    "blake2b": hashlib.blake2b,
    "md5": hashlib.md5,
    "sha1": hashlib.sha1,
    "sha256": hashlib.sha256
}


class FloatHash:
    """
    Class casting a hash function to a float.
    """

    def __init__(self, bit_length: int, function_name: str) -> None:
        self.B = bit_length
        self.h = HASH_FUNS[function_name]

    def __call__(self, data: bytes) -> float:
        hehe = self.h(data).hexdigest()
        # Convert to binary, delete "0b" and cut to B length
        binary_form = bin(int(hehe, 16))[2:self.B+2]
        return int(binary_form, 2) / (2 ** self.B)


def testhash():
    # h_len = [8, 16, 32, 64, 128]
    h_fun = ["awful", "blake2b", "md5", "sha1", "sha256"]
    for fun in h_fun:
        results = []
        for i in range(10**4):
            hash = FloatHash(160, fun)
            results.append(hash(str(i).encode()))
        print(f"{fun} mean = {np.mean(results)}")
        print(f"{fun} var = {np.var(results)}")


def minCount(k: int, h: FloatHash, M_set: Multiset):
    """
    Calculates expected value of distinct elements in a multiset M.
    """
    M = [1 for _ in range(k)]

    elems = sorted(M_set)
    random.shuffle(elems)

    for s in elems:
        h_s = h(str(s).encode())
        if h_s not in M and h_s < M[-1]:
            M[-1] = h_s
            M.sort()

    if M[-1] == 1:
        # number of elements different than 1
        return len(M) - M.count(1)
    else:
        return (k-1)/M[-1]


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


def graphTask5b(x_axis: list, y_axis: list, save: bool = False) -> None:
    plt.scatter(x=x_axis, y=y_axis[0], c='red', label='k=2', s=5)
    plt.scatter(x=x_axis, y=y_axis[1], c='yellow', label='k=3', s=5)
    plt.scatter(x=x_axis, y=y_axis[2], c='green', label='k=10', s=5)
    plt.scatter(x=x_axis, y=y_axis[3], c='cyan', label='k=100', s=5)
    plt.scatter(x=x_axis, y=y_axis[4], c='magenta', label='k=400', s=5)
    plt.xlabel('n')
    plt.ylabel(r'$\frac{\hat{n}}{n}$')
    plt.title(r'$\frac{\hat{n}}{n}$ for different k. (5b)')
    plt.legend()
    if save is True:
        plt.savefig('figures/task5_b.png')
    plt.show()


def graphTask5c(x_axis: list, y_axis: list, save: bool = False) -> None:
    plt.scatter(x=x_axis, y=y_axis, c='cyan', s=5)
    plt.xlabel('n')
    plt.ylabel('k')
    plt.title(r'Plotting $k$ such that $|\frac{\hat{n}}{n} - 1| < 0.1$. (5c)')
    if save is True:
        plt.savefig('figures/task5_c.png')
    plt.show()


def graphTask6len(x_axis: list, y_axis: list, save: bool = False) -> None:
    plt.scatter(x=x_axis, y=y_axis[0], c='red', label='len=32', s=5)
    plt.scatter(x=x_axis, y=y_axis[1], c='yellow', label='len=64', s=5)
    plt.scatter(x=x_axis, y=y_axis[2], c='green', label='len=128', s=5)
    plt.scatter(x=x_axis, y=y_axis[3], c='cyan', label='len=160', s=5)
    plt.scatter(x=x_axis, y=y_axis[4], c='magenta', label='len=256', s=5)
    plt.xlabel('n')
    plt.ylabel(r'$\frac{\hat{n}}{n}$')
    plt.title(r'$\frac{\hat{n}}{n}$ for different hash length. (6)')
    plt.legend()
    if save is True:
        plt.savefig('figures/task6_len.png')
    plt.show()


def graphTask6fun(x_axis: list, y_axis: list, save: bool = False) -> None:
    plt.scatter(x=x_axis, y=y_axis[0], c='red', label='awful hash', s=5)
    plt.scatter(x=x_axis, y=y_axis[1], c='yellow', label='blake2b', s=5)
    plt.scatter(x=x_axis, y=y_axis[2], c='green', label='md5', s=5)
    plt.scatter(x=x_axis, y=y_axis[3], c='cyan', label='sha1', s=5)
    plt.scatter(x=x_axis, y=y_axis[4], c='magenta', label='sha256', s=5)
    plt.xlabel('n')
    plt.ylabel(r'$\frac{\hat{n}}{n}$')
    plt.title(r'$\frac{\hat{n}}{n}$ for different hash function. (6)')
    plt.legend()
    if save is True:
        plt.savefig('figures/task6_fun.png')
    plt.show()


def task5a() -> None:
    """
    Check whether multiplicity affects the results.
    """
    k = 100
    h = FloatHash(64, "sha256")
    N = 10 ** 4
    start = 1
    effect = False

    for n in range(1, N+1):
        print(f"n = {n}")
        mul_check = []
        end = start + n
        for multiplicity in [(1, 2), (1, 3)]:
            mset = generateMultiset(start, end, multiplicity)
            n_hat = minCount(k, h, mset)
            mul_check.append(n_hat)

        if not all(i == mul_check[0] for i in mul_check):
            effect = True
            break

        start = end

    if effect:
        print("Multiplicity has an effect on MinCount.")
    else:
        print("Multiplicity has no effect on MinCount.")


def task5b() -> None:
    """
    Check standard error of different k values and graph results.
    """
    k = [2, 3, 10, 100, 400]
    h = FloatHash(160, "sha1")
    N = 10 ** 3
    plot_x = [i for i in range(1, N+1)]
    plot_y = [[] for _ in range(len(k))]
    start = 1

    for i, ki in enumerate(k):
        print(f"k = {ki}")
        for n in range(1, N+1):
            end = start + n
            mset = generateMultiset(start, end, (1, 1))
            n_hat = minCount(ki, h, mset)
            plot_y[i].append(n_hat/n)
            start = end

    graphTask5b(plot_x, plot_y, True)


def task5c() -> None:
    """
    Find k for which there is 95% chance that SE < 10%
    """
    k = 1
    results = []
    h = FloatHash(256, "sha256")
    N = (10 ** 3) * 2
    plot_x = [i for i in range(1, N+1)]
    plot_y = []
    start = 1

    for n in range(1, N+1):
        print(n)
        end = start + n
        mset = generateMultiset(start, end, (1, 1))
        n_hat = minCount(k, h, mset)
        results.append(n_hat/n)
        while withinDistance(results) < 0.95:
            k += 1
            n_hat = minCount(k, h, mset)
            results[-1] = (n_hat/n)

        plot_y.append(k)
        start = end

    graphTask5c(plot_x, plot_y, True)


def task6() -> None:
    k = 100
    N = 10 ** 3
    h_len = [32, 64, 128, 160, 256]
    h_fun = ["awful", "blake2b", "md5", "sha1", "sha256"]
    h_len2 = [256, 256, 128, 160, 256]
    plot_x = [i for i in range(1, N+1)]
    plot_y_len = [[] for _ in range(5)]
    plot_y_fun = [[] for _ in range(5)]

    # length
    start = 1
    for i, li in enumerate(h_len):
        print(f"len = {li}")
        for n in range(1, N+1):
            end = start + n
            mset = generateMultiset(start, end, (1, 1))
            h = FloatHash(li, "sha256")
            n_hat = minCount(k, h, mset)
            plot_y_len[i].append(n_hat/n)
            start = end
    # hash function
    start = 1
    for i, hi in enumerate(h_fun):
        print(f"fun = {hi}")
        for n in range(1, N+1):
            end = start + n
            mset = generateMultiset(start, end, (1, 1))
            h = FloatHash(h_len2[i], hi)
            n_hat = minCount(k, h, mset)
            plot_y_fun[i].append(n_hat/n)
            start = end
    graphTask6len(plot_x, plot_y_len, True)
    graphTask6fun(plot_x, plot_y_fun, True)


def task7() -> None:
    k = 400
    h = FloatHash(64, "sha256")
    N = 10 ** 4
    start = 1
    # plot_x = [i for i in range(1, N+1)]
    plot_y = [[] for _ in range(4)]
    alpha = [0.05, 0.01, 0.005]
    for i, ai in enumerate(alpha):
        print(f"alpha = {ai}")
        for n in range(1, N+1):
            end = start + n
            mset = generateMultiset(start, end, (1, 1))
            n_hat = minCount(k, h, mset)
            plot_y[i].append(n_hat/n)
            start = end


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MinCount")
    parser.add_argument(
        "--exp", default="5a", type=str,
        help="Experiment to run. Default: 0 \n"
    )
    args = parser.parse_args()

    if args.exp == "5a":
        task5a()
    elif args.exp == "5b":
        task5b()
    elif args.exp == "5c":
        task5c()
    elif args.exp == "6":
        task6()
    elif args.exp == "7":
        task7()
    else:
        testhash()
