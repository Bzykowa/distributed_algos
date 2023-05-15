import argparse
import math
import random

from matplotlib import pyplot as plt
from multiset import Multiset

from test_hashes import FloatHash, testhash
from multiset_utils import generateMultiset, withinDistance


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


def graphTask7(x_axis: list, y_axis: list, chebyshev: float, a: float,
               save: bool = False) -> None:

    plt.axhline(1 + chebyshev, linestyle='--',
                label=f'{chebyshev=}', color='black')
    plt.axhline(1 - chebyshev, linestyle='--', color='black')
    plt.scatter(x=x_axis, y=y_axis, c='cyan', s=5)
    plt.xlabel('n')
    plt.ylabel(r'$\frac{\hat{n}}{n}$')
    plt.title(r'Concentration results $\frac{\hat{n}}{n}$ for ' + f'{a=}. (7)')
    plt.legend()
    if save is True:
        plt.savefig(f'figures/task7{a=}.png')
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
    h = FloatHash(256, "sha256")
    N = (10 ** 3) * 3
    plot_x = [i for i in range(1, N+1)]
    plot_y = []
    start = 1
    results = []

    for n in range(1, N+1):
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
    h_len = [8, 16, 32, 64, 128]
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
    h = FloatHash(256, "sha256")
    N = (10 ** 3) * 2
    start = 1
    plot_x = [i for i in range(1, N+1)]
    plot_y = []
    alpha = [0.05, 0.01, 0.005]

    for n in range(1, N+1):
        end = start + n
        mset = generateMultiset(start, end, (1, 1))
        n_hat = minCount(k, h, mset)
        plot_y.append(n_hat/n)
        start = end

    for a in alpha:
        chebyshev = math.sqrt(1 / (k * a))
        graphTask7(plot_x, plot_y, chebyshev, a, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MinCount")
    parser.add_argument(
        "--exp", default="5a", type=str,
        help="Experiment to run. Default: 5a \n"
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
