from matplotlib import pyplot as plt
from typing import Callable
import numpy as np
import argparse
import math

# Constants
LOWER_BOUND_Q = 0
UPPER_BOUND_Q = 0.5
N_S = [1, 3, 6, 12, 24, 48]
P_S = [0.001, 0.01, 0.1]

# P(n,q) approximate formulas


def nakamoto(n: int, q: float) -> float:
    p = 1 - q
    alpha = n * (q/p)
    return 1 - np.sum(
        [(np.exp(-alpha) * (alpha ** k) / math.factorial(k))
         * (1 - (q / p) ** (n - k)) for k in range(n)]
    )


def grunspan(n: int, q: float) -> float:
    p = 1 - q
    return 1 - np.sum(
        [(p ** n * q ** k - q ** n * p ** k) * math.comb(k + n - 1, k)
         for k in range(n)]
    )


def double_spending_sim(n: int, q: float) -> float:
    """
    Experimental estimation of P(n,q) depending on n and q with Monte Carlo.
    """
    pass

# Experiments


def success_by_q(p: Callable, samples: int, name: str, save: bool) -> None:
    """Plot P(n,q) depending on q for different ns"""
    colors = ["xkcd:neon green", "black", "xkcd:shocking pink",
              "xkcd:hot purple", "xkcd:sunflower yellow", "xkcd:cerulean blue"]

    q_space = np.linspace(LOWER_BOUND_Q, UPPER_BOUND_Q, samples)
    for c, n in enumerate(N_S):
        plt.plot(q_space, [p(n, q)
                           for q in q_space], label=f"n = {n}", color=colors[c]
                 )

    plt.title(r"$P(n, q)$ - "+name)
    plt.ylabel(r"$P(n, q)$")
    plt.xlabel("q")
    plt.legend()

    if save:
        plt.savefig("figures/task9a1" + name + ".png")

    plt.show()


def n_by_q(p: Callable, samples: int, name: str, save: bool) -> None:
    """Plot P(n,q) depending on q for different ns"""
    colors = ["xkcd:forest green", "gold", "black"]

    q_space = np.linspace(LOWER_BOUND_Q, UPPER_BOUND_Q-0.06, samples)

    for i, pbb in enumerate(P_S):
        n_space = []
        n = 1
        for q in q_space:
            while p(n, q) > pbb:
                n = n + 1
            n_space.append(n)

        plt.plot(q_space, n_space, label=f"p = {pbb}", color=colors[i])

    plt.title(r"$n$ meeting $P(n, q)$ for $q$ - "+name)
    plt.ylabel("n")
    plt.xlabel("q")
    plt.legend()

    if save:
        plt.savefig("figures/task9a2" + name + ".png")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Double Spending Analysis")
    parser.add_argument(
        "--exp", default=0, type=int,
        help="Experiment to run. \n" +
        "1 - P(n,q) depending on q (Nakamoto, Grunspan)\n" +
        "2 - n depending on q, P(n,q) (Nakamoto, Grunspan)\n" +
        "3 - attack simulator depending on n,q (1+2 for simulator)\n" +
        "other/no value - run simulator"
    )
    parser.add_argument(
        "--n", default=1, type=int,
        help="n parameter for double spending simulator. Default: 1 \n"
    )
    parser.add_argument(
        "--q", default=0.25, type=float,
        help="q parameter for double spending simulator. Default: 0.25 \n"
    )
    args = parser.parse_args()

    if args.exp == 1:
        success_by_q(nakamoto, 100, "Nakamoto", True)
        success_by_q(grunspan, 100, "Grunspan", True)
    elif args.exp == 2:
        n_by_q(nakamoto, 100, "Nakamoto", True)
        n_by_q(grunspan, 100, "Grunspan", True)
    elif args.exp == 3:
        pass
    else:
        # run the simulator for inputted n and q
        pass
