from matplotlib import pyplot as plt
from typing import Callable
import numpy as np
import argparse
import math
import random

# Constants
# Boundaries of q
LOWER_BOUND_Q = 0
UPPER_BOUND_Q = 0.5
# Test values for experiments
N_S = [1, 3, 6, 12, 24, 48]
P_S = [0.001, 0.01, 0.1]
# Difference in chain length when attacker gives up
GIVE_UP_DIFF = 50
# Number of experiments in simulator
SIM_EXP_NUM = 10000

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


def single_attack(n: int, q: float) -> bool:
    """A single double spending attack experiment."""
    attacker = 0
    others = 0

    while others < n or attacker - others < 0:
        if random.uniform(0, 1) < q:
            attacker += 1
        else:
            others += 1

        if others - attacker >= GIVE_UP_DIFF:
            break

    return attacker - others >= 0


def double_spending_sim(n: int, q: float) -> float:
    """
    Experimental estimation of P(n,q) depending on n and q with Monte Carlo.
    """
    wins = 0

    for _ in range(SIM_EXP_NUM):
        if single_attack(n, q):
            wins += 1

    return wins / SIM_EXP_NUM


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


def success_by_q_agg(samples: int, save: bool):
    """Plot 1st experiment for all approximations."""
    colors = ["xkcd:neon green", "black", "xkcd:shocking pink",
              "xkcd:hot purple", "xkcd:sunflower yellow", "xkcd:cerulean blue"]
    styles = ["solid", "dashed", "dotted"]
    ps = [double_spending_sim, nakamoto, grunspan]
    names = ["Monte Carlo", "Nakamoto", "Grunspan"]
    plt.rcParams["figure.figsize"] = (15, 8)

    q_space = np.linspace(LOWER_BOUND_Q, UPPER_BOUND_Q, samples)
    for j, p in enumerate(ps):
        for i, n in enumerate(N_S):
            plt.plot(
                q_space, [p(n, q) for q in q_space],
                label=f"{names[j]}: n = {n}", color=colors[i],
                linestyle=styles[j]
            )

    plt.title(r"$P(n, q)$ - all approximations")
    plt.ylabel(r"$P(n, q)$")
    plt.xlabel("q")
    plt.legend()

    if save:
        plt.savefig("figures/task9c1.png")

    plt.show()
    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]


def n_by_q_agg(samples: int, save: bool) -> None:
    """Plot P(n,q) depending on q for different ns"""
    colors = ["xkcd:forest green", "gold", "black"]

    q_space = np.linspace(LOWER_BOUND_Q, UPPER_BOUND_Q-0.06, samples)
    styles = ["solid", "dashed", "dotted"]
    ps = [double_spending_sim, nakamoto, grunspan]
    names = ["Monte Carlo", "Nakamoto", "Grunspan"]
    plt.rcParams["figure.figsize"] = (15, 8)

    for j, p in enumerate(ps):
        for i, pbb in enumerate(P_S):
            n_space = []
            n = 1
            for q in q_space:
                while p(n, q) > pbb:
                    n = n + 1
                n_space.append(n)

            plt.plot(
                q_space, n_space, label=f"{names[j]}: p = {pbb}",
                color=colors[i], linestyle=styles[j]
            )

    plt.title(r"$n$ meeting $P(n, q)$ for $q$ - all approximations")
    plt.ylabel("n")
    plt.xlabel("q")
    plt.legend()

    if save:
        plt.savefig("figures/task9c2.png")

    plt.show()
    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]


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
        help="n - number of confirmations needed. Default: 1 \n"
    )
    parser.add_argument(
        "--q", default=0.25, type=float,
        help="q - computational power of an attacker (0,1). Default: 0.25 \n"
    )
    args = parser.parse_args()

    if args.exp == 1:
        success_by_q(nakamoto, 100, "Nakamoto", True)
        success_by_q(grunspan, 100, "Grunspan", True)
    elif args.exp == 2:
        n_by_q(nakamoto, 100, "Nakamoto", True)
        n_by_q(grunspan, 100, "Grunspan", True)
    elif args.exp == 3:
        success_by_q_agg(100, True)
        n_by_q_agg(100, True)
    else:
        # run the simulator for inputted n and q
        pnq = double_spending_sim(args.n, args.q)
        print(
            "Probability that attacker outpaces others " +
            "when/after they have n confirmations."
        )
        print(f"n = {args.n}", f"q = {args.q}", f"P(n,q) = {pnq}", sep="\n")
