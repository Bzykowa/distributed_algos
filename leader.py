import argparse
import math
import matplotlib.pyplot as plt
import statistics
import sys
from random import choices
from typing import Tuple


class Leader:
    def __init__(self, n: int, bounded: bool, u: int) -> None:
        self.n = n
        self.bounded = bounded
        self.u = u

    def p_bounded(self) -> float:
        """
        Generate a repeating sequence of probabilities for upper
        bounded version.
        """
        m = math.ceil(math.log2(self.u)) + 1
        i = 1
        while True:
            yield (1 / 2 ** i)
            i = 1 if i == m else i + 1

    def p(self) -> float:
        """Return a optimal probability for fast leader election."""
        return 1 / self.n

    def election(self) -> Tuple[int, int]:
        """
        Choose a leader.
        Return the successful slot and chosen device id.
        """
        i = 0
        slot = 0
        leader = 0
        p = self.p_bounded() if self.bounded else self.p()
        while slot != 1:
            i = i + 1
            slot = 0
            p_i = next(p) if self.bounded else p
            for j in range(1, self.n + 1):
                broadcasting = choices(
                    [False, True], weights=(100-p_i*100, p_i*100), k=1)[0]
                if broadcasting:
                    slot = slot + 1
                    leader = j
        return i, leader


def experiment1(n: int, bounded: bool, u: int, times: int) -> list:
    """Get a sample of the successful slots."""
    results = []
    for _ in range(times):
        leader = Leader(n, bounded, u)
        slot, _ = leader.election()
        results.append(slot)
    return results


def experiment2(n: int, bounded: bool, u: int, times: int) -> dict:
    """Get a sample of the successful slots plus occurences counted."""
    results = {}
    for _ in range(times):
        leader = Leader(n, bounded, u)
        slot, _ = leader.election()
        results[slot] = results[slot] + 1 if slot in results else 0
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Leader election")

    parser.add_argument("--bound", action="store_true",
                        help="Use a version with upper bound.")

    parser.add_argument("--n", default=2, type=int,
                        help="Number of nodes. Default: 2")
    parser.add_argument("--u", default=2, type=int,
                        help="Upper bound. Default: 2")
    parser.add_argument("--exp", default=0, type=int,
                        help="Experiment to run. Default: 0 \n" +
                        "0: run single election with given parameters.\n")
    args = parser.parse_args()

    # Check if values are appropriate
    try:
        if args.bound:
            assert args.u >= args.n, "u smaller than n"
            assert args.n >= 2, "n smaller than 2"
        else:
            assert args.n >= 2, "n smaller than 2"
    except AssertionError as msg:
        sys.exit(msg)

    if args.exp == 0:
        leader = Leader(args.n, args.bound, args.u)
        slot, lead = leader.election()
        print(f"Success in slot {slot}.")
        print(f"The leader is device nr {lead}.")
    elif args.exp == 1:
        results1 = experiment1(args.n, False, args.u, 1000)
        results2 = experiment1(2, True, args.u, 1000)
        results3 = experiment1(args.u // 2, True, args.u, 1000)
        results4 = experiment1(args.u, True, args.u, 1000)

        plt.rcParams.update({'figure.figsize': (7, 5), 'figure.dpi': 100})

        # Plot Histograms
        # Geometric looking distribution
        plt.figure(1)
        plt.hist(results1, bins=50)
        plt.gca().set(
            title='Unbounded, n = '+str(args.n), ylabel='Frequency')
        # Geometric looking distribution
        plt.figure(2)
        plt.hist(results2, bins=50)
        plt.gca().set(
            title='Bounded, n = 2, u ='+str(args.u), ylabel='Frequency')
        # Binomial looking distribution
        plt.figure(3)
        plt.hist(results3, bins=50)
        plt.gca().set(
            title='Bounded, n = u/2, u ='+str(args.u), ylabel='Frequency')
        # Binomial looking distribution
        plt.figure(4)
        plt.hist(results4, bins=50)
        plt.gca().set(
            title='Bounded, n = u, u ='+str(args.u), ylabel='Frequency')

        plt.show()
    elif args.exp == 2:
        results_exp = experiment2(args.n, False, args.u, 1000)
        results_var = experiment1(args.n, False, args.u, 1000)
        expected_value = 0
        for slot, occurences in results_exp.items():
            expected_value = expected_value + (occurences/1000) * slot
        variance = statistics.variance(results_var)

        print(f"estimated E[L] = {expected_value}")
        print(f"theoretical E[L] < {math.e}")

        print(f"estimated Var[L] = {variance}")
        print(f"theoretical Var[L] > {(1 - 1/math.e)/((1/math.e) ** 2)}")
    elif args.exp == 3:
        # P_r[S_{L,n}] ≥ λ ≈ 0.579
        pass
    else:
        print("Wrong experiment id.")
