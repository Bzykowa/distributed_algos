import argparse
import csv
import random
from typing import List, Tuple
from itertools import product

checked_configs = {}


class ProcessorRing:
    """
    A class that models the structure of the ring of processes in
    the Mutual Exclusion algorithm.
    """

    def __init__(self, n: int) -> None:
        """Initialize the ring of n processes to zeroes."""
        self.n = n
        self.P = [0 for _ in range(n)]

    def get_config(self):
        """Returns current config in a tuple."""
        return tuple(self.P)

    def set_random_config(self) -> None:
        """Set a configuration with random values."""
        self.P = [random.randint(0, self.n) for _ in range(self.n)]

    def set_config(self, P: List[int]) -> None:
        """
        Set a configuration manually.
        Must have values between 0 and n and P of length n.
        """
        if len(P) == self.n:
            if all(i in range(self.n+1) for i in P):
                self.P = P
            else:
                raise ValueError(
                    f"The values should be in range [0, {self.n}]"
                )
        else:
            raise ValueError(
                f"Length of the config list doesn't equal n ({self.n})."
            )

    def state_change(self, index: int) -> Tuple[int, bool]:
        """
        Return the current state (x) of the P_index and if the change
        has been successful.
        """
        if index == 0:
            if self.P[0] == self.P[self.n - 1]:
                self.P[0] = (self.P[0] + 1) % (self.n + 1)
                return self.P[0], True
            else:
                return self.P[0], False
        elif index in range(1, self.n+1):
            if self.P[index] != self.P[index - 1]:
                self.P[index] = self.P[index - 1]
                return self.P[index], True
            else:
                return self.P[index], False
        else:
            raise ValueError(f"There is no P_{index}.")

    def is_P_legal(self) -> bool:
        """
        Check if current configuration is legal.
        Only one process can make a state change.
        """
        access_allowed = 0
        if self.P[0] == self.P[self.n - 1]:
            access_allowed += 1
        for i in range(1, self.n):
            if self.P[i] != self.P[i - 1]:
                access_allowed += 1
        return access_allowed == 1

    def _is_config_legal(self, P) -> bool:
        """
        Check if P configuration is legal.
        Only one process can make a state change.
        """
        access_allowed = 0
        if P[0] == P[self.n - 1]:
            access_allowed += 1
        for i in range(1, self.n):
            if P[i] != P[i - 1]:
                access_allowed += 1
        return access_allowed == 1

    def able_to_move(self) -> List[int]:
        """Indexes that can move."""
        have_access = []
        if self.P[0] == self.P[self.n - 1]:
            have_access.append(0)
        for i in range(1, self.n):
            if self.P[i] != self.P[i - 1]:
                have_access.append(i)
        return have_access

    def get_next_move(self, prev: int) -> int:
        """Return the next possible moved based on previous."""
        possible_moves = self.able_to_move()
        idx = next((i for i in possible_moves if i > prev), -1)
        if idx == -1:
            return possible_moves[0]
        else:
            return idx

    def generate_illegal_configs(self):
        """
        Generate illegal configurations for verification.
        Returns tuples of n size.
        """
        for comb in product(range(self.n+1), repeat=self.n):
            if not self._is_config_legal(comb):
                yield comb


def test_config(ring: ProcessorRing) -> None:
    """
    Check how many steps it takes to reach a legal config from
    a specific move.
    """
    config = ring.get_config()
    if config not in checked_configs:
        checked_configs[config] = {}

        # get all possible moves
        moves = ring.able_to_move()

        for move in moves:
            checked_configs[config][move] = 1

            ring.state_change(move)
            next_conf = ring.get_config()
            next_move = ring.get_next_move(move)

            if ring.is_P_legal():
                continue
            elif next_conf in checked_configs and (
                next_move in checked_configs[next_conf]
            ):
                checked_configs[config][move] += (
                    checked_configs[next_conf][next_move]
                )
            else:
                test_config(ring)
                checked_configs[config][move] += (
                    checked_configs[next_conf][next_move]
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mutual Exclusion"
    )
    parser.add_argument(
        "--exp", default=0, type=int,
        help="Experiment to run. \n" +
        "1 - check how many illegal configs \n" +
        "2 - check every illegal configuration"
    )
    parser.add_argument(
        "--n", default=1, type=int,
        help="n - number of processors. Default: 2 \n"
    )
    args = parser.parse_args()
    ring = ProcessorRing(args.n)

    if args.exp == 1:
        bad_configs = 0
        for bad_config in ring.generate_illegal_configs():
            bad_configs += 1
        print(
            f"There is {bad_configs} illegal configs for n = {args.n}."
        )
    elif args.exp == 2:
        for bad_config in ring.generate_illegal_configs():
            ring.set_config(list(bad_config))
            try:
                test_config(ring)
            except RecursionError:
                print(
                    "1000 steps exceeded. Algorithm failed to stabilize for" +
                    f" {bad_config}"
                )
        print("Every illegal configuration verified")
        max_steps = 1

        with open(
            "me_n_" + str(args.n) + ".csv", "w", newline="", encoding="utf-8"
        ) as results:
            writer = csv.writer(results)
            header = ["Config"]
            for i in range(args.n):
                header.append(str(i))
            writer.writerow(header)
            for config in checked_configs:
                row = [config]
                for i in range(args.n):
                    if i in checked_configs[config]:
                        steps = checked_configs[config][i]
                        row.append(steps)
                        max_steps = steps if steps > max_steps else max_steps
                    else:
                        row.append(0)
                writer.writerow(row)

        print(f"Max steps to legal config: {max_steps}.")
    else:
        print("no such experiment")
