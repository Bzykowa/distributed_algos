from random import sample, randint, choice
from itertools import product
import networkx as nx
import matplotlib.pyplot as plt
import argparse


class GraphVisualization:
    def __init__(self):

        # visual is a list which stores all
        # the set of edges that constitutes a
        # graph
        self.visual = []

    # addEdge function inputs the vertices of an
    # edge and appends it to the visual list
    def addEdge(self, a, b):
        temp = [a, b]
        self.visual.append(temp)

    # In visualize function G is an object of
    # class Graph given by networkx G.add_edges_from(visual)
    # creates a graph with a given list
    # nx.draw_networkx(G) - plots the graph
    # plt.show() - displays the graph
    def visualize(self):
        G = nx.Graph()
        G.add_edges_from(self.visual)
        nx.draw_networkx(G)
        plt.show()


class Graph:
    def __init__(self, v_size: int) -> None:
        self.V = self.gen_V(v_size)
        self.E = self.gen_E(v_size)
        self.in_set = [choice([True, False]) for _ in range(v_size)]

    def gen_V(self, n: int) -> list[int]:
        """Generate vertex set from 1 to n."""
        return [i for i in range(1, n+1)]

    def gen_E(self, n: int) -> list[tuple[int, int]]:
        """Generate edges for the graph."""
        combs = product(range(1, n+1), repeat=2)
        filtered = filter(lambda x: x[0] < x[1], combs)
        return sample(list(filtered), randint(n+1, 2*n))

    def N(self, p: int) -> list[int]:
        """Neighborhood of a vertex."""
        neighbors = []
        for edge in self.E:
            if edge[0] == p:
                neighbors.append(edge[1])
            elif edge[1] == p:
                neighbors.append(edge[0])
        return neighbors

    def max_independent_set(self) -> list[int]:
        """Returns the maximal independent set."""
        no_change = 0
        n = len(self.V)
        while no_change < n + 1:
            for i in range(1, n+1):
                if not self.in_set[i-1] and all(
                    not (self.in_set[j-1]) for j in self.N(i)
                ):
                    self.in_set[i-1] = True
                    no_change = 0
                if self.in_set[i-1] and any(
                    self.in_set[j-1] for j in self.N(i)
                ):
                    self.in_set[i-1] = False
                    no_change = 0
                no_change += 1
        mis = []
        for i, state in enumerate(self.in_set):
            if state:
                mis.append(self.V[i])

        return mis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Maximal Independent Set"
    )
    parser.add_argument(
        "--n", default=5, type=int,
        help="n - number of vertexes. Default: 5 \n"
    )
    args = parser.parse_args()
    g = Graph(args.n)
    print(f"MIS: {g.max_independent_set()}")
    G = GraphVisualization()
    for edge in g.E:
        G.addEdge(edge[0], edge[1])
    G.visualize()
