import random

import networkx as nx
import pytest
from pyformlang.cfg import CFG

from project.cfpq import hellings_based_cfpq
from project.cfpq_matrix import matrix_based_cfpq


def make_graph(
    nodes: list[int] | None = None,
    edges: list[tuple[int, int, str]] | None = None,
) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    if nodes is not None:
        graph.add_nodes_from(nodes)
    for source, target, label in edges or []:
        graph.add_edge(source, target, label=label)
    return graph


@pytest.mark.parametrize(
    "graph, cfg_text, expected",
    [
        (make_graph([1]), "S -> a", set()),
        (make_graph([1]), "S -> S a | $", {(1, 1)}),
        (
            make_graph([0, 1, 2]),
            "S -> S a | $",
            {(0, 0), (1, 1), (2, 2)},
        ),
        (
            make_graph([0, 1], [(0, 1, "b")]),
            "S -> S b | $",
            {(0, 0), (0, 1), (1, 1)},
        ),
        (
            make_graph(
                [0, 1, 2, 3, 4],
                [(0, 1, "a"), (1, 2, "a"), (2, 3, "b"), (3, 4, "b")],
            ),
            "S -> $ | a S b | S S",
            {(node, node) for node in range(5)} | {(0, 4), (1, 3)},
        ),
        (
            make_graph(
                [0, 1, 2],
                [(0, 1, "a"), (1, 2, "a"), (2, 0, "a")],
            ),
            "S -> a | S S",
            {(source, target) for source in range(3) for target in range(3)},
        ),
    ],
)
def test_matrix_based_cfpq_concrete_cases(graph, cfg_text, expected):
    assert matrix_based_cfpq(CFG.from_text(cfg_text), graph) == expected


def test_matrix_based_cfpq_filters_start_and_final_nodes():
    graph = make_graph(
        list(range(6)),
        [(0, 1, "a"), (1, 2, "b"), (3, 4, "a"), (4, 5, "b")],
    )
    cfg = CFG.from_text("S -> a b")

    assert matrix_based_cfpq(cfg, graph, {0, 3}, {2, 5}) == {(0, 2), (3, 5)}
    assert matrix_based_cfpq(cfg, graph, {0}, {2, 5}) == {(0, 2)}
    assert matrix_based_cfpq(cfg, graph, {0, 3}, {5}) == {(3, 5)}


def test_matrix_based_cfpq_empty_sets_and_unknown_nodes_are_ignored():
    graph = make_graph([0, 1, 2], [(0, 1, "a"), (1, 2, "b")])
    cfg = CFG.from_text("S -> a b")

    # An empty (or missing) filter means "all nodes", matching Hellings.
    assert matrix_based_cfpq(cfg, graph, set(), set()) == {(0, 2)}
    assert matrix_based_cfpq(cfg, graph, {100}, {2}) == set()
    assert matrix_based_cfpq(cfg, graph, {0}, {100}) == set()


def test_matrix_based_cfpq_handles_empty_graph():
    graph = make_graph()

    assert matrix_based_cfpq(CFG.from_text("S -> a"), graph) == set()
    assert matrix_based_cfpq(CFG.from_text("S -> $"), graph) == set()


def test_matrix_based_cfpq_handles_isolation_epsilon_and_unlabeled_edges():
    graph = make_graph([0, 1, 2], [(0, 1, "a")])
    graph.add_edge(1, 2)  # An edge without a label is simply ignored.

    assert matrix_based_cfpq(CFG.from_text("S -> $"), graph) == {
        (0, 0),
        (1, 1),
        (2, 2),
    }
    assert matrix_based_cfpq(CFG.from_text("S -> a"), graph) == {(0, 1)}


def test_matrix_based_cfpq_supports_arbitrary_node_values():
    graph = nx.MultiDiGraph()
    graph.add_edge("start", "middle", label="a")
    graph.add_edge("middle", "end", label="b")

    result = matrix_based_cfpq(CFG.from_text("S -> a b"), graph, {"start"}, {"end"})

    assert result == {("start", "end")}


def test_matrix_based_cfpq_matches_hellings_on_random_graphs():
    rng = random.Random(20240517)
    grammars = [
        CFG.from_text("S -> a | S S"),
        CFG.from_text("S -> $ | a S b | S S"),
        CFG.from_text("S -> a b | b a"),
        CFG.from_text("S -> $ | a S | b S"),
        CFG.from_text("S -> A B\nA -> a\nB -> b | $"),
        CFG.from_text("S -> a S b | S S | $"),
        CFG.from_text("S -> $ | a S b | c S d"),
        CFG.from_text("S -> a b c"),
    ]

    for _ in range(20):
        size = rng.randint(1, 8)
        graph = nx.MultiDiGraph()
        graph.add_nodes_from(range(size))
        for _ in range(rng.randint(0, size * 3)):
            source = rng.randrange(size)
            target = rng.randrange(size)
            graph.add_edge(source, target, label=rng.choice(["a", "b", "c", "d"]))

        start_nodes = set(rng.sample(range(size), rng.randint(0, size)))
        final_nodes = set(rng.sample(range(size), rng.randint(0, size)))

        for cfg in grammars:
            assert matrix_based_cfpq(
                cfg, graph, start_nodes, final_nodes
            ) == hellings_based_cfpq(cfg, graph, start_nodes, final_nodes)
