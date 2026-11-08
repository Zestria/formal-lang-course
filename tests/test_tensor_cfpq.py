from copy import deepcopy
from random import Random

import networkx as nx
import pytest
from pyformlang.cfg import CFG, Production, Terminal, Variable
from pyformlang.rsa import RecursiveAutomaton

from project.cfpq import (
    cfg_to_rsm,
    ebnf_to_rsm,
    hellings_based_cfpq,
    tensor_based_cfpq,
)


def path_graph(labels):
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(range(len(labels) + 1))
    for index, label in enumerate(labels):
        graph.add_edge(index, index + 1, label=label)
    return graph


def test_cfg_to_rsm_preserves_production_bodies_and_custom_start():
    cfg = CFG.from_text("Root -> a B c | $\nB -> b B | b", start_symbol="Root")
    rsm = cfg_to_rsm(cfg)

    assert isinstance(rsm, RecursiveAutomaton)
    assert rsm.initial_label.value == "Root"
    assert rsm.get_box("Root").dfa.accepts(["a", "B", "c"])
    assert rsm.get_box("Root").dfa.accepts([])
    assert rsm.get_box("B").dfa.accepts(["b", "B"])
    assert not rsm.get_box("Root").dfa.accepts(["a", "b", "c"])
    assert tensor_based_cfpq(rsm, path_graph("abbc"), {0}, {4}) == {(0, 4)}


def test_cfg_to_rsm_preserves_literal_terminals_and_name_collisions():
    start, other = Variable("S"), Variable("A")
    cfg = CFG(
        start_symbol=start,
        productions={
            Production(start, [Terminal("S"), other, Terminal("|")]),
            Production(other, [Terminal("a b"), Terminal("*")]),
        },
    )
    graph = path_graph(["S", "a b", "*", "|"])
    assert tensor_based_cfpq(cfg_to_rsm(cfg), graph) == {(0, 4)}


@pytest.mark.parametrize("grammar", ["S -> A", "S -> S", "S -> A\nA -> S"])
def test_unproductive_variables_do_not_match_graph_labels(grammar):
    graph = path_graph(["A", "S"])
    assert tensor_based_cfpq(cfg_to_rsm(CFG.from_text(grammar)), graph) == set()


@pytest.mark.parametrize("graph_type", [nx.DiGraph, nx.MultiDiGraph])
def test_tensor_cfpq_handles_recursion_isolated_nodes_and_filters(graph_type):
    graph = graph_type()
    graph.add_nodes_from([10, 30, 70, 90, 110, 999])
    graph.add_edge(10, 30, label="a")
    graph.add_edge(30, 70, label="a")
    graph.add_edge(70, 90, label="b")
    graph.add_edge(90, 110, label="b")
    graph.add_edge(10, 110)  # Unlabeled edges do not participate.
    rsm = ebnf_to_rsm("S -> a S b | $")
    graph_before, rsm_before = deepcopy(graph), deepcopy(rsm)
    expected = {(node, node) for node in graph} | {(10, 110), (30, 90)}

    assert tensor_based_cfpq(rsm, graph) == expected
    assert tensor_based_cfpq(rsm, graph, set(), set()) == expected
    assert tensor_based_cfpq(rsm, graph, {10, 30}, {90, 110}) == {
        (10, 110),
        (30, 90),
    }
    assert tensor_based_cfpq(rsm, graph, {10, -1}, {110, -1}) == {(10, 110)}
    assert tensor_based_cfpq(rsm, graph, {-1}) == set()
    assert nx.utils.graphs_equal(graph, graph_before)
    assert rsm == rsm_before


def test_tensor_cfpq_handles_parallel_edges():
    graph = path_graph("ab")
    graph.add_edge(0, 1, label="c")
    assert tensor_based_cfpq(ebnf_to_rsm("S -> (a | c) b"), graph) == {(0, 2)}


@pytest.mark.parametrize(
    "grammar, expected",
    [
        ("S -> A B\nA -> $\nB -> C\nC -> $", {(7, 7), (12, 12)}),
        ("S -> A B\nA -> $\nB -> A S", set()),
        ("S -> a", set()),
    ],
)
def test_tensor_cfpq_handles_nullable_chains_without_edges(grammar, expected):
    graph = nx.DiGraph()
    graph.add_nodes_from([7, 12])
    assert tensor_based_cfpq(cfg_to_rsm(CFG.from_text(grammar)), graph) == expected


def test_tensor_cfpq_handles_empty_graph_and_grammar():
    assert tensor_based_cfpq(ebnf_to_rsm("S -> $"), nx.DiGraph()) == set()
    assert tensor_based_cfpq(cfg_to_rsm(CFG()), path_graph("a")) == set()


def test_ebnf_to_rsm_unites_repeated_heads_and_handles_empty_bodies():
    rsm = ebnf_to_rsm("\nS -> (a | b)* T\nS -> c\nT ->\nT -> d T\n")
    assert rsm.get_box("S").dfa.accepts(["a", "b", "T"])
    assert rsm.get_box("S").dfa.accepts(["c"])
    assert rsm.get_box("T").dfa.accepts([])
    assert rsm.get_box("T").dfa.accepts(["d", "T"])
    assert (0, 4) in tensor_based_cfpq(rsm, path_graph("abdd"))


@pytest.mark.parametrize("text", ["", "A -> a", "S a", "-> a"])
def test_ebnf_to_rsm_rejects_missing_start_or_malformed_rules(text):
    with pytest.raises(ValueError):
        ebnf_to_rsm(text)


@pytest.mark.parametrize(
    "cfg_text, ebnf",
    [
        ("S -> $ | a S", "S -> a*"),
        ("S -> A B\nA -> a A | $\nB -> b B | $", "S -> a* b*"),
        ("S -> a S b | $ | S S", "S -> (a S b)*"),
        ("S -> a S b | c S d | $ | S S", "S -> (a S b | c S d)*"),
        ("S -> a B | b B\nB -> c B | d", "S -> (a | b) c* d"),
        ("S -> A\nA -> B\nB -> S | a", "S -> A\nA -> B\nB -> S | a"),
    ],
)
def test_tensor_cfpq_matches_hellings_on_seeded_graphs(cfg_text, ebnf):
    cfg = CFG.from_text(cfg_text)
    from_cfg, from_ebnf = cfg_to_rsm(cfg), ebnf_to_rsm(ebnf)
    random = Random(8)
    for _ in range(5):
        graph = nx.MultiDiGraph()
        nodes = [10, 30, 50, 70, 90]
        graph.add_nodes_from(nodes)
        for source in nodes:
            for target in nodes:
                for label in "abcd":
                    if random.random() < 0.12:
                        graph.add_edge(source, target, label=label)
        for starts, finals in [(None, None), ({10, 30}, {70, 90})]:
            expected = hellings_based_cfpq(cfg, graph, starts, finals)
            assert tensor_based_cfpq(from_cfg, graph, starts, finals) == expected
            assert tensor_based_cfpq(from_ebnf, graph, starts, finals) == expected
