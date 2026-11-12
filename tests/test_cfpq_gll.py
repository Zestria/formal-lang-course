from copy import deepcopy
from random import Random

import networkx as nx
import pytest
from pyformlang.cfg import CFG, Production, Terminal, Variable
from pyformlang.rsa import RecursiveAutomaton

from project.cfpq import cfg_to_rsm, ebnf_to_rsm, hellings_based_cfpq
from project.cfpq_gll import gll_based_cfpq


def path_graph(labels):
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(range(len(labels) + 1))
    graph.add_edges_from(
        (index, index + 1, {"label": label}) for index, label in enumerate(labels)
    )
    return graph


@pytest.mark.parametrize("graph_type", [nx.DiGraph, nx.MultiDiGraph])
def test_gll_handles_recursion_filters_and_isolated_vertices(graph_type):
    graph = graph_type()
    graph.add_nodes_from([10, 30, 70, 90, 110, 999])
    graph.add_edges_from(
        [(10, 30, {"label": "a"}), (30, 70, {"label": "a"})]
        + [(70, 90, {"label": "b"}), (90, 110, {"label": "b"})]
    )
    graph.add_edge(10, 110)  # No label: this is not a terminal transition.
    rsm = ebnf_to_rsm("S -> a S b | $")
    graph_before, rsm_before = deepcopy(graph), deepcopy(rsm)
    expected = {(node, node) for node in graph} | {(10, 110), (30, 90)}

    assert gll_based_cfpq(rsm, graph) == expected
    assert gll_based_cfpq(rsm, graph, set(), set()) == expected
    assert gll_based_cfpq(rsm, graph, {10, 30}, {90, 110}) == {(10, 110), (30, 90)}
    assert gll_based_cfpq(rsm, graph, {10, -1}, {110, -1}) == {(10, 110)}
    assert gll_based_cfpq(rsm, graph, {-1}) == set()
    assert gll_based_cfpq(rsm, graph, {10}, {-1}) == set()
    assert gll_based_cfpq(rsm, graph, {999}, {999}) == {(999, 999)}
    assert nx.utils.graphs_equal(graph, graph_before)
    assert rsm == rsm_before


@pytest.mark.parametrize(
    "grammar",
    [
        "S -> S | $",
        "S -> A\nA -> B\nB -> S | $",
        "S -> A A\nA -> $",
        "S -> S S | S S S | $",
    ],
)
def test_gll_handles_nullable_and_left_recursive_cycles(grammar):
    graph = nx.DiGraph()
    graph.add_nodes_from([3, 7, 100])
    rsm = cfg_to_rsm(CFG.from_text(grammar))
    assert gll_based_cfpq(rsm, graph) == {(node, node) for node in graph}


@pytest.mark.parametrize("grammar", ["S -> S", "S -> A", "S -> A\nA -> S"])
def test_gll_unproductive_calls_do_not_consume_graph_labels(grammar):
    graph = path_graph(["S", "A"])
    assert gll_based_cfpq(cfg_to_rsm(CFG.from_text(grammar)), graph) == set()


def test_gll_replays_completions_for_late_callers():
    graph = nx.DiGraph()
    graph.add_edge(0, 0, label="y")
    graph.add_edge(0, 1, label="z")
    graph.add_edge(0, 2, label="x")
    rsm = ebnf_to_rsm("S -> A x | y y y A z\nA -> $")
    assert gll_based_cfpq(rsm, graph, {0}) == {(0, 1), (0, 2)}


def test_gll_resumes_all_callers_of_a_shared_invocation():
    graph = path_graph("apx")
    graph.add_edge(1, 4, label="q")
    graph.add_edge(4, 5, label="y")
    rsm = ebnf_to_rsm("S -> A x | B y\nA -> C p\nB -> C q\nC -> a")
    assert gll_based_cfpq(rsm, graph, {0}) == {(0, 3), (0, 5)}


def test_gll_processes_outgoing_transitions_of_final_states():
    graph = path_graph("aaa")
    rsm = ebnf_to_rsm("S -> a*")
    assert gll_based_cfpq(rsm, graph) == {
        (source, target) for source in graph for target in graph if source <= target
    }


def test_gll_preserves_transitions_after_a_nullable_call_continuation():
    graph = path_graph("abb")
    rsm = ebnf_to_rsm("S -> A b*\nA -> a")
    assert gll_based_cfpq(rsm, graph) == {(0, 1), (0, 2), (0, 3)}


def test_gll_deduplicates_parallel_edges_and_ambiguous_derivations():
    graph = nx.MultiDiGraph()
    for source in range(8):
        for _ in range(10):
            graph.add_edge(source, (source + 1) % 8, label="a")
    rsm = ebnf_to_rsm("S -> S S | S S S | a | $")
    assert gll_based_cfpq(rsm, graph) == {
        (source, target) for source in graph for target in graph
    }


def test_gll_handles_custom_start_and_literal_terminal_names():
    root, other = Variable("Root"), Variable("A")
    cfg = CFG(
        start_symbol=root,
        productions={
            Production(root, [Terminal("Root"), other, Terminal("|")]),
            Production(other, [Terminal("a b"), Terminal("*")]),
        },
    )
    graph = path_graph(["Root", "a b", "*", "|"])
    assert gll_based_cfpq(cfg_to_rsm(cfg), graph) == {(0, 4)}


def test_gll_supports_noninteger_node_values():
    graph = nx.DiGraph()
    graph.add_edge("source", (1, 2), label="a")
    graph.add_edge((1, 2), "target", label="b")
    assert gll_based_cfpq(ebnf_to_rsm("S -> a b"), graph) == {("source", "target")}


def test_gll_handles_empty_inputs():
    assert gll_based_cfpq(ebnf_to_rsm("S -> $"), nx.DiGraph()) == set()
    assert gll_based_cfpq(RecursiveAutomaton(), path_graph("a")) == set()
    assert gll_based_cfpq(cfg_to_rsm(CFG()), path_graph("a")) == set()


def test_gll_handles_deep_nesting_without_python_recursion():
    depth = 1500
    graph = path_graph("a" * depth + "b" * depth)
    assert gll_based_cfpq(ebnf_to_rsm("S -> a S b | $"), graph, {0}, {2 * depth}) == {
        (0, 2 * depth)
    }


def test_gll_handles_long_right_recursion_without_caching_every_suffix():
    length = 10_000
    graph = path_graph("a" * length)
    rsm = cfg_to_rsm(CFG.from_text("S -> a S | $"))
    assert gll_based_cfpq(rsm, graph, {0}) == {
        (0, target) for target in range(length + 1)
    }


def test_gll_handles_mutually_recursive_tail_calls():
    graph = path_graph("ab" * 1000)
    rsm = cfg_to_rsm(CFG.from_text("S -> a A | $\nA -> b S"))
    assert gll_based_cfpq(rsm, graph, {0}) == {
        (0, target) for target in range(0, 2001, 2)
    }


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
def test_gll_matches_hellings_on_seeded_graphs(cfg_text, ebnf):
    cfg = CFG.from_text(cfg_text)
    rsms = [cfg_to_rsm(cfg), ebnf_to_rsm(ebnf)]
    random = Random(9)
    for _ in range(10):
        graph = nx.MultiDiGraph()
        nodes = [10, 30, 50, 70, 90, 110]
        graph.add_nodes_from(nodes)
        for source in nodes:
            for target in nodes:
                for label in "abcd":
                    if random.random() < 0.12:
                        graph.add_edge(source, target, label=label)
        for starts, finals in [(None, None), ({10, 30}, {70, 90})]:
            expected = hellings_based_cfpq(cfg, graph, starts, finals)
            for rsm in rsms:
                assert gll_based_cfpq(rsm, graph, starts, finals) == expected
