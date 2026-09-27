import networkx as nx
import pytest
from pyformlang.finite_automaton import DeterministicFiniteAutomaton

from project.automata import graph_to_nfa, regex_to_dfa


def test_returns_dfa():
    dfa = regex_to_dfa("a b*")
    assert isinstance(dfa, DeterministicFiniteAutomaton)
    assert dfa.is_deterministic()


@pytest.mark.parametrize(
    "regex,accepted,rejected",
    [
        ("a b*", ["a", "ab", "abb", "abbb"], ["", "b", "ba", "aab"]),
        ("a*", ["", "a", "aa", "aaa"], ["b", "ab"]),
        ("a|b", ["a", "b"], ["", "ab", "ba", "c"]),
        ("(a b)*", ["", "ab", "abab", "ababab"], ["a", "b", "aba"]),
    ],
)
def test_accepts_and_rejects_expected_words(regex, accepted, rejected):
    dfa = regex_to_dfa(regex)
    for word in accepted:
        assert dfa.accepts(list(word)), f"{regex!r} should accept {word!r}"
    for word in rejected:
        assert not dfa.accepts(list(word)), f"{regex!r} should reject {word!r}"


def test_result_is_minimal():
    dfa = regex_to_dfa("a b*")
    assert dfa.is_equivalent_to(dfa.minimize())
    assert len(dfa.states) == len(dfa.minimize().states)


def _build_graph():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="b")
    graph.add_edge(2, 0, label="c")
    return graph


def test_default_start_and_final_states():
    graph = _build_graph()
    nfa = graph_to_nfa(graph, set(), set())

    assert nfa.start_states == {0, 1, 2}
    assert nfa.final_states == {0, 1, 2}
    assert nfa.accepts(["a"])
    assert nfa.accepts(["b"])
    assert nfa.accepts(["c"])
    assert nfa.accepts(["a", "b", "c"])


def test_explicit_start_and_final_states():
    graph = _build_graph()
    nfa = graph_to_nfa(graph, start_states={0}, final_states={2})

    assert nfa.start_states == {0}
    assert nfa.final_states == {2}
    assert nfa.accepts(["a", "b"])
    assert not nfa.accepts(["a"])
    assert not nfa.accepts(["b", "c"])


def test_graph_with_multiple_edges_between_nodes():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(0, 1, label="b")

    nfa = graph_to_nfa(graph, start_states={0}, final_states={1})

    assert nfa.accepts(["a"])
    assert nfa.accepts(["b"])
    assert not nfa.accepts(["c"])


def test_isolated_node_without_edges():
    graph = nx.MultiDiGraph()
    graph.add_node(0)

    nfa = graph_to_nfa(graph)

    assert nfa.start_states == {0}
    assert nfa.final_states == {0}
    assert nfa.accepts([])
