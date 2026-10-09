import random

import networkx as nx
import pytest
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
    State,
    Symbol,
)
from scipy.sparse import kron

from project.automata import (
    AdjacencyMatrixFA,
    graph_to_nfa,
    intersect_automata,
    regex_to_dfa,
    tensor_based_rpq,
    ms_bfs_based_rpq,
)


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


def _small_nfa() -> NondeterministicFiniteAutomaton:
    nfa = NondeterministicFiniteAutomaton()
    nfa.add_transition(State(0), "a", State(1))
    nfa.add_transition(State(0), "b", State(1))
    nfa.add_transition(State(1), "a", State(2))
    nfa.add_start_state(State(0))
    nfa.add_final_state(State(2))
    return nfa


def _random_nfa(rng, labels) -> NondeterministicFiniteAutomaton:
    size = rng.randint(1, 4)
    nfa = NondeterministicFiniteAutomaton()
    for state in range(size):
        for label in labels:
            if rng.random() < 0.5:
                nfa.add_transition(State(state), label, State(rng.randrange(size)))
    for state in range(size):
        if rng.random() < 0.5:
            nfa.add_start_state(State(state))
        if rng.random() < 0.5:
            nfa.add_final_state(State(state))
    return nfa


def test_adjacency_matrix_fa_stores_states_and_marks():
    automaton = AdjacencyMatrixFA(_small_nfa())
    index = automaton.state_to_index

    assert automaton.num_states == 3
    assert set(automaton.states) == {State(0), State(1), State(2)}
    assert automaton.start_states == {index[State(0)]}
    assert automaton.final_states == {index[State(2)]}


def test_adjacency_matrix_fa_builds_one_boolean_matrix_per_symbol():
    automaton = AdjacencyMatrixFA(_small_nfa())

    assert set(automaton.adjacency_matrices) == {Symbol("a"), Symbol("b")}
    for matrix in automaton.adjacency_matrices.values():
        assert matrix.shape == (automaton.num_states, automaton.num_states)
        assert matrix.dtype == bool


def test_adjacency_matrix_fa_encodes_transitions():
    automaton = AdjacencyMatrixFA(_small_nfa())
    index = automaton.state_to_index

    a_matrix = automaton.adjacency_matrices[Symbol("a")]
    assert a_matrix[index[State(0)], index[State(1)]]
    assert a_matrix[index[State(1)], index[State(2)]]
    assert not a_matrix[index[State(0)], index[State(0)]]

    b_matrix = automaton.adjacency_matrices[Symbol("b")]
    assert b_matrix[index[State(0)], index[State(1)]]
    assert not b_matrix[index[State(1)], index[State(2)]]


def test_adjacency_matrix_fa_roundtrips_index_mapping():
    automaton = AdjacencyMatrixFA(_small_nfa())

    for state, position in automaton.state_to_index.items():
        assert automaton.index_to_state[position] == state


def test_adjacency_matrix_fa_accepts_deterministic_automaton():
    dfa = regex_to_dfa("a b*")
    automaton = AdjacencyMatrixFA(dfa)

    assert automaton.num_states == len(dfa.states)
    assert len(automaton.start_states) == 1
    assert automaton.final_states
    assert set(automaton.adjacency_matrices) == {Symbol("a"), Symbol("b")}
    for matrix in automaton.adjacency_matrices.values():
        assert matrix.shape == (automaton.num_states, automaton.num_states)


def test_adjacency_matrix_fa_accepts_nfa_from_graph():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="b")

    automaton = AdjacencyMatrixFA(graph_to_nfa(graph, {0}, {2}))

    assert automaton.num_states == 3
    assert set(automaton.adjacency_matrices) == {Symbol("a"), Symbol("b")}


def test_adjacency_matrix_fa_of_empty_automaton():
    automaton = AdjacencyMatrixFA(NondeterministicFiniteAutomaton())

    assert automaton.num_states == 0
    assert automaton.states == []
    assert automaton.start_states == set()
    assert automaton.final_states == set()
    assert automaton.adjacency_matrices == {}


def test_accepts_accepts_valid_words():
    automaton = AdjacencyMatrixFA(_small_nfa())

    assert automaton.accepts([Symbol("a"), Symbol("a")])
    assert automaton.accepts(["a", "a"])


def test_accepts_rejects_invalid_words():
    automaton = AdjacencyMatrixFA(_small_nfa())

    assert not automaton.accepts([])
    assert not automaton.accepts(["a"])
    assert not automaton.accepts(["b"])
    assert not automaton.accepts(["a", "b"])
    assert not automaton.accepts(["c"])


@pytest.mark.parametrize(
    "word",
    [[], ["a"], ["a", "b"], ["a", "b", "b"], ["b"], ["b", "a"], ["a", "a"]],
)
def test_accepts_matches_reference_dfa(word):
    dfa = regex_to_dfa("a b*")
    automaton = AdjacencyMatrixFA(dfa)

    assert automaton.accepts(word) == dfa.accepts(word)


def test_is_empty_false_when_language_non_empty():
    automaton = AdjacencyMatrixFA(_small_nfa())

    assert not automaton.is_empty()


def test_is_empty_true_when_no_final_states():
    automaton = AdjacencyMatrixFA(_small_nfa())
    automaton.final_states = set()

    assert automaton.is_empty()


def test_is_empty_true_when_final_state_unreachable():
    nfa = NondeterministicFiniteAutomaton()
    nfa.add_transition(State(1), "a", State(0))
    nfa.add_start_state(State(0))
    nfa.add_final_state(State(1))

    automaton = AdjacencyMatrixFA(nfa)

    assert automaton.num_states == 2
    assert automaton.is_empty()


def test_is_empty_false_when_empty_word_accepted():
    nfa = NondeterministicFiniteAutomaton()
    nfa.add_transition(State(0), "a", State(1))
    nfa.add_start_state(State(0))
    nfa.add_final_state(State(0))

    automaton = AdjacencyMatrixFA(nfa)

    assert not automaton.is_empty()
    assert automaton.accepts([])


def test_empty_automaton_is_empty_and_rejects_everything():
    automaton = AdjacencyMatrixFA(NondeterministicFiniteAutomaton())

    assert automaton.is_empty()
    assert not automaton.accepts([])
    assert not automaton.accepts(["a"])


def test_accepts_and_is_empty_match_pyformlang_reference():
    random.seed(0)
    labels = ["a", "b"]

    for _ in range(50):
        size = random.randint(1, 5)
        nfa = NondeterministicFiniteAutomaton()
        for state in range(size):
            for label in labels:
                if random.random() < 0.5:
                    nfa.add_transition(
                        State(state), label, State(random.randrange(size))
                    )
        for state in range(size):
            if random.random() < 0.5:
                nfa.add_start_state(State(state))
            if random.random() < 0.5:
                nfa.add_final_state(State(state))

        if not nfa.start_states or not nfa.final_states:
            continue

        automaton = AdjacencyMatrixFA(nfa)
        assert automaton.is_empty() == nfa.is_empty()

        for _ in range(3):
            word = [random.choice(labels) for _ in range(random.randint(0, 5))]
            assert automaton.accepts(word) == nfa.accepts(word)


def test_intersect_automata_state_count_is_product():
    automaton1 = AdjacencyMatrixFA(regex_to_dfa("a b*"))
    automaton2 = AdjacencyMatrixFA(regex_to_dfa("a* b"))

    product = intersect_automata(automaton1, automaton2)

    assert product.num_states == automaton1.num_states * automaton2.num_states
    assert len(product.start_states) == 1
    assert product.final_states


def test_intersect_automata_matrices_are_kronecker_products():
    automaton1 = AdjacencyMatrixFA(regex_to_dfa("a b*"))
    automaton2 = AdjacencyMatrixFA(regex_to_dfa("a* b"))

    product = intersect_automata(automaton1, automaton2)

    assert set(product.adjacency_matrices) == {Symbol("a"), Symbol("b")}
    for symbol, matrix in product.adjacency_matrices.items():
        assert matrix.shape == (product.num_states, product.num_states)
        assert matrix.dtype == bool
        expected = kron(
            automaton1.adjacency_matrices[symbol],
            automaton2.adjacency_matrices[symbol],
            format="csr",
        ).astype(bool)
        assert (matrix != expected).nnz == 0


@pytest.mark.parametrize(
    "word",
    [
        [],
        ["a"],
        ["b"],
        ["a", "b"],
        ["a", "a", "b"],
        ["a", "b", "b"],
        ["b", "a"],
        ["a", "a", "b", "b"],
        ["c"],
    ],
)
def test_intersect_automata_accepts_intersection_language(word):
    regex1 = "a* b*"
    regex2 = "a b*"
    automaton1 = AdjacencyMatrixFA(regex_to_dfa(regex1))
    automaton2 = AdjacencyMatrixFA(regex_to_dfa(regex2))

    product = intersect_automata(automaton1, automaton2)
    dfa1 = regex_to_dfa(regex1)
    dfa2 = regex_to_dfa(regex2)
    expected = dfa1.accepts(word) and dfa2.accepts(word)

    assert product.accepts(word) == expected


def test_intersect_automata_with_disjoint_alphabets_is_empty():
    automaton1 = AdjacencyMatrixFA(regex_to_dfa("a"))
    automaton2 = AdjacencyMatrixFA(regex_to_dfa("b"))

    product = intersect_automata(automaton1, automaton2)

    assert product.adjacency_matrices == {}
    assert product.is_empty()


def test_intersect_automata_is_commutative_on_language():
    automaton1 = AdjacencyMatrixFA(regex_to_dfa("a b*"))
    automaton2 = AdjacencyMatrixFA(regex_to_dfa("a* b"))

    forward = intersect_automata(automaton1, automaton2)
    backward = intersect_automata(automaton2, automaton1)

    for word in [[], ["a"], ["b"], ["a", "b"], ["a", "a", "b"], ["b", "a"]]:
        assert forward.accepts(word) == backward.accepts(word)


def test_intersect_automata_with_self_is_equivalent():
    automaton = AdjacencyMatrixFA(regex_to_dfa("a b*"))

    product = intersect_automata(automaton, automaton)

    for word in [[], ["a"], ["ab"], ["abb"], ["b"], ["ba"]]:
        assert product.accepts(word) == automaton.accepts(word)


def test_intersect_automata_matches_pyformlang_reference():
    random.seed(1)
    labels = ["a", "b"]

    for _ in range(30):
        nfa1 = _random_nfa(random, labels)
        nfa2 = _random_nfa(random, labels)
        if not nfa1.start_states or not nfa1.final_states:
            continue
        if not nfa2.start_states or not nfa2.final_states:
            continue

        product = intersect_automata(AdjacencyMatrixFA(nfa1), AdjacencyMatrixFA(nfa2))
        reference = nfa1.get_intersection(nfa2)

        assert product.is_empty() == reference.is_empty()
        for _ in range(3):
            word = [random.choice(labels) for _ in range(random.randint(0, 5))]
            assert product.accepts(word) == reference.accepts(word)


def test_tensor_based_rpq_filters_by_start_final_and_regex():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="b")
    graph.add_edge(2, 3, label="b")

    result = tensor_based_rpq("a b*", graph, {0, 1}, {1, 2, 3})

    # Node 1 has no outgoing "a", so only paths from 0 qualify.
    assert result == {(0, 1), (0, 2), (0, 3)}


def test_tensor_based_rpq_handles_cycles_and_empty_word():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 0, label="a")
    graph.add_node(2)

    result = tensor_based_rpq("a*", graph, {0, 2}, {0, 1, 2})

    # 0 -> 0 (empty word or "aa"), 0 -> 1 ("a"), 2 -> 2 (empty word only).
    assert result == {(0, 0), (0, 1), (2, 2)}


def test_ms_bfs_based_rpq_filters_by_start_final_and_regex():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="b")
    graph.add_edge(2, 3, label="b")

    result = ms_bfs_based_rpq("a b*", graph, {0, 1}, {1, 2, 3})

    assert result == {(0, 1), (0, 2), (0, 3)}


def test_ms_bfs_based_rpq_handles_cycles_and_empty_word():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 0, label="a")
    graph.add_node(2)

    result = ms_bfs_based_rpq("a*", graph, {0, 2}, {0, 1, 2})

    assert result == {(0, 0), (0, 1), (2, 2)}


def test_ms_bfs_based_rpq_single_start_node():
    graph = _build_graph()

    assert ms_bfs_based_rpq("a b", graph, {0}, {2}) == {(0, 2)}
    assert ms_bfs_based_rpq("a b", graph, {1}, {2}) == set()


def test_ms_bfs_based_rpq_without_common_symbols():
    graph = _build_graph()

    assert ms_bfs_based_rpq("x", graph, {0, 1}, {0, 1, 2}) == set()


def test_ms_bfs_based_rpq_empty_sets_mean_all_nodes():
    graph = _build_graph()

    assert ms_bfs_based_rpq("a|b|c", graph, set(), set()) == {
        (0, 1),
        (1, 2),
        (2, 0),
    }


def test_ms_bfs_based_rpq_matches_tensor_based_rpq():
    rng = random.Random(2)
    labels = ["a", "b", "c"]
    regexes = ["a b*", "(a|b)* c", "a* b* c*", "(a b)*", "c a*"]

    for _ in range(30):
        size = rng.randint(1, 7)
        graph = nx.MultiDiGraph()
        graph.add_nodes_from(range(size))
        for _ in range(rng.randint(0, 15)):
            graph.add_edge(
                rng.randrange(size), rng.randrange(size), label=rng.choice(labels)
            )
        start = {n for n in range(size) if rng.random() < 0.5}
        final = {n for n in range(size) if rng.random() < 0.5}

        for regex in regexes:
            assert ms_bfs_based_rpq(regex, graph, start, final) == tensor_based_rpq(
                regex, graph, start, final
            )
