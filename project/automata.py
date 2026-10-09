from __future__ import annotations

from typing import Iterable, Set, Optional

from networkx import MultiDiGraph
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
    State,
    Symbol,
)
from pyformlang.regular_expression import Regex
from scipy.sparse import csr_matrix, kron


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    """Builds a minimal DFA equivalent to the given regular expression"""
    epsilon_nfa = Regex(regex).to_epsilon_nfa()
    dfa = epsilon_nfa.to_deterministic()
    return dfa.minimize()


def graph_to_nfa(
    graph: MultiDiGraph,
    start_states: Optional[Set[int]] = None,
    final_states: Optional[Set[int]] = None,
) -> NondeterministicFiniteAutomaton:
    """Builds an NFA from a labeled graph"""
    nfa = NondeterministicFiniteAutomaton()

    for node in graph.nodes:
        nfa.states.add(State(node))

    for u, v, data in graph.edges(data=True):
        label = data.get("label")
        if label is not None:
            nfa.add_transition(State(u), label, State(v))

    all_nodes = set(graph.nodes)
    start = start_states if start_states else all_nodes
    final = final_states if final_states else all_nodes

    for state in start:
        nfa.add_start_state(State(state))

    for state in final:
        nfa.add_final_state(State(state))

    return nfa


class AdjacencyMatrixFA:
    """A finite automaton represented via sparse Boolean adjacency matrices."""

    def __init__(self, automaton: NondeterministicFiniteAutomaton):
        self.states: list[State] = list(automaton.states)
        self.state_to_index: dict[State, int] = {
            state: index for index, state in enumerate(self.states)
        }
        self.index_to_state: dict[int, State] = {
            index: state for index, state in enumerate(self.states)
        }
        self.start_states: set[int] = {
            self.state_to_index[state] for state in automaton.start_states
        }
        self.final_states: set[int] = {
            self.state_to_index[state] for state in automaton.final_states
        }
        self.num_states: int = len(self.states)
        self.adjacency_matrices: dict[Symbol, csr_matrix] = (
            self._build_adjacency_matrices(automaton.to_dict())
        )

    def _build_adjacency_matrices(
        self, transitions: dict[State, dict[Symbol, State | set[State]]]
    ) -> dict[Symbol, csr_matrix]:
        """Builds one square Boolean matrix per transition symbol."""
        indices: dict[Symbol, tuple[list[int], list[int]]] = {}
        for source, edges in transitions.items():
            source_index = self.state_to_index[source]
            for symbol, targets in edges.items():
                if isinstance(targets, State):
                    targets = {targets}
                if not targets:
                    continue
                rows, cols = indices.setdefault(symbol, ([], []))
                for target in targets:
                    rows.append(source_index)
                    cols.append(self.state_to_index[target])

        return {
            symbol: csr_matrix(
                ([True] * len(rows), (rows, cols)),
                shape=(self.num_states, self.num_states),
                dtype=bool,
            )
            for symbol, (rows, cols) in indices.items()
        }

    @classmethod
    def _from_components(
        cls,
        states: list,
        start_states: set[int],
        final_states: set[int],
        adjacency_matrices: dict[Symbol, csr_matrix],
    ) -> AdjacencyMatrixFA:
        """Builds an instance directly from precomputed components."""
        instance = cls.__new__(cls)
        instance.states = list(states)
        instance.state_to_index = {
            state: index for index, state in enumerate(instance.states)
        }
        instance.index_to_state = {
            index: state for index, state in enumerate(instance.states)
        }
        instance.start_states = set(start_states)
        instance.final_states = set(final_states)
        instance.num_states = len(instance.states)
        instance.adjacency_matrices = dict(adjacency_matrices)
        return instance

    def _initial_states_vector(self) -> csr_matrix:
        """Builds a row vector marking the initial states."""
        columns = list(self.start_states)
        return csr_matrix(
            ([True] * len(columns), ([0] * len(columns), columns)),
            shape=(1, self.num_states),
            dtype=bool,
        )

    def _identity_matrix(self) -> csr_matrix:
        """Builds the identity matrix used for the reflexive closure."""
        indices = list(range(self.num_states))
        return csr_matrix(
            ([True] * self.num_states, (indices, indices)),
            shape=(self.num_states, self.num_states),
            dtype=bool,
        )

    def _symbol_union_matrix(self) -> csr_matrix:
        """Builds a matrix marking every transition, regardless of symbol."""
        if not self.adjacency_matrices:
            return csr_matrix((self.num_states, self.num_states), dtype=bool)

        total = None
        for matrix in self.adjacency_matrices.values():
            total = matrix if total is None else total.maximum(matrix)
        return total.astype(bool)

    def _reachability_matrix(self) -> csr_matrix:
        """Reflexive transitive closure of the per-symbol adjacency matrices."""
        if self.num_states == 0:
            return csr_matrix((0, 0), dtype=bool)

        reach = self._symbol_union_matrix().maximum(self._identity_matrix())

        length = 1
        while length < self.num_states:
            reach = (reach @ reach).astype(bool)
            length *= 2
        return reach

    def accepts(self, word: Iterable[Symbol]) -> bool:
        """Checks whether the automaton accepts the given word of symbols."""
        current = self._initial_states_vector()

        for symbol in word:
            symbol = symbol if isinstance(symbol, Symbol) else Symbol(symbol)
            matrix = self.adjacency_matrices.get(symbol)
            if matrix is None:
                return False
            current = (current @ matrix).astype(bool)
            if current.nnz == 0:
                return False

        reached = set(current.nonzero()[1])
        return not reached.isdisjoint(self.final_states)

    def is_empty(self) -> bool:
        """Checks whether the language defined by the automaton is empty."""
        reachability = self._reachability_matrix()
        for start in self.start_states:
            for final in self.final_states:
                if reachability[start, final]:
                    return False
        return True

    def __repr__(self) -> str:
        symbols = sorted(str(symbol) for symbol in self.adjacency_matrices)
        return (
            f"{type(self).__name__}(states={self.num_states}, "
            f"symbols={symbols}, "
            f"start_states={self.start_states}, "
            f"final_states={self.final_states})"
        )


def intersect_automata(
    automaton1: AdjacencyMatrixFA,
    automaton2: AdjacencyMatrixFA,
) -> AdjacencyMatrixFA:
    """Builds the automaton recognising the intersection of two languages."""
    states = [
        (state1, state2) for state1 in automaton1.states for state2 in automaton2.states
    ]

    num_states_second = automaton2.num_states
    start_states = {
        index1 * num_states_second + index2
        for index1 in automaton1.start_states
        for index2 in automaton2.start_states
    }
    final_states = {
        index1 * num_states_second + index2
        for index1 in automaton1.final_states
        for index2 in automaton2.final_states
    }

    common_symbols = (
        automaton1.adjacency_matrices.keys() & automaton2.adjacency_matrices.keys()
    )
    adjacency_matrices = {
        symbol: kron(
            automaton1.adjacency_matrices[symbol],
            automaton2.adjacency_matrices[symbol],
            format="csr",
        )
        for symbol in common_symbols
    }

    return AdjacencyMatrixFA._from_components(
        states, start_states, final_states, adjacency_matrices
    )


def tensor_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
) -> set[tuple[int, int]]:
    """Executes a regular path query on a labeled graph."""

    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    regex_fa = AdjacencyMatrixFA(regex_to_dfa(regex))

    product = intersect_automata(graph_fa, regex_fa)
    reachability = product._reachability_matrix()

    regex_size = regex_fa.num_states
    result: set[tuple[int, int]] = set()
    for start in product.start_states:
        for final in product.final_states:
            if reachability[start, final]:
                graph_start = graph_fa.index_to_state[start // regex_size].value
                graph_final = graph_fa.index_to_state[final // regex_size].value
                result.add((graph_start, graph_final))
    return result
