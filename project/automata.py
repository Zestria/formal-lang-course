from typing import Set, Optional

from networkx import MultiDiGraph
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
    State,
)
from pyformlang.regular_expression import Regex


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
