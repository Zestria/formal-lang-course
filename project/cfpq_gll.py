from __future__ import annotations

from collections import deque
from typing import Optional

import networkx as nx
from pyformlang.finite_automaton import Symbol
from pyformlang.rsa import RecursiveAutomaton

from project.automata import graph_to_nfa


def gll_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: Optional[set[int]] = None,
    final_nodes: Optional[set[int]] = None,
) -> set[tuple[int, int]]:
    """Executes an RSM on a graph using GLL with a shared graph-structured stack."""

    nodes = list(graph.nodes)
    size = len(nodes)
    if not size or rsm.initial_label not in rsm.boxes:
        return set()

    node_index = {node: index for index, node in enumerate(nodes)}
    sources = (
        {node_index[node] for node in start_nodes if node in node_index}
        if start_nodes
        else set(range(size))
    )
    targets = (
        {node_index[node] for node in final_nodes if node in node_index}
        if final_nodes
        else set(range(size))
    )
    if not sources or not targets:
        return set()

    labels = list(rsm.boxes)
    box_index = {label: index for index, label in enumerate(labels)}
    state_index = {
        (label, state): index
        for index, (label, state) in enumerate(
            (label, state)
            for label, box in rsm.boxes.items()
            for state in box.dfa.states
        )
    }
    entries: list[list[int]] = []
    is_final = [False] * len(state_index)
    calls: list[list[tuple[int, int]]] = [[] for _ in state_index]
    terminals: list[list[tuple[Symbol, int]]] = [[] for _ in state_index]
    terminal_labels: set[Symbol] = set()
    for label, box in rsm.boxes.items():
        entries.append([state_index[label, state] for state in box.dfa.start_states])
        for state in box.dfa.final_states:
            is_final[state_index[label, state]] = True
        for source, transitions in box.dfa.to_dict().items():
            state = state_index[label, source]
            for symbol, target in transitions.items():
                next_state = state_index[label, target]
                if symbol in box_index:
                    calls[state].append((box_index[symbol], next_state))
                else:
                    terminals[state].append((symbol, next_state))
                    terminal_labels.add(symbol)

    has_outgoing = [
        bool(calls[state] or terminals[state]) for state in state_index.values()
    ]
    tail_calls: list[list[int]] = [[] for _ in state_index]
    for state, transitions in enumerate(calls):
        calls[state] = []
        for box, return_state in transitions:
            if is_final[return_state] and not has_outgoing[return_state]:
                tail_calls[state].append(box)
            else:
                calls[state].append((box, return_state))

    edges: list[dict[Symbol, list[int]]] = [{} for _ in nodes]
    for source, transitions in graph_to_nfa(graph).to_dict().items():
        edges[node_index[source.value]] = {
            symbol: [node_index[target.value] for target in destinations]
            for symbol, destinations in transitions.items()
            if symbol in terminal_labels
        }

    if len(targets) < size:
        incoming: list[list[int]] = [[] for _ in nodes]
        for source, transitions in enumerate(edges):
            for destinations in transitions.values():
                for target in destinations:
                    incoming[target].append(source)
        useful = set(targets)
        pending = list(targets)
        while pending:
            for source in incoming[pending.pop()]:
                if source not in useful:
                    useful.add(source)
                    pending.append(source)
        sources.intersection_update(useful)
        if not sources:
            return set()
        edges = [
            {
                symbol: [target for target in destinations if target in useful]
                for symbol, destinations in transitions.items()
            }
            if source in useful
            else {}
            for source, transitions in enumerate(edges)
        ]

    queue: deque[tuple[int, int, int]] = deque()
    invocations: dict[int, int] = {}
    continuations: list[set[tuple[int, int]]] = []
    completed: list[set[int]] = []
    seen: list[set[int]] = []

    def add(state: int, stack: int, vertex: int) -> None:
        descriptor = state * size + vertex
        if descriptor not in seen[stack]:
            seen[stack].add(descriptor)
            queue.append((state, stack, vertex))

    def create(box: int, vertex: int) -> int:
        key = box * size + vertex
        stack = invocations.get(key)
        if stack is None:
            stack = len(continuations)
            invocations[key] = stack
            continuations.append(set())
            completed.append(set())
            seen.append(set())
            for state in entries[box]:
                add(state, stack, vertex)
        return stack

    start_box = box_index[rsm.initial_label]
    roots = {source: create(start_box, source) for source in sources}
    while queue:
        state, stack, vertex = queue.popleft()

        if is_final[state] and vertex not in completed[stack]:
            completed[stack].add(vertex)
            for return_state, caller in continuations[stack]:
                add(return_state, caller, vertex)

        for symbol, next_state in terminals[state]:
            for target in edges[vertex].get(symbol, ()):
                add(next_state, stack, target)

        for box in tail_calls[state]:
            for entry in entries[box]:
                add(entry, stack, vertex)

        for box, return_state in calls[state]:
            callee = create(box, vertex)
            continuation = (return_state, stack)
            if continuation not in continuations[callee]:
                continuations[callee].add(continuation)
                for target in completed[callee]:
                    add(return_state, stack, target)

    return {
        (nodes[source], nodes[target])
        for source, stack in roots.items()
        for target in completed[stack] & targets
    }
