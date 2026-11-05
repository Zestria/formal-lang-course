from __future__ import annotations

from collections import deque
from typing import Optional

import networkx as nx
from pyformlang.cfg import CFG, Production, Terminal, Variable
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State, Symbol
from pyformlang.rsa import Box, RecursiveAutomaton
from scipy.sparse import csr_matrix

from project.automata import (
    AdjacencyMatrixFA,
    graph_to_nfa,
    intersect_automata,
    regex_to_dfa,
)


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    """Converts a context-free grammar to weak Chomsky normal form.

    The resulting grammar contains only the following kinds of productions:

    * ``A -> a`` where ``a`` is a terminal;
    * ``A -> B C`` where ``B`` and ``C`` are variables;
    * ``S -> epsilon`` for the start symbol ``S`` (only when the original
      grammar generates the empty word).
    """

    if cfg.start_symbol is None:
        return cfg

    generates_epsilon = cfg.generate_epsilon()
    start_symbol = cfg.start_symbol

    cnf = (
        cfg.remove_useless_symbols()
        .remove_epsilon()
        .eliminate_unit_productions()
        .remove_useless_symbols()
        .to_normal_form()
    )

    productions = set(cnf.productions)
    if generates_epsilon:
        productions.add(Production(start_symbol, []))

    return CFG(start_symbol=start_symbol, productions=productions)


def hellings_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: Optional[set[int]] = None,
    final_nodes: Optional[set[int]] = None,
) -> set[tuple[int, int]]:
    """Solves CFPQ on a labeled graph using Hellings' algorithm.

    The grammar is first converted to weak Chomsky normal form. Then the set
    of triples ``(A, u, v)`` is saturated, where a triple means that the
    variable ``A`` derives a path from ``u`` to ``v`` in the graph.
    """

    wnf = cfg_to_weak_normal_form(cfg)
    if wnf.start_symbol is None:
        return set()

    nodes = set(graph.nodes)

    epsilon_heads: set[Variable] = set()
    terminal_heads: dict[Terminal, set[Variable]] = {}
    binary_productions: list[tuple[Variable, Variable, Variable]] = []

    for production in wnf.productions:
        head = production.head
        body = production.body

        if len(body) == 0:
            epsilon_heads.add(head)
        elif len(body) == 1 and isinstance(body[0], Terminal):
            terminal_heads.setdefault(body[0], set()).add(head)
        elif len(body) == 2 and all(isinstance(symbol, Variable) for symbol in body):
            binary_productions.append((head, body[0], body[1]))

    # For a triple (B, u, v), these maps tell which binary productions can
    # use it as their left or right operand respectively.
    as_left: dict[Variable, list[tuple[Variable, Variable]]] = {}
    as_right: dict[Variable, list[tuple[Variable, Variable]]] = {}
    for head, left, right in binary_productions:
        as_left.setdefault(left, []).append((head, right))
        as_right.setdefault(right, []).append((head, left))

    relations: dict[Variable, set[tuple[int, int]]] = {}
    index_by_start: dict[Variable, dict[int, set[int]]] = {}
    index_by_end: dict[Variable, dict[int, set[int]]] = {}

    def add(variable: Variable, source: int, target: int) -> bool:
        pairs = relations.setdefault(variable, set())
        if (source, target) in pairs:
            return False

        pairs.add((source, target))
        index_by_start.setdefault(variable, {}).setdefault(source, set()).add(target)
        index_by_end.setdefault(variable, {}).setdefault(target, set()).add(source)
        return True

    queue: deque[tuple[Variable, int, int]] = deque()

    for head in epsilon_heads:
        for node in nodes:
            if add(head, node, node):
                queue.append((head, node, node))

    for source, target, data in graph.edges(data=True):
        label = data.get("label")
        if label is None:
            continue

        for head in terminal_heads.get(Terminal(label), ()):
            if add(head, source, target):
                queue.append((head, source, target))

    while queue:
        variable, source, target = queue.popleft()

        for head, right in as_left.get(variable, ()):
            for new_target in tuple(index_by_start.get(right, {}).get(target, ())):
                if add(head, source, new_target):
                    queue.append((head, source, new_target))

        for head, left in as_right.get(variable, ()):
            for new_source in tuple(index_by_end.get(left, {}).get(source, ())):
                if add(head, new_source, target):
                    queue.append((head, new_source, target))

    if not start_nodes:
        start_nodes = nodes
    if not final_nodes:
        final_nodes = nodes

    start_relation = relations.get(wnf.start_symbol, set())
    return {
        (source, target)
        for source in start_nodes
        for target in final_nodes
        if (source, target) in start_relation
    }


def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    """Builds one box per variable, accepting its production bodies.

    Each production becomes a separate NFA path; an empty body makes the
    entry state final. Box construction determinizes and minimizes the NFA.
    Variable transitions call other boxes, while terminals label graph edges.
    No normal-form conversion or regular-expression escaping is needed.
    """
    if cfg.start_symbol is None:
        return RecursiveAutomaton()

    variables = cfg.variables | {cfg.start_symbol}
    terminal_labels = {Symbol(terminal.value) for terminal in cfg.terminals}
    labels = {
        variable: (
            Symbol((Variable, variable.value))
            if Symbol(variable.value) in terminal_labels
            else Symbol(variable.value)
        )
        for variable in variables
    }
    automata = {}
    for variable in variables:
        automaton = NondeterministicFiniteAutomaton()
        automaton.add_start_state(0)
        automata[variable] = automaton

    for index, production in enumerate(cfg.productions):
        automaton = automata[production.head]
        source = State(0)
        for position, symbol in enumerate(production.body):
            target = State((index, position))
            label = (
                labels[symbol] if isinstance(symbol, Variable) else Symbol(symbol.value)
            )
            automaton.add_transition(source, label, target)
            source = target
        automaton.add_final_state(source)

    boxes = {Box(automata[variable], labels[variable]) for variable in variables}
    return RecursiveAutomaton(set(labels.values()), labels[cfg.start_symbol], boxes)


def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    """Builds an RSM from rules ``Head -> regex`` with start symbol ``S``.

    Bodies use pyformlang regex syntax: union ``|``, concatenation, Kleene
    star ``*`` and epsilon ``$``. Repeated heads are joined by union. Each
    body is converted by the regex-to-DFA function from task 2; symbols
    matching rule heads are recursive calls, all others are terminals.
    """
    bodies: dict[str, list[str]] = {}
    for line in ebnf.splitlines():
        line = line.strip()
        if not line:
            continue
        if "->" not in line:
            raise ValueError("Expected an EBNF rule 'Head -> regex'")
        head, body = line.split("->", 1)
        head = head.strip()
        if not head:
            raise ValueError("An EBNF rule must have a nonempty head")
        bodies.setdefault(head, []).append(f"({body.strip() or '$'})")

    if "S" not in bodies:
        raise ValueError("The EBNF grammar must define the start symbol S")
    boxes = {Box(regex_to_dfa(" | ".join(body)), head) for head, body in bodies.items()}
    return RecursiveAutomaton(set(bodies), "S", boxes)


def tensor_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: Optional[set[int]] = None,
    final_nodes: Optional[set[int]] = None,
) -> set[tuple[int, int]]:
    """Solves all-pairs CFPQ by saturating graph relations for RSM boxes.

    The disjoint union of box automata is intersected with the graph using
    sparse Boolean Kronecker products. Its reflexive transitive closure
    yields new nonterminal edges between box entries and exits. Iteration
    stops when no relation grows. The reflexive part handles nullable boxes.

    As in the previous query functions, None or an empty node filter means
    all graph vertices. Filters apply only to the resulting endpoint pairs.
    Neither the graph nor the RSM is modified.
    """
    if not graph or rsm.initial_label not in rsm.boxes:
        return set()

    # Qualifying states by their box prevents collisions between local IDs.
    union = NondeterministicFiniteAutomaton()
    for label, box in rsm.boxes.items():
        for state in box.dfa.states:
            union.states.add(State((label, state)))
        for source, transitions in box.dfa.to_dict().items():
            for symbol, target in transitions.items():
                union.add_transition((label, source), symbol, (label, target))
    rsm_fa = AdjacencyMatrixFA(union)
    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph))
    size = graph_fa.num_states
    empty = csr_matrix((size, size), dtype=bool)

    boundaries = {}
    for label, box in rsm.boxes.items():
        entries = {
            rsm_fa.state_to_index[State((label, state))]
            for state in box.dfa.start_states
        }
        exits = {
            rsm_fa.state_to_index[State((label, state))]
            for state in box.dfa.final_states
        }
        boundaries[label] = (entries, exits)
        # A box label denotes a call, not a terminal present in the graph.
        graph_fa.adjacency_matrices[label] = empty.copy()

    changed = True
    while changed:
        changed = False
        product = intersect_automata(rsm_fa, graph_fa)
        reachability = product._reachability_matrix()
        for label, (entries, exits) in boundaries.items():
            relation = graph_fa.adjacency_matrices[label]
            previous_size = relation.nnz
            for entry in entries:
                for exit_state in exits:
                    block = reachability[
                        entry * size : (entry + 1) * size,
                        exit_state * size : (exit_state + 1) * size,
                    ]
                    relation = relation.maximum(block)
            graph_fa.adjacency_matrices[label] = relation
            changed |= relation.nnz > previous_size

    sources = start_nodes or set(graph.nodes)
    targets = final_nodes or set(graph.nodes)
    rows, columns = graph_fa.adjacency_matrices[rsm.initial_label].nonzero()
    return {
        (source, target)
        for row, column in zip(rows, columns)
        if (source := graph_fa.index_to_state[row].value) in sources
        and (target := graph_fa.index_to_state[column].value) in targets
    }
