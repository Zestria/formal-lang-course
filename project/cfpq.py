from __future__ import annotations

from collections import deque
from typing import Optional

import networkx as nx
from pyformlang.cfg import CFG, Production, Terminal, Variable


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
