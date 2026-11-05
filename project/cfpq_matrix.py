from __future__ import annotations

from typing import Optional

import networkx as nx
from pyformlang.cfg import CFG, Terminal, Variable
from scipy.sparse import csr_matrix, eye

from project.cfpq import cfg_to_weak_normal_form


def matrix_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: Optional[set[int]] = None,
    final_nodes: Optional[set[int]] = None,
) -> set[tuple[int, int]]:
    """Solves CFPQ on a labeled graph using the matrix-based algorithm."""

    wnf = cfg_to_weak_normal_form(cfg)
    if wnf.start_symbol is None:
        return set()

    nodes = list(graph.nodes)
    node_to_index = {node: index for index, node in enumerate(nodes)}
    num_nodes = len(nodes)

    epsilon_heads: set[Variable] = set()
    terminal_heads: dict[Terminal, set[Variable]] = {}
    binary_productions: list[tuple[Variable, Variable, Variable]] = []
    unit_productions: list[tuple[Variable, Variable]] = []
    variables: set[Variable] = set()

    for production in wnf.productions:
        head = production.head
        body = production.body
        variables.add(head)

        if len(body) == 0:
            epsilon_heads.add(head)
        elif len(body) == 1 and isinstance(body[0], Terminal):
            terminal_heads.setdefault(body[0], set()).add(head)
        elif len(body) == 2 and all(isinstance(symbol, Variable) for symbol in body):
            binary_productions.append((head, body[0], body[1]))
            variables.update(body)
        elif len(body) == 1 and isinstance(body[0], Variable):
            # Weak Chomsky normal form has no unit productions, but handling
            # them keeps the function correct for equivalent grammars too.
            unit_productions.append((head, body[0]))
            variables.add(body[0])

    def zero_matrix() -> csr_matrix:
        return csr_matrix((num_nodes, num_nodes), dtype=bool)

    matrices: dict[Variable, csr_matrix] = {
        variable: zero_matrix() for variable in variables
    }

    if epsilon_heads and num_nodes:
        reflexive = eye(num_nodes, dtype=bool, format="csr")
        for head in epsilon_heads:
            matrices[head] = matrices[head].maximum(reflexive)

    if terminal_heads and num_nodes:
        terminal_edges: dict[Terminal, tuple[list[int], list[int]]] = {}
        for source, target, data in graph.edges(data=True):
            label = data.get("label")
            if label is None:
                continue

            terminal = Terminal(label)
            if terminal not in terminal_heads:
                continue

            rows, columns = terminal_edges.setdefault(terminal, ([], []))
            rows.append(node_to_index[source])
            columns.append(node_to_index[target])

        for terminal, (rows, columns) in terminal_edges.items():
            terminal_matrix = csr_matrix(
                ([True] * len(rows), (rows, columns)),
                shape=(num_nodes, num_nodes),
                dtype=bool,
            )
            for head in terminal_heads[terminal]:
                matrices[head] = matrices[head].maximum(terminal_matrix)

    changed = True
    while changed:
        changed = False

        for head, left, right in binary_productions:
            left_matrix = matrices[left]
            right_matrix = matrices[right]
            if left_matrix.nnz == 0 or right_matrix.nnz == 0:
                continue

            product = (left_matrix @ right_matrix).astype(bool)
            if product.nnz == 0:
                continue

            merged = matrices[head].maximum(product)
            if merged.nnz != matrices[head].nnz:
                matrices[head] = merged
                changed = True

        for head, body_variable in unit_productions:
            merged = matrices[head].maximum(matrices[body_variable])
            if merged.nnz != matrices[head].nnz:
                matrices[head] = merged
                changed = True

    all_nodes = set(nodes)
    start_nodes = set(start_nodes) if start_nodes else all_nodes
    final_nodes = set(final_nodes) if final_nodes else all_nodes

    start_matrix = matrices.get(wnf.start_symbol)
    if start_matrix is None:
        return set()

    rows, columns = start_matrix.nonzero()
    return {
        (nodes[row], nodes[column])
        for row, column in zip(rows, columns)
        if nodes[row] in start_nodes and nodes[column] in final_nodes
    }
