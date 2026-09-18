from pathlib import Path
from typing import NamedTuple

import cfpq_data
import networkx as nx


class GraphInfo(NamedTuple):
    nodes_num: int
    edges_num: int
    labels: set[str]


def get_graph_info_from_cfpq_data(name: str) -> GraphInfo:
    """Downloads a graph from the CFPQ_Data dataset by name and extracts its core metrics."""
    path = cfpq_data.download(name)
    graph = cfpq_data.graph_from_csv(path)

    return GraphInfo(
        nodes_num=graph.number_of_nodes(),
        edges_num=graph.number_of_edges(),
        labels=set(cfpq_data.get_sorted_labels(graph)),
    )


def create_and_save_two_cycles_graph(
    first_cycle_nodes: int,
    second_cycle_nodes: int,
    labels: tuple[str, str],
    output_path: str | Path,
) -> None:
    """Generates a two-cycles graph with given parameters and saves it to a DOT file."""
    graph = cfpq_data.labeled_two_cycles_graph(
        n=first_cycle_nodes, m=second_cycle_nodes, labels=labels
    )

    file_path = Path(output_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    dot_graph = nx.drawing.nx_pydot.to_pydot(graph)
    dot_graph.write_raw(str(file_path))
