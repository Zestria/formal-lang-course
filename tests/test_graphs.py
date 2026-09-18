import pytest
import pydot

from pathlib import Path
from unittest.mock import MagicMock, patch

from project.graphs import (
    get_graph_info_from_cfpq_data,
    create_and_save_two_cycles_graph
)


def test_get_graph_info_isolated() -> None:
    """Unit test with mocked cfpq_data calls (isolated from network and filesystem)."""
    with patch("project.graphs.cfpq_data") as mock_cfpq_data:
        # Arrange
        mock_graph = MagicMock()
        mock_graph.number_of_nodes.return_value = 10
        mock_graph.number_of_edges.return_value = 25

        mock_cfpq_data.download.return_value = "/fake/path/graph.csv"
        mock_cfpq_data.graph_from_csv.return_value = mock_graph
        mock_cfpq_data.get_sorted_labels.return_value = ["a", "b", "a"]

        # Act
        nodes, edges, labels = get_graph_info_from_cfpq_data("test_graph")

        # Assert
        mock_cfpq_data.download.assert_called_once_with("test_graph")
        mock_cfpq_data.graph_from_csv.assert_called_once_with("/fake/path/graph.csv")

        assert nodes == 10
        assert edges == 25
        assert labels == {"a", "b"}


@pytest.mark.integration
def test_get_graph_info_real_download() -> None:
    """Integration test using a real CFPQ_Data dataset ('generations')."""
    nodes, edges, labels = get_graph_info_from_cfpq_data("generations")

    assert nodes > 0
    assert edges > 0
    assert isinstance(labels, set)


@pytest.mark.parametrize(
    ("n", "m", "labels"),
    [
        (3, 4, ("a", "b")),
        (1, 1, ("x", "y")),
        (10, 5, ("label_1", "label_2")),
    ],
)
def test_create_and_save_two_cycles_graph_creates_valid_dot(
    tmp_path: Path, n: int, m: int, labels: tuple[str, str]
) -> None:
    """Tests generation of a valid DOT file, including nested directory creation."""
    # Arrange
    output_file = tmp_path / "nested_dir" / "output_graph.dot"

    # Act
    create_and_save_two_cycles_graph(
        first_cycle_nodes=n,
        second_cycle_nodes=m,
        labels=labels,
        output_path=output_file,
    )

    # Assert
    assert output_file.exists()
    assert output_file.stat().st_size > 0

    # Validate DOT format correctness using pydot
    parsed_pydot = pydot.graph_from_dot_file(str(output_file))
    assert parsed_pydot is not None
    assert len(parsed_pydot) > 0

    # Verify that the specified labels are present in the output file
    dot_content = output_file.read_text()
    assert labels[0] in dot_content
    assert labels[1] in dot_content


def test_create_and_save_accepts_path_object_and_string(tmp_path: Path) -> None:
    """Verifies support for both str and Path type annotations."""
    str_path = str(tmp_path / "str_graph.dot")
    path_obj = tmp_path / "path_graph.dot"

    create_and_save_two_cycles_graph(2, 2, ("a", "b"), str_path)
    create_and_save_two_cycles_graph(2, 2, ("a", "b"), path_obj)

    assert Path(str_path).exists()
    assert path_obj.exists()
