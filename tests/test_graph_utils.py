from project.graph_utils import get_graph_info, save_two_cycles_graph, GraphInfo
from filecmp import cmp
from os import remove
import pytest


def test_get_graph_info_existing_graph():
    actual = get_graph_info("bzip")
    expected = GraphInfo(556, 632, labels=("d", "a"))
    assert actual == expected


def test_get_graph_info_non_existent_graph():
    with pytest.raises(FileNotFoundError):
        get_graph_info("ocaml")


def test_save_two_cycles_graph():
    save_two_cycles_graph("actual.dot", 3, 4, common_node=1, labels=("a", "b"))
    cmp("actual.dot", "tests/resources/task1/expected.dot")
    remove("actual.dot")
