import cfpq_data
from typing import Tuple, Any
from networkx.drawing.nx_pydot import write_dot


class GraphInfo:
    def __init__(
        self, number_of_edges: int, number_of_nodes: int, labels: Tuple[Any] = ()
    ):
        self.number_of_edges = number_of_edges
        self.number_of_nodes = number_of_nodes
        self.labels = labels

    def __eq__(self, other):
        return (
            self.number_of_edges == other.number_of_edges
            and self.number_of_nodes == other.number_of_nodes
            and self.labels == other.labels
        )

    @classmethod
    def from_file(cls, name: str):
        g = cfpq_data.download(name)
        graph = cfpq_data.graph_from_csv(g)
        return cls(
            graph.number_of_edges(),
            graph.number_of_nodes(),
            labels=tuple(cfpq_data.get_sorted_labels(graph)),
        )


def get_graph_info(name: str):
    return GraphInfo.from_file(name)


def save_two_cycles_graph(
    name, fst_cycle_node_count, snd_cycle_node_count, common_node=0, labels=()
):
    graph = cfpq_data.labeled_two_cycles_graph(
        fst_cycle_node_count,
        snd_cycle_node_count,
        common_node=common_node,
        labels=labels,
    )
    write_dot(graph, name)
