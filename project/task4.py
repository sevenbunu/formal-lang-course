from project.adjacency_matrix_fa import AdjacencyMatrixFA
from project.automaton_transition import graph_to_nfa, regex_to_dfa
from networkx.classes import MultiDiGraph
from scipy.sparse import csr_matrix


def ms_bfs_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    automaton_from_regex = AdjacencyMatrixFA(regex_to_dfa(regex))
    automaton_from_graph = AdjacencyMatrixFA(
        graph_to_nfa(graph, start_nodes, final_nodes)
    )
    front_matrices = {}
    for start_node in start_nodes:
        graph_state = automaton_from_graph.state_to_int[start_node]
        front_matrix = csr_matrix(
            (automaton_from_regex.states_count, automaton_from_graph.states_count),
            dtype=bool,
        )
        for start_state in automaton_from_regex.start_states:
            front_matrix[start_state, graph_state] = True
        front_matrices[start_node] = front_matrix

    visited_matrices = {node: matrix.copy() for node, matrix in front_matrices.items()}
    result = set()

    for start_state in automaton_from_regex.start_states:
        if start_state in automaton_from_regex.final_states:
            for start_node in start_nodes:
                if start_node in final_nodes:
                    result.add((start_node, start_node))

    while any(matrix.nnz > 0 for matrix in front_matrices.values()):
        new_front_matrices = {
            node: csr_matrix(
                (automaton_from_regex.states_count, automaton_from_graph.states_count),
                dtype=bool,
            )
            for node in start_nodes
        }

        for symbol in automaton_from_regex.boolean_decomposition.keys():
            if symbol not in automaton_from_graph.boolean_decomposition:
                continue

            regex_matrix = automaton_from_regex.boolean_decomposition[symbol]
            graph_matrix = automaton_from_graph.boolean_decomposition[symbol]

            for start_node in start_nodes:
                if front_matrices[start_node].nnz == 0:
                    continue

                next_front = (
                    regex_matrix.T @ front_matrices[start_node]
                ) @ graph_matrix

                next_front = next_front.astype(bool)
                next_front = next_front - (
                    next_front.multiply(visited_matrices[start_node])
                )
                next_front.eliminate_zeros()

                new_front_matrices[start_node] = (
                    new_front_matrices[start_node] + next_front
                )
                visited_matrices[start_node] = visited_matrices[start_node] + next_front

        for start_node in start_nodes:
            for final_state in automaton_from_regex.final_states:
                row_indices = new_front_matrices[start_node].getrow(final_state).indices
                for graph_state in row_indices:
                    node = automaton_from_graph.int_to_state[graph_state].value
                    if node in final_nodes:
                        result.add((start_node, node))

        front_matrices = new_front_matrices

    return result
