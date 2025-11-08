import pyformlang
from pyformlang.finite_automaton import (
    NondeterministicFiniteAutomaton,
)
import networkx as nx
from project.adjacency_matrix_fa import (
    AdjacencyMatrixFA,
    get_transitive_closure,
    intersect_automata,
)

from project.automaton_transition import graph_to_nfa

from scipy.sparse import csr_matrix
from pyformlang.finite_automaton.finite_automaton import State
from typing import Set


def rsm_to_nfa(
    rsm: pyformlang.rsa.RecursiveAutomaton,
) -> NondeterministicFiniteAutomaton:
    nfa = NondeterministicFiniteAutomaton()

    for symbol, box in rsm.boxes.items():
        dfa = box.dfa
        for start_state in dfa.start_states:
            nfa.add_start_state(State((symbol, start_state)))
        for final_state in dfa.final_states:
            nfa.add_final_state(State((symbol, final_state)))

        for current_state, transitions in dfa.to_dict().items():
            for transition, destination_state in transitions.items():
                current = State((symbol, current_state))
                if not isinstance(destination_state, Set):
                    destination_state = {destination_state}
                for state in destination_state:
                    destination = State((symbol, state))
                    nfa.add_transition(current, transition, destination)

    return nfa


def tensor_based_cfpq(
    rsm: pyformlang.rsa.RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    adj_matrix_from_rsm = AdjacencyMatrixFA(rsm_to_nfa(rsm))
    adj_matrix_from_graph = AdjacencyMatrixFA(
        graph_to_nfa(nx.MultiDiGraph(graph), start_nodes, final_nodes)
    )

    for nt in rsm.boxes:
        if nt not in adj_matrix_from_graph.boolean_decomposition:
            adj_matrix_from_graph.boolean_decomposition[nt] = csr_matrix(
                (
                    adj_matrix_from_graph.states_count,
                    adj_matrix_from_graph.states_count,
                ),
                dtype=bool,
            )

    changed = True
    while changed:
        changed = False

        intersection = intersect_automata(adj_matrix_from_rsm, adj_matrix_from_graph)
        n = intersection.states_count
        diagonal = [i for i in range(n)]
        adj_matrix = csr_matrix(
            ([True] * n, (diagonal, diagonal)),
            shape=(n, n),
            dtype=bool,
        )
        if intersection.boolean_decomposition:
            adj_matrix += intersection.get_adjacency_matrix()

        tc = get_transitive_closure(adj_matrix)
        tc = tc.astype(bool)
        rows, cols = tc.nonzero()

        for k in range(len(rows)):
            i, j = rows[k], cols[k]
            current = intersection.int_to_state[i].value
            destination = intersection.int_to_state[j].value
            current_rsm_state, current_graph_state = current
            current_symbol = current_rsm_state.value[0]
            current_rsm = adj_matrix_from_rsm.state_to_int[current_rsm_state]
            current_graph = adj_matrix_from_graph.state_to_int[current_graph_state]
            destination_rsm_state, destination_graph_state = destination
            destination_rsm = adj_matrix_from_rsm.state_to_int[destination_rsm_state]
            destination_graph = adj_matrix_from_graph.state_to_int[
                destination_graph_state
            ]

            if (
                current_rsm in adj_matrix_from_rsm.start_states
                and destination_rsm in adj_matrix_from_rsm.final_states
                and not adj_matrix_from_graph.boolean_decomposition[current_symbol][
                    current_graph, destination_graph
                ]
            ):
                changed = True
                adj_matrix_from_graph.boolean_decomposition[current_symbol][
                    current_graph, destination_graph
                ] = True

    result = set()

    for start_state in adj_matrix_from_graph.start_states:
        for final_state in adj_matrix_from_graph.final_states:
            if adj_matrix_from_graph.boolean_decomposition[rsm.initial_label][
                start_state, final_state
            ]:
                result.add(
                    (
                        adj_matrix_from_graph.int_to_state[start_state],
                        adj_matrix_from_graph.int_to_state[final_state],
                    )
                )

    return result


def cfg_to_rsm(cfg: pyformlang.cfg.CFG) -> pyformlang.rsa.RecursiveAutomaton:
    return pyformlang.rsa.RecursiveAutomaton.from_text(cfg.to_text())


def ebnf_to_rsm(ebnf: str) -> pyformlang.rsa.RecursiveAutomaton:
    return pyformlang.rsa.RecursiveAutomaton.from_text(ebnf)
