from typing import Iterable, Set, Dict
from networkx.classes import MultiDiGraph
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, Symbol, State
from scipy.sparse import csr_matrix, kron

from project.automaton_transition import regex_to_dfa, graph_to_nfa


class AdjacencyMatrixFA:
    def __init__(self, nfa: NondeterministicFiniteAutomaton):
        state_number = 0
        state_to_int, int_to_state = {}, {}
        for state in nfa.states:
            state_to_int[state] = state_number
            int_to_state[state_number] = state
            state_number += 1
        states_count = state_number

        boolean_decomposition = {}
        for current_state, transitions in nfa.to_dict().items():
            for transition, destination_state in transitions.items():
                current = state_to_int[current_state]
                if not isinstance(destination_state, Set):
                    destination_state = {destination_state}
                for state in destination_state:
                    destination = state_to_int[state]
                    data = boolean_decomposition.get(
                        transition, [False] * (states_count * states_count)
                    )
                    data[current * states_count + destination] = True
                    boolean_decomposition[transition] = data
        row = [x for x in range(states_count) for _ in range(states_count)]
        col = [y for _ in range(states_count) for y in range(states_count)]
        for transition, data in boolean_decomposition.items():
            boolean_decomposition[transition] = csr_matrix(
                (data, (row, col)), shape=(states_count, states_count), dtype=bool
            )

        self.boolean_decomposition = boolean_decomposition
        self.start_states = [state_to_int[state] for state in nfa.start_states]
        self.final_states = [state_to_int[state] for state in nfa.final_states]
        self.states_count = states_count
        self.state_to_int = state_to_int
        self.int_to_state = int_to_state

    def accepts(self, word: Iterable[Symbol]) -> bool:
        current_states = self.start_states
        next_states = set()
        for symbol in word:
            try:
                boolean_decomposition_for_symbol = self.boolean_decomposition[symbol]
            except KeyError:
                return False
            for state in current_states:
                row = boolean_decomposition_for_symbol.getrow(state).toarray().ravel()
                next_states.update(row.nonzero()[0])
            current_states = next_states.copy()
            next_states.clear()
        for state in current_states:
            if state in self.final_states:
                return True
        return False

    def is_empty(self) -> csr_matrix:
        adjacency_matrix = self.get_adjacency_matrix()
        transitive_closure = get_transitive_closure(adjacency_matrix)

        for start_state in self.start_states:
            for final_state in self.final_states:
                if transitive_closure[start_state, final_state]:
                    return False
        return True

    @classmethod
    def from_data(
        cls,
        start_states: Set[int],
        final_states: Set[int],
        states_count: int,
        boolean_decomposition: Dict[Symbol, csr_matrix],
        state_to_int: Dict[State, int],
        int_to_state: Dict[int, State],
    ):
        matrix = cls(NondeterministicFiniteAutomaton())
        matrix.boolean_decomposition = boolean_decomposition
        matrix.start_states = start_states
        matrix.final_states = final_states
        matrix.states_count = states_count
        matrix.state_to_int = state_to_int
        matrix.int_to_state = int_to_state
        return matrix

    def get_adjacency_matrix(self) -> csr_matrix:
        adjacency_matrix = csr_matrix(
            ([], ([], [])), shape=(self.states_count, self.states_count), dtype=bool
        )
        for boolean_decomposition_for_symbol in self.boolean_decomposition.values():
            adjacency_matrix += boolean_decomposition_for_symbol

        return adjacency_matrix


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    start_states, final_states = set(), set()
    for start_state1 in automaton1.start_states:
        for start_state2 in automaton2.start_states:
            start_states.add(start_state1 * automaton2.states_count + start_state2)
    for final_state1 in automaton1.final_states:
        for final_state2 in automaton2.final_states:
            final_states.add(final_state1 * automaton2.states_count + final_state2)

    boolean_decomposition = {}
    for (
        symbol,
        boolean_decomposition_for_symbol,
    ) in automaton1.boolean_decomposition.items():
        if symbol not in automaton2.boolean_decomposition:
            continue
        boolean_decomposition[symbol] = kron(
            boolean_decomposition_for_symbol,
            automaton2.boolean_decomposition[symbol],
            format="csr",
        )

    state_to_int, int_to_state = {}, {}
    for int1 in automaton1.state_to_int.values():
        for int2 in automaton2.state_to_int.values():
            new_int = int1 * automaton2.states_count + int2
            new_state = (automaton1.int_to_state[int1], automaton2.int_to_state[int2])
            state_to_int[State(new_state)] = new_int
            int_to_state[new_int] = State(new_state)

    return AdjacencyMatrixFA.from_data(
        start_states,
        final_states,
        automaton1.states_count * automaton2.states_count,
        boolean_decomposition,
        state_to_int,
        int_to_state,
    )


def get_transitive_closure(automaton: csr_matrix):
    transitive_closure = automaton.copy()
    zero_element_count = transitive_closure.nnz
    current_zero_element_count = 0

    while current_zero_element_count != zero_element_count:
        transitive_closure = transitive_closure + transitive_closure @ automaton
        zero_element_count = current_zero_element_count
        current_zero_element_count = transitive_closure.nnz
    return transitive_closure


def tensor_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    automaton_from_regex = AdjacencyMatrixFA(regex_to_dfa(regex))
    automaton_from_graph = AdjacencyMatrixFA(
        graph_to_nfa(graph, start_nodes, final_nodes)
    )
    intersection = intersect_automata(automaton_from_regex, automaton_from_graph)
    adjacency_matrix = intersection.get_adjacency_matrix()

    transitive_closure = get_transitive_closure(adjacency_matrix)
    result = set()
    for start_state1 in automaton_from_regex.start_states:
        for start_state2 in automaton_from_graph.start_states:
            for final_state1 in automaton_from_regex.final_states:
                for final_state2 in automaton_from_graph.final_states:
                    start_state = (
                        start_state1 * automaton_from_graph.states_count + start_state2
                    )
                    final_state = (
                        final_state1 * automaton_from_graph.states_count + final_state2
                    )
                    if (
                        transitive_closure[start_state, final_state]
                        or start_state == final_state
                    ):
                        result.add(
                            (
                                automaton_from_graph.int_to_state[start_state2],
                                automaton_from_graph.int_to_state[final_state2],
                            )
                        )
    return result
