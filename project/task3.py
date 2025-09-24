import itertools
from typing import List, Iterable, Set, Dict
from numpy import clip
from networkx.classes import MultiDiGraph
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, Symbol, State
from pyformlang.finite_automaton.finite_automaton import to_state
from scipy.sparse import csr_matrix, kron
from scipy.sparse.linalg import matrix_power

from project.automaton_transition import regex_to_dfa, graph_to_nfa
# from automaton_transition import regex_to_dfa, graph_to_nfa


class AdjacencyMatrixFA:
    def __init__(self, nfa: NondeterministicFiniteAutomaton):
        state_number = 0
        state_to_int = {}
        for state in nfa.states:
            state_to_int[state] = state_number
            state_number += 1
        states_count = state_number

        boolean_decomposition = {}
        for current_state, transitions in nfa.to_dict().items():
            for transition, destination_state in transitions.items():
                if nfa.is_deterministic():
                    current = state_to_int[current_state]
                    destination = state_to_int[destination_state]
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
        adjacency_matrix = csr_matrix(
            ([], ([], [])), shape=(self.states_count, self.states_count), dtype=bool
        )
        for _, boolean_decomposition_for_symbol in self.boolean_decomposition.items():
            adjacency_matrix += boolean_decomposition_for_symbol

        transitive_closure = adjacency_matrix.copy()
        zero_element_count = transitive_closure.nnz
        current_zero_element_count = 0

        while current_zero_element_count != zero_element_count:
            transitive_closure = (
                transitive_closure + transitive_closure @ adjacency_matrix
            )
            zero_element_count = current_zero_element_count
            current_zero_element_count = transitive_closure.nnz

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
    ):
        matrix = cls(NondeterministicFiniteAutomaton())
        matrix.boolean_decomposition = boolean_decomposition
        matrix.start_states = start_states
        matrix.final_states = final_states
        matrix.states_count = states_count
        return matrix


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    start_states, final_states = set(), set()
    for start_state1 in automaton1.start_states:
        for start_state2 in automaton2.start_states:
            start_states.add(start_state1 * automaton2.states_count + start_state2)
    for final_state1 in automaton1.final_states:
        for final_state2 in automaton2.final_states:
            start_states.add(final_state1 * automaton2.states_count + final_state2)

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

    return AdjacencyMatrixFA.from_data(
        start_states,
        final_states,
        automaton1.states_count * automaton2.states_count,
        boolean_decomposition,
    )


# graph = regex_to_dfa("ab*")
# for state in graph.states:
#     print(state)
# print(len(graph.states))
# print(graph.to_dict())

# matrix = AdjacencyMatrixFA(regex_to_dfa("a b*"))
# print(matrix.is_empty())
# print(matrix.accepts(['ab', 'b', 'b']))
# for transition, el in matrix.boolean_decomposition.items():
#     print(transition, el.toarray())
