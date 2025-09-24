import itertools
from typing import List, Iterable, Set

from networkx.classes import MultiDiGraph
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, Symbol, State
from pyformlang.finite_automaton.finite_automaton import to_state
from scipy.sparse import csr_matrix, kron
from scipy.sparse.linalg import matrix_power

# from project.automaton_transition import regex_to_dfa, graph_to_nfa
from automaton_transition import regex_to_dfa, graph_to_nfa


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
                    data = boolean_decomposition.get(transition, [False] * (states_count * states_count))
                    data[current * states_count + destination] = True
                    boolean_decomposition[transition] = data
        row = [x for x in range(states_count) for _ in range(states_count)]
        col = [y for _ in range(states_count) for y in range(states_count)]
        for transition, data in boolean_decomposition.items():
            boolean_decomposition[transition] = csr_matrix((data, (row, col)), shape=(states_count, states_count), dtype=bool)
        
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

        
# graph = regex_to_dfa("ab*")
# for state in graph.states:
#     print(state)
# print(len(graph.states))
# print(graph.to_dict())

# matrix = AdjacencyMatrixFA(regex_to_dfa("a b*"))
# print(matrix.accepts(['ab', 'b', 'b']))
# for transition, el in matrix.boolean_decomposition.items():
#     print(transition, el.toarray())