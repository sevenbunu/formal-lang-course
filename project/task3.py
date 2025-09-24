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
        states_count = len(nfa.states)
        for state in nfa.states:
            state_to_int[state] = state_number

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
        for symbol in word:


        
# graph = regex_to_dfa("ab*")
# for state in graph.states:
#     print(state)
# print(len(graph.states))
# print(graph.to_dict())
# graph.write_as_dot("bebra.dot")

# matrix = AdjacencyMatrixFA(regex_to_dfa("ab*"))
# for _, el in matrix.boolean_decomposition.items():
    # print(el.toarray())