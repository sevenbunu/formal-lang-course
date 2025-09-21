import itertools
from typing import List, Iterable, Set

from networkx.classes import MultiDiGraph
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, Symbol, State
from pyformlang.finite_automaton.finite_automaton import to_state
from scipy.sparse import bsr_matrix, kron
from scipy.sparse.linalg import matrix_power

from project.automaton_transition import regex_to_dfa, graph_to_nfa


class AdjacencyMatrixFA:
    def __init__(self, nfa: NondeterministicFiniteAutomaton | None):
        if nfa is None:
            self._states_count: int = 0
            self._state_to_id: dict[State, int] = {}
            self._start_states_ids: Set[int] = set()
            self._final_states_ids: Set[int] = set()
            self._bool_decomposition: dict[Symbol, bsr_matrix] = {}
            return

        self._states_count = len(nfa.states)
        self._state_to_id: dict[State, int] = {}
        current_number = 0
        for state in nfa.states:
            self._state_to_id[state] = current_number
            current_number += 1

        self._start_states_ids = set(
            map(
                lambda x: self._state_to_id[to_state(x)],
                list(nfa.start_states),
            )
        )
        self._final_states_ids = set(
            map(
                lambda x: self._state_to_id[to_state(x)],
                list(nfa.final_states),
            )
        )

        matrices = {}
        for edges_from_v in nfa.to_dict().items():
            v_from = self._state_to_id[edges_from_v[0]]
            for edge in edges_from_v[1].items():
                (symbol, to) = edge
                if symbol not in matrices:
                    matrices[symbol] = []
                if isinstance(to, State):
                    matrices[symbol].append((v_from, self._state_to_id[to]))
                elif hasattr(to, "__iter__"):
                    for v_to in to:
                        matrices[symbol].append((v_from, self._state_to_id[v_to]))
                else:
                    raise ValueError("Unexpected format of graph")

        self._bool_decomposition = {}
        for symbol_info in matrices.items():
            symbol: Symbol = symbol_info[0]
            rows: List[int] = list(map(lambda x: x[0], symbol_info[1]))
            cols: List[int] = list(map(lambda x: x[1], symbol_info[1]))
            data: List[bool] = [True] * len(rows)
            self._bool_decomposition[symbol] = bsr_matrix(
                (data, (rows, cols)),
                shape=(self._states_count, self._states_count),
                dtype=bool,
            )

    @classmethod
    def from_bool_decomposition(
        cls,
        states_count: int,
        state_to_id: dict[State, int],
        bool_decomposition: dict[Symbol, bsr_matrix],
        start_states_ids: Set[int],
        final_states_ids: Set[int],
    ):
        if not start_states_ids:
            start_states_ids = set([i for i in range(states_count)])
        if not final_states_ids:
            final_states_ids = set([i for i in range(states_count)])

        for state in start_states_ids:
            if state >= states_count:
                raise ValueError(
                    f"Id of state {state} is greater than or equal to the count of states in automation"
                )
        for state in final_states_ids:
            if state >= states_count:
                raise ValueError(
                    f"Id of state {state} is greater than or equal to the count of states in automation"
                )

        adj_matrix_fa = cls(None)
        adj_matrix_fa._states_count = states_count
        adj_matrix_fa._state_to_id = state_to_id
        adj_matrix_fa._bool_decomposition = bool_decomposition
        adj_matrix_fa._start_states_ids = start_states_ids
        adj_matrix_fa._final_states_ids = final_states_ids
        return adj_matrix_fa

    @property
    def states_count(self) -> int:
        return self._states_count

    @property
    def state_to_id(self) -> dict[State, int]:
        return self._state_to_id

    @property
    def id_to_state(self) -> dict[int, State]:
        res = {}
        for state, idx in self._state_to_id.items():
            res[idx] = state
        return res

    @property
    def start_states_ids(self) -> Set[int]:
        return self._start_states_ids

    @property
    def final_states_ids(self) -> Set[int]:
        return self._final_states_ids

    @property
    def bool_decomposition(self):
        return self._bool_decomposition

    def add_transition(self, from_id, symbol, to_id):
        if self.has_transition(from_id, symbol, to_id):
            return

        if symbol not in self._bool_decomposition.keys():
            self._bool_decomposition[symbol] = bsr_matrix(
                ([True], ([from_id], [to_id])),
                shape=(self._states_count, self._states_count),
                dtype=bool,
            )
        else:
            matrix_csr = self._bool_decomposition[symbol].tocsr()
            matrix_csr[from_id, to_id] = True
            self._bool_decomposition[symbol] = matrix_csr.tobsr()

    def has_transition(self, from_id, symbol, to_id) -> bool:
        if symbol in self._bool_decomposition.keys():
            return self._bool_decomposition[symbol].getrow(from_id)[0, to_id]
        return False

    def accepts(self, word: Iterable[Symbol]) -> bool:
        current_states: Set[int] = self._start_states_ids
        for symbol in word:
            new_states: Set[int] = set()
            for state in current_states:
                if symbol in self._bool_decomposition:
                    matrix = self._bool_decomposition[symbol]
                    row = matrix.getrow(state)
                    for new_state in row.indices:
                        new_states.add(new_state)
            current_states = new_states
        for state in current_states:
            if state in self._final_states_ids:
                return True
        return False

    def is_empty(self) -> bool:
        transitive_closure = self.get_transitive_closure()
        for start, final in itertools.product(
            self._start_states_ids, self._final_states_ids
        ):
            row = transitive_closure.getrow(start)
            if final in row.indices:
                return False
        return True

    def get_transitive_closure(self) -> bsr_matrix:
        indexes = [i for i in range(self._states_count)]
        bool_id_matrix = bsr_matrix(
            ([True] * self._states_count, (indexes, indexes)),
            shape=(self._states_count, self._states_count),
            dtype=bool,
        )

        if not self._bool_decomposition:
            return bool_id_matrix

        return matrix_power(
            bool_id_matrix + self._get_bool_adjacency_matrix(),
            self._states_count,
        )

    def _get_bool_adjacency_matrix(self) -> bsr_matrix | None:
        if not self._bool_decomposition:
            return None

        matrices = list(self._bool_decomposition.values())
        matrices_count = len(matrices)
        result = matrices[0]
        for i in range(1, matrices_count):
            result += matrices[i]
        return result


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    start_states_ids: Set[int] = set()
    for start1, start2 in itertools.product(
        automaton1.start_states_ids, automaton2.start_states_ids
    ):
        start_states_ids.add(start1 * automaton2.states_count + start2)

    final_states_ids: Set[int] = set()
    for final1, final2 in itertools.product(
        automaton1.final_states_ids, automaton2.final_states_ids
    ):
        final_states_ids.add(final1 * automaton2.states_count + final2)

    state_to_id: dict[State, int] = {}
    for state1, id1 in automaton1.state_to_id.items():
        for state2, id2 in automaton2.state_to_id.items():
            state_to_id[to_state((state1, state2))] = (
                id1 * automaton2.states_count + id2
            )

    bool_decomposition1 = automaton1.bool_decomposition
    bool_decomposition2 = automaton2.bool_decomposition
    symbols1 = set(bool_decomposition1.keys())
    symbols2 = set(bool_decomposition2.keys())
    symbols = symbols1.intersection(symbols2)
    bool_decomposition: dict[Symbol, bsr_matrix] = {}
    for symbol in symbols:
        bool_decomposition[symbol] = kron(
            bool_decomposition1[symbol], bool_decomposition2[symbol]
        ).tobsr()

    return AdjacencyMatrixFA.from_bool_decomposition(
        states_count=automaton1.states_count * automaton2.states_count,
        state_to_id=state_to_id,
        bool_decomposition=bool_decomposition,
        start_states_ids=start_states_ids,
        final_states_ids=final_states_ids,
    )


def tensor_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    regex_dfa = AdjacencyMatrixFA(regex_to_dfa(regex))
    nfa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    intersection = intersect_automata(regex_dfa, nfa)
    intersection_tc = intersection.get_transitive_closure()
    print(intersection_tc.toarray())

    result = set()
    for dfa_start_id, dfa_final_id in itertools.product(
        regex_dfa.start_states_ids, regex_dfa.final_states_ids
    ):
        for nfa_start_id, nfa_final_id in itertools.product(
            nfa.start_states_ids, nfa.final_states_ids
        ):
            row_id = dfa_start_id * nfa.states_count + nfa_start_id
            col_id = dfa_final_id * nfa.states_count + nfa_final_id
            row = intersection_tc.getrow(row_id)
            if col_id in row.indices:
                result.add(
                    (
                        nfa.id_to_state[nfa_start_id].value,
                        nfa.id_to_state[nfa_final_id].value,
                    )
                )

    return result
