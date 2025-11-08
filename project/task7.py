import pyformlang
import networkx as nx
from typing import Set
from scipy.sparse import lil_matrix
from project.adjacency_matrix_fa import AdjacencyMatrixFA
from project.automaton_transition import graph_to_nfa
from project.task6 import cfg_to_weak_normal_form
import numpy as np


def is_equal(d1, d2):
    if (not d1 and d2) or (d1 and not d2):
        return False
    for k in d1:
        if k not in d2:
            return False
        m1 = d1[k]
        m2 = d2[k]
        if not np.array_equal(m1.toarray(), m2.toarray()):
            return False
    return True


def matrix_based_cfpq(
    cfg: pyformlang.cfg.CFG,
    graph: nx.DiGraph,
    start_nodes: Set[int] = None,
    final_nodes: Set[int] = None,
) -> set[tuple[int, int]]:
    automaton = AdjacencyMatrixFA(
        graph_to_nfa((nx.MultiDiGraph(graph)), start_nodes, final_nodes)
    )
    n = automaton.states_count
    cfg_wnf = cfg_to_weak_normal_form(cfg)
    epsilon = ("epsilon",)

    t = {
        nonterminal.value: lil_matrix((n, n), dtype=bool)
        for nonterminal in cfg_wnf.variables
    }
    reversed_productions = {}
    for production in cfg_wnf.productions:
        head, body = (
            production.head.value,
            tuple(elem.value for elem in production.body),
        )
        if body not in reversed_productions:
            reversed_productions[body] = [head]
        else:
            reversed_productions[body].append(head)
        for (
            symbol,
            boolean_decomposition_for_symbol,
        ) in automaton.boolean_decomposition.items():
            rows, cols = boolean_decomposition_for_symbol.nonzero()
            for i in range(len(rows)):
                if symbol in body:
                    t[head][rows[i], cols[i]] = True
    if epsilon in reversed_productions:
        for i in range(n):
            for head in reversed_productions[epsilon]:
                t[head][i, i] = True

    for nt, bool_decomp_for_nt in t.items():
        t[nt] = bool_decomp_for_nt.tocsr()
    prev_t = {}
    while not is_equal(prev_t, t):
        prev_t = {nt: bool_decomp_for_nt.copy() for nt, bool_decomp_for_nt in t.items()}
        for nt1, bool_decomp_for_nt1 in t.items():
            for nt2, bool_decomp_for_nt2 in t.items():
                bool_decomp_for_nt = bool_decomp_for_nt1 @ bool_decomp_for_nt2
                if (nt1, nt2) not in reversed_productions:
                    continue
                for head in reversed_productions[(nt1, nt2)]:
                    t[head] += bool_decomp_for_nt
    result = set()
    rows, cols = t[cfg_wnf.start_symbol].nonzero()
    for i in range(len(rows)):
        if rows[i] in automaton.start_states and cols[i] in automaton.final_states:
            result.add(
                (automaton.int_to_state[rows[i]], automaton.int_to_state[cols[i]])
            )
    return result
