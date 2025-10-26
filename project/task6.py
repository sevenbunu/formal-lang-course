import pyformlang
from pyformlang.cfg import Production, CFG, Epsilon
import networkx as nx
from project.adjacency_matrix_fa import AdjacencyMatrixFA
from project.automaton_transition import graph_to_nfa


def cfg_to_weak_normal_form(cfg: pyformlang.cfg.CFG) -> pyformlang.cfg.CFG:
    is_cfg_generate_epsilon = cfg.generate_epsilon()
    normal_form = cfg.to_normal_form()

    if is_cfg_generate_epsilon:
        normal_form.terminals.add(Epsilon())
        normal_form.productions.add(Production(normal_form.start_symbol, [Epsilon()], False))
        return CFG(
            variables=normal_form.variables,
            terminals=normal_form.terminals,
            start_symbol=normal_form.start_symbol,
            productions=normal_form.productions,
        )
    return normal_form


def hellings_based_cfpq(
    cfg: pyformlang.cfg.CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    cfg_weak_normal_form = cfg_to_weak_normal_form(cfg)
    nfa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))

    r = set()
    for production in cfg_weak_normal_form.productions:
        if len(production.body) > 1:
            continue
        if production.body[0] == Epsilon():
            for v in nfa.int_to_state:
                r.add((production.head, v, v))
        else:
            if production.body[0].value in nfa.boolean_decomposition:
                boolean_decomposition_for_symbol = nfa.boolean_decomposition[
                    production.body[0].value
                ]
                rows, cols = boolean_decomposition_for_symbol.nonzero()
                triples = ((production.head, i, j) for i, j in zip(rows, cols))
                r.update(triples)
    m = r.copy()
    while m:
        new_m = set()
        N_i, v, u = m.pop()
        for fst, snd, thrd in r:
            if thrd == v:
                for production in cfg_weak_normal_form.productions:
                    if (
                        len(production.body) == 2
                        and production.body == [fst, N_i]
                        and (production.head, snd, u) not in r
                    ):
                        new_m.add((production.head, snd, u))
            if snd == u:
                for production in cfg_weak_normal_form.productions:
                    if (
                        len(production.body) == 2
                        and production.body == [N_i, fst]
                        and (production.head, v, thrd) not in r
                    ):
                        new_m.add((production.head, v, thrd))
        m.update(new_m)
        r.update(new_m)

    result = set()
    for N, v, u in r:
        v_state, u_state = nfa.int_to_state[v], nfa.int_to_state[u]
        if (
            N == cfg_weak_normal_form.start_symbol
            and v in nfa.start_states
            and u in nfa.final_states
        ):
            result.add((v_state, u_state))
    return result
