import pyformlang
import networkx as nx
from project.adjacency_matrix_fa import AdjacencyMatrixFA
from project.automaton_transition import graph_to_nfa

# Recursive State Automaton with Adjacency Matrix
class RSAAD(pyformlang.rsa.RecursiveAutomaton):
    def __init__():
        super().__init__()


    def get_adjacency_matrix():
        pass

    @classmethod
    def from_rsa():


def tensor_based_cfpq(
  rsm: pyformlang.rsa.RecursiveAutomaton,
  graph: nx.DiGraph,
  start_nodes: set[int] = None,
  final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    automaton = AdjacencyMatrixFA(graph_to_nfa(nx.MultiDiGraph(graph), start_nodes, final_nodes))
    rsmad = RSAAD.from_rsa(rsm)

    adj_matrix_from_graph = automaton.get_adjacency_matrix()
    adj_matrix_from_rsm = rsmad.get_adjacency_matrix()
    
    prev_matrix = {}
    while prev_matrix != adj_matrix_from_graph:
        prev_matrix = {k: v.copy() for k, v in t.items()}

def cfg_to_rsm(cfg: pyformlang.cfg.CFG) -> pyformlang.rsa.RecursiveAutomaton:
    return pyformlang.rsa.RecursiveAutomaton.from_text(cfg.to_text())


def ebnf_to_rsm(ebnf: str) -> pyformlang.rsa.RecursiveAutomaton:
    return pyformlang.rsa.RecursiveAutomaton.from_ebnf(ebnf)
