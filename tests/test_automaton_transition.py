from project.automaton_transition import regex_to_dfa, graph_to_nfa
from project.graph_utils import get_graph_info
import pytest
from networkx import MultiDiGraph
from cfpq_data import download, graph_from_csv


def test_regex_to_dfa_empty_regex():
    regex = ""
    dfa = regex_to_dfa(regex)
    assert dfa.is_empty()


def test_regex_to_dfa_simple_regex():
    regex = "ab"
    dfa = regex_to_dfa(regex)
    assert dfa.accepts(["ab"])
    assert not dfa.accepts(["a", "b"])


def test_regex_to_dfa_disjunction():
    regex = "abc|d"
    dfa = regex_to_dfa(regex)
    assert dfa.accepts(["abc"])
    assert dfa.accepts(["d"])
    assert not dfa.accepts([])


def test_regex_to_dfa_kleene_star():
    regex = "a*"
    dfa = regex_to_dfa(regex)
    assert dfa.accepts(["a"])
    assert dfa.accepts(["a", "a"])
    assert not dfa.accepts(["aa"])


def test_graph_to_nfa_invalid_start_state():
    graph = MultiDiGraph([(1, 2), (2, 3), (1, 3)])
    with pytest.raises(ValueError):
        graph_to_nfa(graph, {5}, {1})


def test_graph_to_nfa_invalid_final_state():
    graph = MultiDiGraph([(1, 2), (2, 3), (1, 3)])
    with pytest.raises(ValueError):
        graph_to_nfa(graph, {1}, {5})


def test_graph_to_nfa_correct_case():
    name = "bzip"
    graph_info = get_graph_info(name)
    g = download(name)
    graph = graph_from_csv(g)
    start_states = {1}
    final_states = {2, 3}
    nfa = graph_to_nfa(graph, start_states, final_states)

    assert nfa.start_states == start_states
    assert nfa.final_states == final_states
    assert nfa.states == set(graph.nodes())
    assert nfa.get_number_transitions() == graph_info.number_of_edges
    assert nfa.symbols == set(graph_info.labels)
