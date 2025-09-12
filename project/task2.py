from typing import Set, Any
from pyformlang.finite_automaton import DeterministicFiniteAutomaton, NondeterministicFiniteAutomaton, Symbol, State
from pyformlang.regular_expression import Regex
from networkx import MultiDiGraph


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    regular_expression = Regex(regex)
    nfa = regular_expression.to_epsilon_nfa()
    dfa = nfa.to_deterministic()
    min_dfa = dfa.minimize()
    return min_dfa

def convert_to_int_if_possible(value: Any) -> Any:
    try:
        result_value = int(value)
    except (TypeError, ValueError):
        result_value = value
    return result_value

def graph_to_nfa(graph: MultiDiGraph, start_states: Set[int], final_states: Set[int]) -> NondeterministicFiniteAutomaton:
    nfa = NondeterministicFiniteAutomaton()

    nodes = set(graph.nodes())

    if not start_states:
        states_to_add = nodes
    else:
        states_to_add = start_states
    for state in states_to_add:
        if state not in nodes:
            raise KeyError
        nfa.add_start_state(str(state))

    if not final_states:
        states_to_add = nodes
    else:
        states_to_add = final_states
    for state in states_to_add:
        if state not in nodes:
            raise KeyError
        nfa.add_final_state(str(state))

    for node1, node2, data in graph.edges(data=True):
        nfa.add_transition(State(str(node1)), Symbol(data.get("label", "epsilon")), State(str(node2)))

    return nfa

graph_for_test = MultiDiGraph()
graph_for_test.add_nodes_from(["(1, 2)", 1, 2])
graph_for_test.add_edge((1, 2), 1, "a")
graph_for_test.add_edge(1, 1, "b")
graph_for_test.add_edge(1, 2, "c")
print(graph_to_nfa(graph_for_test, (), ()).write_as_dot("bebra"))
