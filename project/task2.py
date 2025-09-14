from typing import Set
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
    State,
)
from pyformlang.regular_expression import Regex
from networkx import MultiDiGraph


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    regular_expression = Regex(regex)
    nfa = regular_expression.to_epsilon_nfa()
    dfa = nfa.to_deterministic()
    min_dfa = dfa.minimize()
    return min_dfa


def graph_to_nfa(
    graph: MultiDiGraph, start_states: Set[int], final_states: Set[int]
) -> NondeterministicFiniteAutomaton:
    nfa = NondeterministicFiniteAutomaton().from_networkx(graph)

    nodes = set(graph.nodes())

    if not start_states:
        states_to_add = nodes
    else:
        states_to_add = start_states
    for state in states_to_add:
        if state not in nodes:
            raise ValueError(
                f"node {state} not in graph and can't be used as final state"
            )
        nfa.add_start_state(State(state))

    if not final_states:
        states_to_add = nodes
    else:
        states_to_add = final_states
    for state in states_to_add:
        if state not in nodes:
            raise ValueError(
                f"node {state} not in graph and can't be used as final state"
            )
        nfa.add_final_state(State(state))

    return nfa
