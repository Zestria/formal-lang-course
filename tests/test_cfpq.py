import networkx as nx
from pyformlang.cfg import CFG, Terminal, Variable

from project.cfpq import cfg_to_weak_normal_form, hellings_based_cfpq


def test_cfg_to_weak_normal_form_preserves_language_and_shape():
    cfg = CFG.from_text("S -> $ | a S b | S S")
    wnf = cfg_to_weak_normal_form(cfg)

    assert wnf.generate_epsilon()
    for production in wnf.productions:
        body = production.body
        assert len(body) <= 2
        if len(body) == 2:
            assert all(isinstance(symbol, Variable) for symbol in body)
        elif len(body) == 1:
            assert isinstance(body[0], Terminal)
        else:
            assert production.head == wnf.start_symbol

    for word in ["", "ab", "aabb", "abab", "aababb"]:
        assert cfg.contains(word) == wnf.contains(word)


def test_hellings_based_cfpq_finds_paths_and_filters_nodes():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="b")
    graph.add_edge(3, 4, label="a")
    graph.add_edge(4, 5, label="b")
    cfg = CFG.from_text("S -> a b")

    assert hellings_based_cfpq(cfg, graph, {0, 3}, {2, 5}) == {(0, 2), (3, 5)}
    assert hellings_based_cfpq(cfg, graph, {0}, {2, 5}) == {(0, 2)}


def test_hellings_based_cfpq_handles_epsilon_and_recursion():
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(range(5))
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="a")
    graph.add_edge(2, 3, label="b")
    graph.add_edge(3, 4, label="b")
    cfg = CFG.from_text("S -> $ | a S b")

    expected = {(node, node) for node in range(5)} | {(0, 4), (1, 3)}
    assert hellings_based_cfpq(cfg, graph) == expected
