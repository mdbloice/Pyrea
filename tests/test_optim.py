import copy
import random

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.metrics import silhouette_score

from pyrea import nn, optim


class Offset(nn.Block):
    input_kinds = {"data": nn.FEATURES}

    def __init__(self, amount=0, name="offset"):
        self.amount, self.name, self.calls = amount, name, 0

    def forward(self, data):
        self.calls += 1
        if self.amount == "fail":
            raise ValueError("deliberate execution failure")
        return data + self.amount

    def get_params(self):
        return {"amount": self.amount}

    def set_params(self, **params):
        if set(params) != {"amount"}:
            raise ValueError("unknown parameter")
        self.amount = params["amount"]
        return self


def make_graph(shared=False):
    source = nn.Input("x")
    block = Offset()
    output = {"first": block(source), "second": block(source)} if shared else block(source)
    return nn.Graph({"x": source}, output), block


def search(graph, **kwargs):
    options = dict(
        inputs={"x": np.zeros((6, 2))},
        search_space={"offset": {"amount": [-2, 0, 2]}},
        fitness=lambda output: float(output[0, 0]),
        population_size=8,
        generations=3,
        random_state=17,
    )
    options.update(kwargs)
    return optim.genetic_search(graph, **options)


@pytest.mark.parametrize("maximize,expected", [(True, 2), (False, -2)])
def test_direction_reproducibility_budget_and_trial_isolation(maximize, expected):
    graph, block = make_graph()
    before = copy.deepcopy(graph.get_params())
    python_state, numpy_state = random.getstate(), np.random.get_state()
    first = search(graph, maximize=maximize, mutation_probability=1, crossover_probability=1)
    second = search(graph, maximize=maximize, mutation_probability=1, crossover_probability=1)
    assert first.history == second.history
    assert first.logbook == second.logbook
    assert first.best_score == expected
    assert first.best_params == {"offset__amount": expected}
    assert first.n_evaluations == len(first.history) == 8 + 3 * 7
    assert len(first.logbook) == 4
    assert sum(row["evaluations"] for row in first.logbook) == first.n_evaluations
    scores = first.logbook.select("best_score")
    assert scores == sorted(scores, reverse=not maximize)
    assert graph.get_params() == before
    assert block.calls == 0
    best_block = dict(first.best_graph.named_blocks())["offset"]
    assert best_block.calls == 0
    assert_array_equal(first.best_graph(x=np.zeros((6, 2))), np.full((6, 2), expected))
    assert best_block.calls == 1  # Each trial and the returned graph have independent state.
    assert set(first.best_graph.graph.edges) == set(graph.graph.edges)
    assert random.getstate() == python_state
    for actual, original in zip(np.random.get_state(), numpy_state):
        assert_array_equal(actual, original)


def test_shared_block_named_outputs_and_zero_generation_cache():
    graph, original = make_graph(shared=True)
    result = search(
        graph,
        search_space={"offset": {"amount": 3}},
        generations=0,
        cache=True,
        fitness=lambda output: float(output["first"][0, 0]),
    )
    assert result.best_score == 3
    assert result.n_evaluations == 1
    assert len(result.history) == 8
    assert sum(row["cached"] for row in result.history) == 7
    assert result.logbook[0]["cache_hits"] == 7
    outputs = result.best_graph(x=np.ones((6, 2)))
    assert_array_equal(outputs["first"], outputs["second"])
    assert dict(result.best_graph.named_blocks())["offset"].calls == 2
    assert original.calls == 0


def test_cached_scores_are_optional_and_elites_keep_scores():
    graph, _ = make_graph()
    calls = []

    def fitness(output):
        calls.append(None)
        return float(len(calls))

    result = search(graph, search_space={"offset": {"amount": 0}}, fitness=fitness)
    assert result.n_evaluations == len(calls) == 29
    assert result.best_score == 29
    assert not any(row["cached"] for row in result.history)


def test_conditional_cluster_families_survive_crossover_and_mutation():
    source = nn.Input("x")
    graph = nn.Graph({"x": source}, nn.Cluster(name="cluster")(source))
    data = np.random.RandomState(11).normal(size=(20, 3))
    space = {
        "cluster": [
            {
                "clusterer": "hierarchical",
                "method": ["ward", "complete"],
                "n_clusters": optim.Int(2, 4),
            },
            {
                "clusterer": "spectral",
                "affinity": "rbf",
                "gamma": optim.Float(0.1, 2, log=True),
                "n_clusters": optim.Int(2, 4),
                "random_state": 11,
            },
        ]
    }
    result = optim.genetic_search(
        graph,
        {"x": data},
        space,
        lambda labels: silhouette_score(data, labels),
        population_size=10,
        generations=3,
        random_state=11,
        crossover_probability=1,
        mutation_probability=1,
    )
    families = set()
    for row in result.history:
        params = row["params"]
        family = params["cluster__clusterer"]
        families.add(family)
        assert row["status"] == "ok"
        assert 2 <= params["cluster__n_clusters"] <= 4
        if family == "spectral":
            assert "cluster__method" not in params
            assert 0.1 <= params["cluster__gamma"] <= 2
        else:
            assert "cluster__gamma" not in params
            assert "cluster__affinity" not in params
    assert families == {"hierarchical", "spectral"}
    assert silhouette_score(data, result.best_graph(x=data)) == pytest.approx(result.best_score)
    assert graph.get_params() == {
        "cluster__clusterer": "hierarchical",
        "cluster__method": "ward",
        "cluster__n_clusters": 2,
    }


@pytest.mark.parametrize(
    "phase,options",
    [
        ("configuration", {"search_space": {"offset": {"unknown": 1}}}),
        ("execution", {"search_space": {"offset": {"amount": "fail"}}}),
        ("fitness", {"fitness": lambda output: float("nan")}),
        ("fitness", {"fitness": lambda output: True}),
    ],
)
def test_default_errors_preserve_phase_cause_and_history(phase, options):
    graph, _ = make_graph()
    with pytest.raises(optim.SearchError, match="Candidate failed") as caught:
        search(graph, **options)
    assert caught.value.__cause__ is not None
    assert len(caught.value.history) == 1
    row = caught.value.history[0]
    assert row["phase"] == phase
    assert row["status"] == "failed"
    assert row["score"] is None
    assert row["error"]


@pytest.mark.parametrize("maximize", [True, False])
def test_penalize_mixed_and_all_failed_candidates(maximize):
    graph, _ = make_graph()
    result = search(
        graph,
        search_space={"offset": {"amount": ["fail", -5]}},
        on_error="penalize",
        maximize=maximize,
        cache=True,
    )
    assert result.best_score == -5
    assert any(row["status"] == "failed" for row in result.history)
    assert any(row["status"] == "ok" for row in result.history)
    with pytest.raises(optim.SearchError, match="No candidate produced") as caught:
        search(
            graph,
            search_space={"offset": {"amount": "fail"}},
            on_error="penalize",
            cache=True,
            maximize=maximize,
        )
    assert len(caught.value.history) == 29
    assert len(caught.value.logbook) == 4
    assert all(row["status"] == "failed" for row in caught.value.history)
    assert sum(not row["cached"] for row in caught.value.history) == 1


def test_domains_and_mutable_choice_values():
    rng = random.Random(5)
    assert {optim.Int(2, 3).sample(rng) for _ in range(100)} == {2, 3}
    values = [[1], [2]]
    choice = optim.Choice(values)
    values[0].append(3)
    sampled = choice.sample(rng)
    sampled.append(4)
    assert choice.values == ([1], [2])
    assert optim.Float(2, 2).sample(rng) == 2
    assert all(0.001 <= optim.Float(0.001, 0.1, log=True).sample(rng) <= 0.1 for _ in range(100))


@pytest.mark.parametrize(
    "construct,exception",
    [
        (lambda: optim.Choice([]), ValueError),
        (lambda: optim.Choice("ward"), TypeError),
        (lambda: optim.Int(0.1, 2), TypeError),
        (lambda: optim.Int(True, 2), TypeError),
        (lambda: optim.Int(3, 2), ValueError),
        (lambda: optim.Float(2, 1), ValueError),
        (lambda: optim.Float(0, 1, log=True), ValueError),
        (lambda: optim.Float(0, float("inf")), ValueError),
    ],
)
def test_invalid_domains(construct, exception):
    with pytest.raises(exception):
        construct()


@pytest.mark.parametrize(
    "options,exception",
    [
        ({"population_size": 1}, ValueError),
        ({"population_size": True}, TypeError),
        ({"generations": -1}, ValueError),
        ({"elitism": 8}, ValueError),
        ({"tournament_size": 1}, ValueError),
        ({"mutation_probability": float("nan")}, ValueError),
        ({"crossover_probability": 1.1}, ValueError),
        ({"random_state": -2}, ValueError),
        ({"on_error": "ignore"}, ValueError),
        ({"maximize": "yes"}, TypeError),
        ({"cache": 1}, TypeError),
        ({"fitness": None}, TypeError),
        ({"inputs": {"wrong": np.ones((6, 2))}}, ValueError),
        ({"search_space": {}}, ValueError),
        ({"search_space": {"missing": {"amount": 1}}}, ValueError),
        ({"search_space": {"offset": []}}, ValueError),
        ({"search_space": {"offset": [{}]}}, ValueError),
        ({"search_space": {"offset": {"": [1]}}}, ValueError),
    ],
)
def test_invalid_search_configuration(options, exception):
    graph, block = make_graph()
    with pytest.raises(exception):
        search(graph, **options)
    assert block.calls == 0
