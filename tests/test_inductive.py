import pickle

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.distance import cdist, squareform, pdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

from pyrea import models, nn, optim


TRAIN = np.array([[-4.0, 0.0], [-3.0, 1.0], [-2.0, 0.0], [2.0, 0.0], [3.0, 1.0], [4.0, 0.0]])


def simple(assignment="nearest"):
    x = nn.Input("x")
    model = nn.Graph({"x": x}, nn.Hierarchical(2, "complete", name="cluster")(x))
    return model.fit(x=TRAIN, assignment=assignment)


@pytest.mark.parametrize("builder", [models.parea_1, models.parea_2])
@pytest.mark.parametrize("assignment", ["nearest", "average", "medoid"])
def test_parea_single_batch_order_and_frozen_labels(builder, assignment):
    graph = builder(["gene", "lipid"], n_clusters=2, return_matrix=True)
    data = {"gene": TRAIN.copy(), "lipid": TRAIN * [2, 3] + 20}
    expected = graph(**data)
    assert graph.fit(data, assignment=assignment) is graph
    assert_array_equal(graph.labels_, expected["labels"])
    assert_array_equal(graph.training_output_["distance"], expected["distance"])
    batch = graph.predict(data)
    assert_array_equal(
        batch,
        np.concatenate(
            [
                graph.predict({key: value[[i]] for key, value in data.items()})
                for i in range(len(TRAIN))
            ]
        ),
    )
    assert_array_equal(
        batch[::-1], graph.predict({key: value[::-1] for key, value in data.items()})
    )
    assert_array_equal(batch, expected["labels"])
    assert graph.n_samples_fit_ == len(TRAIN)
    # The first new row belongs to a later learned cluster: no renumbering.
    assert (
        graph.predict({key: value[-1:] for key, value in data.items()})[0] == expected["labels"][-1]
    )
    state = pickle.dumps(graph)
    graph.predict(data)
    assert pickle.dumps(graph) == state
    restored = pickle.loads(state)
    assert_array_equal(restored.predict(data), batch)


@pytest.mark.parametrize("method", ["disagreement", "agreement", "consensus"])
@pytest.mark.parametrize("normalize", [False, True])
def test_fusion_uses_new_by_training_memberships_without_diagonal_changes(method, normalize):
    a, b = nn.Input("a", nn.LABELS), nn.Input("b", nn.LABELS)
    graph = nn.Graph({"a": a, "b": b}, nn.Fusion(method, normalize=normalize)(a, b))
    first, second = np.array([10, 10, 20, 20]), np.array([30, 40, 30, 40])
    graph.fit(a=first, b=second)
    result = graph.transform(a=np.array([20]), b=np.array([40]))
    expected = {"disagreement": [2, 1, 1, 0], "agreement": [0, 1, 1, 2], "consensus": [0, 0, 0, 1]}[
        method
    ]
    expected = np.array([expected], dtype=float)
    if normalize and method != "consensus":
        expected /= 2
    assert_allclose(result, expected)
    repeated = graph.transform(a=first[::-1], b=second[::-1])
    # Even a square new-by-training matrix has no artificial zero diagonal.
    assert repeated.shape == (4, 4)
    assert_allclose(repeated[0], expected[0])


def test_consensus_known_ids_and_unseen_tuple_policy():
    a, b = nn.Input("a", nn.LABELS), nn.Input("b", nn.LABELS)
    graph = nn.Graph({"a": a, "b": b}, nn.Consensus()(a, b))
    data = {"a": np.array([10, 10, 20, 20]), "b": np.array([30, 40, 30, 30])}
    assert_array_equal(graph.fit_predict(data), [1, 2, 3, 3])
    assert_array_equal(graph.predict(a=np.array([20]), b=np.array([30])), [3])
    # (20, 40) was never observed. The nearest observed tuples tie;
    # first training occurrence chooses tuple (10, 40), with frozen label 2.
    assert_array_equal(graph.predict(a=np.array([20]), b=np.array([40])), [2])
    graph.fit(data, consensus="raise")
    with pytest.raises(RuntimeError, match="Unseen consensus"):
        graph.predict(a=np.array([20]), b=np.array([40]))
    assert_array_equal(graph.predict(a=np.array([20]), b=np.array([30])), [3])


class AuditScale(BaseEstimator, TransformerMixin):
    fit_events = []

    def fit(self, data, y=None):
        self.fit_events.append(len(data))
        self.mean_ = np.mean(data, axis=0)
        return self

    def transform(self, data):
        return data - self.mean_


def test_preprocessing_and_native_predict_are_retained_without_refitting(monkeypatch):
    AuditScale.fit_events = []
    x = nn.Input("x")
    scaled = nn.Transform(AuditScale(), name="scale")(x)
    labels = nn.Estimator(KMeans(n_clusters=2, random_state=42, n_init=10), name="kmeans")(scaled)
    graph = nn.Graph({"x": x}, {"labels": labels, "scaled": scaled}).fit(x=TRAIN)
    assert AuditScale.fit_events == [len(TRAIN)]
    original = graph.labels_.copy()

    def forbidden(*args, **kwargs):
        raise AssertionError("prediction tried to fit")

    monkeypatch.setattr(AuditScale, "fit", forbidden)
    monkeypatch.setattr(KMeans, "fit", forbidden)
    monkeypatch.setattr(KMeans, "fit_predict", forbidden)
    point = np.array([[2.0, 4.0]])
    out = graph.transform(x=point)
    assert_allclose(out["scaled"], point - TRAIN.mean(axis=0))
    assert not np.allclose(out["scaled"], 0)
    assert_array_equal(graph.predict(x=point), out["labels"])
    assert_array_equal(graph.labels_, original)


def test_prediction_never_invokes_clustering_fusion_or_fit_again(monkeypatch):
    class Ensemble(nn.Module):
        def __init__(self):
            self.scale = nn.Standardize()
            self.cluster = nn.Hierarchical(2)
            self.fusion = nn.Fusion()
            self.first = nn.Hierarchical(2, input_kind=nn.DISTANCE)
            self.second = nn.Hierarchical(2, "complete", input_kind=nn.DISTANCE)
            self.consensus = nn.Consensus()

        def forward(self, a, b):
            distance = self.fusion(self.cluster(self.scale(a)), self.cluster(self.scale(b)))
            return self.consensus(self.first(distance), self.second(distance))

    model = Ensemble().compile().fit(a=TRAIN, b=TRAIN + 100)

    def forbidden(*args, **kwargs):
        raise AssertionError("transductive computation during prediction")

    for cls in [nn.Cluster, nn.Fusion, nn.Consensus]:
        monkeypatch.setattr(cls, "forward", forbidden)
    monkeypatch.setattr(nn.Transform, "fit_forward", forbidden)
    monkeypatch.setattr(nn.Standardize, "fit_forward", forbidden)
    assert model.predict(a=TRAIN[:1], b=TRAIN[:1] + 100).shape == (1,)


def test_standardize_matches_fitted_scaler_and_preserves_schema():
    frame = pd.DataFrame(TRAIN, columns=["a", "b"], index=list("abcdef"))
    x = nn.Input("x")
    scaled = nn.Standardize()(x)
    graph = nn.Graph({"x": x}, scaled).fit(x=frame)
    point = pd.DataFrame([[100.0, 3.0]], columns=["a", "b"], index=["patient"])
    assert_allclose(
        graph.transform(x=point), StandardScaler().fit(TRAIN).transform(point.to_numpy())
    )
    with pytest.raises(ValueError, match="feature names/order"):
        graph.transform(x=point[["b", "a"]])
    with pytest.raises(ValueError, match="feature count"):
        graph.transform(x=np.ones((1, 3)))
    with pytest.raises(ValueError, match="two-dimensional"):
        graph.transform(x=np.ones(2))


def test_inplace_transformer_cannot_modify_patient_data_or_shared_inputs():
    x = nn.Input("x")
    graph = nn.Graph({"x": x}, {"raw": x, "scaled": nn.Standardize(copy=False)(x)})
    training = TRAIN.copy()
    graph.fit(x=training)
    assert_array_equal(training, TRAIN)
    assert_array_equal(graph.training_output_["raw"], TRAIN)
    patient = np.array([[100.0, 3.0]])
    before = patient.copy()
    output = graph.transform(x=patient)
    assert_array_equal(patient, before)
    assert_array_equal(output["raw"], before)
    assert_allclose(output["scaled"], StandardScaler().fit(TRAIN).transform(before))


def test_shared_block_has_independent_training_state_per_call():
    x, y = nn.Input("x"), nn.Input("y")
    shared = nn.Standardize(name="scale")
    graph = nn.Graph({"x": x, "y": y}, {"first": shared(x), "second": shared(y)})
    graph.fit(x=TRAIN, y=TRAIN + 100)
    output = graph.transform(x=TRAIN[:1], y=TRAIN[:1] + 100)
    assert_allclose(output["first"], output["second"])
    assert len(list(graph.named_blocks())) == 1
    assert not hasattr(shared, "transformer_")
    assert len(graph._fitted.states) == 2


@pytest.mark.parametrize("rule,expected_index", [("nearest", 2), ("average", 3), ("medoid", 3)])
def test_assignment_rules_are_explicit_and_differ(rule, expected_index):
    data = np.array([[0.0], [1.0], [10.0], [30.0], [31.0]])
    x = nn.Input("x")
    graph = nn.Graph({"x": x}, nn.Hierarchical(2, "complete", name="cluster")(x))
    graph.fit(x=data, assignment={"cluster": rule})
    assert graph.predict(x=np.array([[18.0]]))[0] == graph.labels_[expected_index]


@pytest.mark.parametrize("kind", [nn.DISTANCE, nn.AFFINITY])
@pytest.mark.parametrize("assignment", ["nearest", "average", "medoid"])
def test_precomputed_input_cross_values_and_training_column_ids(kind, assignment):
    ids = list("abcdef")
    distances = squareform(pdist(TRAIN))
    values = distances if kind == nn.DISTANCE else np.exp(-(distances**2) / 4)
    frame = pd.DataFrame(values, index=ids, columns=ids)
    x = nn.Input("x", kind)
    block = (
        nn.Hierarchical(2, "complete", input_kind=kind)
        if kind == nn.DISTANCE
        else nn.Cluster("spectral", n_clusters=2, random_state=42, input_kind=kind)
    )
    graph = nn.Graph({"x": x}, block(x)).fit(x=frame, assignment=assignment)
    query = pd.DataFrame(values[-1:], index=["patient"], columns=ids)
    assert graph.predict(x=query)[0] == graph.labels_[-1]
    with pytest.raises(ValueError, match="training sample order"):
        graph.predict(x=query[ids[::-1]])
    with pytest.raises(ValueError, match="shape"):
        graph.predict(x=np.ones((1, 1)))
    bad = query.copy()
    bad.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        graph.predict(x=bad)


@pytest.mark.parametrize(
    "clusterer,params",
    [
        ("spectral", {"n_clusters": 2, "affinity": "rbf", "gamma": 0.1, "random_state": 42}),
        ("dbscan", {"eps": 2, "min_samples": 2}),
        ("optics", {"min_samples": 2, "min_cluster_size": 2}),
    ],
)
def test_non_native_families_use_documented_frozen_label_transfer(clusterer, params):
    x = nn.Input("x")
    graph = nn.Graph({"x": x}, nn.Cluster(clusterer, **params)(x)).fit(x=TRAIN)
    query = np.array([[-2.1, 0.1], [2.1, 0.1]])
    expected = graph.labels_[np.argmin(cdist(query, TRAIN), axis=1)]
    assert_array_equal(graph.predict(x=query), expected)


@pytest.mark.parametrize("metric", ["cityblock", "seuclidean", "mahalanobis"])
def test_assignment_honors_feature_distance_and_freezes_metric_statistics(metric):
    data = np.random.RandomState(21).normal(size=(20, 3))
    x = nn.Input("x")
    graph = nn.Graph({"x": x}, nn.Hierarchical(3, "complete", distance_metric=metric)(x)).fit(
        x=data
    )
    points = np.array([[0.2, 0.3, -0.2], [100.0, -30.0, 0.1]])
    params = {
        "seuclidean": {"V": data.var(axis=0, ddof=1)},
        "mahalanobis": {"VI": np.linalg.inv(np.cov(data.T))},
    }.get(metric, {})
    expected = graph.labels_[np.argmin(cdist(points, data, metric=metric, **params), axis=1)]
    assert_array_equal(graph.predict(x=points), expected)
    assert_array_equal(graph.predict(x=points[:1]), expected[:1])


def test_supplied_mahalanobis_metric_does_not_require_invertible_training_covariance():
    data = np.array([[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]])
    x = nn.Input("x")
    block = nn.Cluster(
        "dbscan", eps=3, min_samples=2, metric="mahalanobis", metric_params={"VI": np.eye(2)}
    )
    graph = nn.Graph({"x": x}, block(x)).fit(x=data)
    assert graph.predict(x=np.array([[0.25, 0.25]]))[0] == graph.labels_[0]


def test_frozen_affinity_conversion_has_no_batch_maximum_or_fake_diagonal():
    x = nn.Input("x", nn.AFFINITY)
    graph = nn.Graph({"x": x}, nn.AffinityToDistance()(x))
    graph.fit(x=np.array([[0.0, 0.3, 0.7], [0.3, 0.0, 0.5], [0.7, 0.5, 0.0]]))
    query = np.array([[0.5, 1.0, 0.2]])
    expected = np.array([[0.2, 0.0, 0.5]]) / 0.7
    assert_allclose(graph.transform(x=query), expected)
    assert_allclose(graph.transform(x=np.vstack([query, [0.1, 0.1, 0.1]]))[:1], expected)


def test_fit_input_and_exposed_training_output_are_isolated():
    data = TRAIN.copy()
    model = simple()
    model.fit(x=data)
    before = model.predict(x=TRAIN)
    data[:] = 999
    model.labels_[:] = 123
    model.training_output_[:] = 456
    assert_array_equal(model.predict(x=TRAIN), before)
    assert set(model.labels_) != {123}
    # Transductive calls are separate and do not change the fitted snapshot.
    model(x=TRAIN + 50)
    assert_array_equal(model.predict(x=TRAIN), before)


def test_refit_and_parameter_copy_invalidate_old_training():
    model = simple()
    before = model.predict(x=TRAIN)
    trial = model.with_params(cluster__n_clusters=3)
    with pytest.raises(NotFittedError):
        trial.predict(x=TRAIN[:1])
    assert_array_equal(model.predict(x=TRAIN), before)
    model.fit(x=TRAIN[:4])
    assert model.n_samples_fit_ == 4
    with pytest.raises(ValueError):
        model.fit(x=np.empty((0, 2)))
    with pytest.raises(NotFittedError):
        model.predict(x=TRAIN[:1])


def test_optimize_fitted_graph_returns_unfitted_candidate_and_preserves_original():
    graph = simple()
    original = pickle.dumps(graph)
    result = optim.genetic_search(
        graph,
        {"x": TRAIN},
        {"cluster": {"n_clusters": [2, 3]}},
        lambda labels: -float(len(np.unique(labels))),
        population_size=4,
        generations=1,
        random_state=4,
    )
    assert pickle.dumps(graph) == original
    with pytest.raises(NotFittedError):
        result.best_graph.predict(x=TRAIN[:1])
    result.best_graph.fit(x=TRAIN)
    assert result.best_graph.predict(x=TRAIN[:1]).shape == (1,)


def test_input_alignment_is_checked_for_new_patients():
    graph = models.parea_2(["a", "b"], n_clusters=2).fit(a=TRAIN, b=TRAIN)
    a = pd.DataFrame(TRAIN[:2], index=["p", "q"])
    b = a.iloc[::-1]
    with pytest.raises(ValueError, match="identifiers/order"):
        graph.predict(a=a, b=b)
    with pytest.raises(ValueError, match="sample"):
        graph.predict(a=TRAIN[:1], b=TRAIN[:2])
    with pytest.raises(ValueError, match="Input mismatch"):
        graph.predict(a=TRAIN[:1])
    assert graph.predict(
        a=nn.SampleData(TRAIN[:1], ["p"]), b=nn.SampleData(TRAIN[:1], ["p"])
    ).shape == (1,)


class Center(nn.Block):
    input_kinds = {"data": nn.FEATURES}

    def forward(self, data):
        return data - data.mean(axis=0)

    def fit_forward(self, data):
        self.mean_ = data.mean(axis=0)
        return data - self.mean_

    def predict_forward(self, data):
        return data - self.mean_


def test_custom_protocol_and_explicit_stateless_functions():
    x = nn.Input("x")
    centered = Center()(x)
    doubled = nn.Function(lambda data: data * 2, {"data": nn.FEATURES}, inductive=True)(centered)
    graph = nn.Graph({"x": x}, doubled).fit(x=TRAIN)
    assert_allclose(graph.transform(x=TRAIN[:1]), (TRAIN[:1] - TRAIN.mean(axis=0)) * 2)
    unsupported = nn.Function(lambda data: data - data.mean(axis=0), {"data": nn.FEATURES})
    with pytest.raises(RuntimeError, match="inductive=True"):
        nn.Graph({"x": x}, unsupported(x)).fit(x=TRAIN)

    class ForwardOnly(nn.Block):
        input_kinds = {"data": nn.FEATURES}

        def forward(self, data):
            return data

    with pytest.raises(TypeError, match="no inductive implementation"):
        nn.Graph({"x": x}, ForwardOnly()(x)).fit(x=TRAIN)


def test_custom_fusion_cannot_silently_use_the_base_fitted_rule():
    class WeightedFusion(nn.Fusion):
        def forward(self, *partitions):
            return 3 * super().forward(*partitions)

    x = nn.Input("x", nn.LABELS)
    graph = nn.Graph({"x": x}, WeightedFusion()(x))
    labels = np.array([0, 0, 1, 1])
    assert graph(x=labels).shape == (4, 4)
    with pytest.raises(TypeError, match="Custom fusion requires"):
        graph.fit(x=labels)


def test_output_selection_and_api_errors():
    x = nn.Input("x")
    a, b = nn.Hierarchical(2)(x), nn.Hierarchical(3)(x)
    graph = nn.Graph({"x": x}, {"first": a, "second": b})
    with pytest.raises(NotFittedError):
        graph.predict(x=TRAIN[:1])
    with pytest.raises(NotFittedError):
        _ = graph.labels_
    graph.fit(x=TRAIN)
    with pytest.raises(ValueError, match="Choose"):
        graph.predict(x=TRAIN[:1])
    assert graph.predict(x=TRAIN[:1], output="second").shape == (1,)
    with pytest.raises(ValueError, match="Choose"):
        graph.predict(x=TRAIN[:1], output="missing")
    with pytest.raises(TypeError, match="mapping"):
        graph.fit({"x": TRAIN}, x=TRAIN)
    with pytest.raises(ValueError, match="assignment"):
        graph.fit(x=TRAIN, assignment="unknown")
    with pytest.raises(ValueError, match="Assignment"):
        graph.fit(x=TRAIN, assignment={"missing": "nearest"})
    with pytest.raises(ValueError, match="consensus"):
        graph.fit(x=TRAIN, consensus="new-clusters")


def test_prediction_rejects_custom_blocks_inventing_new_label_ids():
    class BadPredict(nn.Block):
        input_kinds, output_kind = {"data": nn.FEATURES}, nn.LABELS

        def fit_forward(self, data):
            return np.zeros(len(data), dtype=int)

        def predict_forward(self, data):
            return np.ones(len(data), dtype=int)

    x = nn.Input("x")
    graph = nn.Graph({"x": x}, BadPredict()(x)).fit(x=TRAIN)
    with pytest.raises(RuntimeError, match="not observed"):
        graph.predict(x=TRAIN[:1])
