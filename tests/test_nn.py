import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

import pyrea
from pyrea import models, nn


@pytest.mark.parametrize(
    "method", ["single", "complete", "average", "weighted", "centroid", "median", "ward", "ward2"]
)
@pytest.mark.parametrize("precomputed", [False, True])
def test_hierarchical_methods_match_scipy(method, precomputed):
    data = np.random.RandomState(5).normal(size=(18, 4))
    distance = pdist(data)
    expected = hierarchy.cut_tree(
        hierarchy.linkage(
            distance**2 if method == "ward2" else distance,
            method="ward" if method == "ward2" else method,
            optimal_ordering=True,
        ),
        n_clusters=3,
    )[:, 0]
    block = nn.Hierarchical(
        3, method, input_kind=nn.DISTANCE if precomputed else nn.FEATURES, optimal_ordering=True
    )
    assert_array_equal(block(squareform(distance) if precomputed else data), expected)


def test_selected_linkage_changes_partition():
    data = np.random.RandomState(5).normal(size=(18, 4))
    assert not np.array_equal(nn.Hierarchical(3, "single")(data), nn.Hierarchical(3, "ward")(data))


def test_dbscan_returns_labels_and_matches_sklearn():
    data, _ = make_blobs(n_samples=30, centers=2, random_state=4)
    assert_array_equal(nn.Cluster("dbscan", eps=1.5)(data), DBSCAN(eps=1.5).fit_predict(data))
    assert_array_equal(
        pyrea.clusterer("dbscan", eps=1.5).execute(data), DBSCAN(eps=1.5).fit_predict(data)
    )


@pytest.mark.parametrize("method", ["agreement", "disagreement", "consensus"])
def test_fusion_matches_legacy_and_normalization(method):
    partitions = [np.array([0, 0, 1, 1]), np.array([5, 5, 5, 7])]
    expected = pyrea.fuser(method).execute(partitions)
    assert_array_equal(nn.Fusion(method)(*partitions), expected)
    if method != "consensus":
        assert_allclose(nn.Fusion(method, normalize=True)(*partitions), expected / 2)


def test_consensus_is_strict_and_label_invariant():
    expected = [1, 1, 2, 3]
    assert_array_equal(nn.Consensus()([0, 0, 1, 1], [5, 5, 5, 7]), expected)
    assert_array_equal(nn.Consensus()(["a", "a", "b", "b"], [9, 9, 9, 3]), expected)
    assert_array_equal(nn.Consensus()([8, 8, 2]), [1, 1, 2])
    with pytest.raises(ValueError, match="At least one"):
        nn.Consensus()()
    with pytest.raises(ValueError, match="same samples"):
        nn.Fusion()([0, 1], [0, 1, 2])


def test_legacy_consensus_handles_scipy_column_labels():
    partitions = [np.array([[0], [0], [1], [1]]), np.array([[5], [5], [5], [7]])]
    expected = np.array([1, 1, 2, 3])
    assert_array_equal(pyrea.consensus(partitions), expected)
    assert_array_equal(
        pyrea.fuser("consensus").execute(partitions),
        (expected[:, None] == expected[None, :]).astype(int),
    )
    for method in ["agreement", "disagreement"]:
        assert_array_equal(pyrea.fuser(method).execute(partitions), nn.Fusion(method)(*partitions))


def test_affinity_distance_conversion_and_spectral_graph():
    x, y = nn.Input("x"), nn.Input("y")
    first = nn.Hierarchical()(x)
    second = nn.Hierarchical()(y)
    affinity = nn.Fusion("agreement")(first, second)
    distance = nn.AffinityToDistance()(affinity)
    spectral = nn.Cluster("spectral", input_kind=nn.AFFINITY, random_state=42)(affinity)
    hierarchical = nn.Hierarchical(input_kind=nn.DISTANCE)(distance)
    model = nn.Graph({"x": x, "y": y}, {"spectral": spectral, "hierarchical": hierarchical})
    data, truth = make_blobs(
        n_samples=30, centers=[[-4, -4], [4, 4]], cluster_std=0.2, random_state=42
    )
    output = model(x=data, y=data[:, ::-1])
    for labels in output.values():
        assert_array_equal(labels[:, None] == labels[None, :], truth[:, None] == truth[None, :])
    assert_array_equal(nn.AffinityToDistance()(np.zeros((3, 3))), np.zeros((3, 3)))


def test_parameter_trials_cannot_silently_change_connection_semantics():
    a = nn.Input("a", nn.LABELS)
    model = nn.Graph({"a": a}, nn.Fusion(name="fuse")(a))
    with pytest.raises(ValueError, match="semantics"):
        model.with_params(fuse__method="agreement")
    for method, kind in [("spectral", nn.DISTANCE), ("hierarchical", nn.AFFINITY)]:
        with pytest.raises(ValueError, match="Unsupported"):
            nn.Cluster(method, input_kind=kind)
    with pytest.raises(ValueError, match="precomputed"):
        nn.Cluster("spectral", affinity="precomputed")


@pytest.mark.parametrize("n_views", [1, 2, 4])
@pytest.mark.parametrize("k_final", [None, 3])
def test_parea1_graph_matches_existing_helper(n_views, k_final):
    rng = np.random.RandomState(7)
    data = [rng.normal(size=(24, 4)) for _ in range(n_views)]
    names = ["view{}".format(i) for i in range(n_views)]
    model = models.parea_1(
        names,
        methods=("single", "average"),
        n_clusters=3,
        precomputed_methods=("complete", "ward"),
        k_final=k_final,
        return_matrix=True,
    )
    expected = pyrea.parea_1(
        data,
        c_1_method="single",
        c_2_method="average",
        c_1_k=3,
        c_2_k=3,
        c_1_pre_method="complete",
        c_2_pre_method="ward",
        c_1_pre_k=3,
        c_2_pre_k=3,
        k_final=k_final,
    )
    output = model(**dict(zip(names, data)))
    assert_array_equal(output["labels"], expected)
    assert output["distance"].shape == (24, 24)


@pytest.mark.parametrize("k_final", [None, 3])
def test_parea2_graph_matches_existing_helper(k_final):
    rng = np.random.RandomState(8)
    data = [rng.normal(size=(24, 4)) for _ in range(3)]
    model = models.parea_2(["a", "b", "c"], methods=("ward", "complete", "single"), k_final=k_final)
    assert_array_equal(model(a=data[0], b=data[1], c=data[2]), pyrea.parea_2(data, k_final=k_final))


@pytest.mark.parametrize("n_views", [1, 2, 4])
def test_parea2_supports_arbitrary_view_and_final_branch_counts(n_views):
    rng = np.random.RandomState(9)
    data = {"view{}".format(i): rng.normal(size=(20, i + 2)) for i in range(n_views)}
    model = models.parea_2(data, precomputed_methods=("ward", "single", "average"))
    assert model(**data).shape == (20,)
    assert sum(name.startswith("recluster_") for name in model.graph.nodes) == 3


@pytest.mark.parametrize(
    "builder,kwargs",
    [
        (models.parea_1, {"methods": ("ward",)}),
        (models.parea_2, {"methods": ("ward",)}),
        (models.parea_2, {"precomputed_methods": ()}),
    ],
)
def test_builder_rejects_inconsistent_method_lists(builder, kwargs):
    with pytest.raises(ValueError):
        builder(["x", "y"], **kwargs)


@pytest.mark.parametrize("builder", [models.parea_1, models.parea_2])
@pytest.mark.parametrize("inputs", [[], ["x", "x"]])
def test_builder_requires_unique_input_names(builder, inputs):
    with pytest.raises(ValueError):
        builder(inputs)


def test_search_can_switch_clustering_family_without_retaining_old_parameters():
    source = nn.Input("data")
    model = nn.Graph({"data": source}, nn.Cluster(name="cluster")(source))
    trial = model.with_params(cluster__clusterer="dbscan", cluster__eps=1.5)
    data, _ = make_blobs(n_samples=30, centers=2, random_state=4)
    assert_array_equal(trial(data), DBSCAN(eps=1.5).fit_predict(data))
    assert "cluster__method" not in trial.get_params()
    assert model.get_params()["cluster__clusterer"] == "hierarchical"


@pytest.mark.parametrize(
    "method,kind,params",
    [
        ("spectral", nn.AFFINITY, {"affinity": "rbf"}),
        ("dbscan", nn.DISTANCE, {"metric": "euclidean"}),
    ],
)
def test_precomputed_parameters_cannot_override_connection_kinds(method, kind, params):
    with pytest.raises(ValueError, match="require precomputed"):
        nn.Cluster(method, input_kind=kind, **params)


@pytest.mark.parametrize("builder", [models.parea_1, models.parea_2])
def test_builder_accepts_single_input_name(builder):
    assert list(builder("rna").inputs) == ["rna"]


def test_legacy_fusion_accepts_array_of_partitions():
    partitions = np.array([[0, 0, 1], [0, 1, 1]])
    assert_array_equal(
        pyrea.fuser("agreement").execute(partitions), nn.Fusion("agreement")(*partitions)
    )
