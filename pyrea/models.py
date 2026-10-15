"""Predefined ensemble topologies built entirely through the public graph API.

These use Python disagreement fusion; HCfused is not part of these workflows.
"""

from . import nn


def _inputs(input_names):
    names = [input_names] if isinstance(input_names, str) else list(input_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("Supply at least one unique input name")
    return {name: nn.Input(name) for name in names}


def parea_1(
    input_names,
    methods=("ward", "complete"),
    n_clusters=2,
    precomputed_methods=None,
    k_final=None,
    return_matrix=False,
):
    """Build the two-branch Parea 1 DAG for any positive number of views.

    Initial clustering blocks share parameters across views in each branch;
    precomputed blocks are reused at the intermediate and final layers.
    return_matrix exposes labels and the final distance matrix as named outputs.
    """
    inputs = _inputs(input_names)
    methods = tuple(methods)
    precomputed_methods = methods if precomputed_methods is None else tuple(precomputed_methods)
    if len(methods) != 2 or len(precomputed_methods) != 2:
        raise ValueError("Parea 1 requires exactly two methods at each stage")
    branches = []
    reclusterers = []
    for i, (method, precomputed) in enumerate(zip(methods, precomputed_methods), 1):
        cluster = nn.Hierarchical(n_clusters, method, name="branch_{}".format(i))
        fusion = nn.Fusion(name="fusion_{}".format(i))
        recluster = nn.Hierarchical(
            n_clusters, precomputed, input_kind=nn.DISTANCE, name="recluster_{}".format(i)
        )
        branches.append(recluster(fusion(*(cluster(value) for value in inputs.values()))))
        reclusterers.append(recluster)
    distance = nn.Fusion(name="fusion")(*branches)
    if k_final is None:
        labels = nn.Consensus(name="consensus")(*(cluster(distance) for cluster in reclusterers))
    else:
        labels = nn.Hierarchical(
            k_final, precomputed_methods[0], input_kind=nn.DISTANCE, name="final"
        )(distance)
    outputs = {"labels": labels, "distance": distance} if return_matrix else labels
    return nn.Graph(inputs, outputs)


def parea_2(
    input_names,
    methods=None,
    n_clusters=2,
    precomputed_methods=None,
    k_final=None,
    return_matrix=False,
):
    """Build Parea 2 with one initial method per view and arbitrary final branches."""
    inputs = _inputs(input_names)
    methods = ("ward",) * len(inputs) if methods is None else tuple(methods)
    if len(methods) != len(inputs):
        raise ValueError("Supply one initial method per input view")
    precomputed_methods = methods if precomputed_methods is None else tuple(precomputed_methods)
    if not precomputed_methods:
        raise ValueError("Supply at least one precomputed method")
    partitions = [
        nn.Hierarchical(n_clusters, method, name="cluster_{}".format(i))(value)
        for i, (value, method) in enumerate(zip(inputs.values(), methods), 1)
    ]
    distance = nn.Fusion(name="fusion")(*partitions)
    if k_final is None:
        final = [
            nn.Hierarchical(
                n_clusters, method, input_kind=nn.DISTANCE, name="recluster_{}".format(i)
            )(distance)
            for i, method in enumerate(precomputed_methods, 1)
        ]
        labels = nn.Consensus(name="consensus")(*final)
    else:
        labels = nn.Hierarchical(
            k_final, precomputed_methods[0], input_kind=nn.DISTANCE, name="final"
        )(distance)
    outputs = {"labels": labels, "distance": distance} if return_matrix else labels
    return nn.Graph(inputs, outputs)
