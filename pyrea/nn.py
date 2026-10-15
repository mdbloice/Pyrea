"""Executable ensemble blocks, with eager and symbolic calls.

Use Module.forward for reusable compositions and Block.forward for custom
operations. Symbolic Input calls construct a Graph without running clustering.
"""

import copy
import inspect

import numpy as np

from .core import clusterer as make_clusterer
from .graph import (
    AFFINITY,
    DISTANCE,
    FEATURES,
    LABELS,
    Block,
    Graph,
    Input,
    Module,
    Node,
    SampleData,
)
from .structure import _label_vectors


__all__ = [
    "Module",
    "Block",
    "Graph",
    "Input",
    "Node",
    "SampleData",
    "Cluster",
    "Hierarchical",
    "Fusion",
    "Consensus",
    "Function",
    "AffinityToDistance",
    "FEATURES",
    "LABELS",
    "DISTANCE",
    "AFFINITY",
]


class Cluster(Block):
    """Wrap a Pyrea clustering method, constructing a fresh clusterer per call.

    input_kind is features, distance (hierarchical/DBSCAN/OPTICS), or affinity
    (spectral). All remaining keyword arguments configure the clustering method.
    """

    output_kind = LABELS

    def __init__(self, clusterer="hierarchical", input_kind=FEATURES, name=None, **params):
        supported = {
            "hierarchical": (FEATURES, DISTANCE),
            "spectral": (FEATURES, AFFINITY),
            "dbscan": (FEATURES, DISTANCE),
            "optics": (FEATURES, DISTANCE),
        }
        if clusterer not in supported or input_kind not in supported[clusterer]:
            raise ValueError(
                "Unsupported clusterer/input kind: {}/{}".format(clusterer, input_kind)
            )
        self.clusterer = clusterer
        self.input_kind = input_kind
        self.name = name
        self.params = copy.deepcopy(params)
        if clusterer in ("hierarchical", "spectral"):
            self.params.setdefault("n_clusters", 2)
        if clusterer == "hierarchical":
            self.params.setdefault("method", "ward")
        if input_kind == FEATURES and (
            self.params.get("affinity") == "precomputed"
            or self.params.get("metric") == "precomputed"
        ):
            raise ValueError("Declare the precomputed input_kind explicitly")
        if (
            input_kind == AFFINITY
            and self.params.get("affinity", "precomputed") != "precomputed"
            or input_kind == DISTANCE
            and clusterer in ("dbscan", "optics")
            and self.params.get("metric", "precomputed") != "precomputed"
        ):
            raise ValueError("Precomputed inputs require precomputed affinity/metric")
        self._make_clusterer()  # Reject unsupported configuration before execution.

    @property
    def input_kinds(self):
        """Expected kind of the data input."""
        return {"data": self.input_kind}

    def _make_clusterer(self):
        return make_clusterer(
            self.clusterer, precomputed=self.input_kind != FEATURES, **copy.deepcopy(self.params)
        )

    def forward(self, data):
        labels = np.asarray(self._make_clusterer().execute(np.asarray(data)))
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels[:, 0]
        return labels

    def get_params(self):
        return dict(clusterer=self.clusterer, **copy.deepcopy(self.params))

    def set_params(self, **params):
        clusterer = params.pop("clusterer", self.clusterer)
        updated = copy.deepcopy(self.params) if clusterer == self.clusterer else {}
        updated.update(params)
        validated = Cluster(clusterer, input_kind=self.input_kind, name=self.name, **updated)
        self.clusterer, self.params = validated.clusterer, validated.params
        return self


class Hierarchical(Cluster):
    """Convenience block for hierarchical clustering."""

    def __init__(self, n_clusters=2, method="ward", input_kind=FEATURES, name=None, **params):
        super().__init__(
            "hierarchical",
            input_kind=input_kind,
            name=name,
            n_clusters=n_clusters,
            method=method,
            **params,
        )


class Fusion(Block):
    """Fuse partitions using disagreement, agreement, or strict consensus.

    Disagreement produces distances; agreement and consensus produce affinities.
    normalize divides agreement/disagreement counts by the number of partitions.
    """

    input_kinds = {"partitions": LABELS}

    def __init__(self, method="disagreement", normalize=False, name=None):
        if method not in ("disagreement", "agreement", "consensus"):
            raise ValueError("Unknown fusion method {!r}".format(method))
        self.method = method
        self.normalize = normalize
        self.name = name

    @property
    def output_kind(self):
        """Distances for disagreement, affinities for agreement or consensus."""
        return DISTANCE if self.method == "disagreement" else AFFINITY

    def forward(self, *partitions):
        partitions = _label_vectors(partitions)
        if self.method == "consensus":
            labels = Consensus()(*partitions)
            return (labels[:, None] == labels[None, :]).astype(int)
        n_samples = len(partitions[0])
        result = np.zeros((n_samples, n_samples), dtype=int)
        for labels in partitions:
            same = labels[:, None] == labels[None, :]
            result += ~same if self.method == "disagreement" else same
        np.fill_diagonal(result, 0)
        return result / len(partitions) if self.normalize else result

    def get_params(self):
        return {"method": self.method, "normalize": self.normalize}

    def set_params(self, **params):
        unknown = set(params) - {"method", "normalize"}
        if unknown:
            raise ValueError("Unknown fusion parameters: {}".format(sorted(unknown)))
        updated = self.get_params()
        updated.update(params)
        validated = Fusion(name=self.name, **updated)
        if validated.output_kind != self.output_kind:
            raise ValueError("Changing distance/affinity semantics requires rebuilding the graph")
        self.method, self.normalize = validated.method, validated.normalize
        return self


class Consensus(Block):
    """Return the common refinement of partitions, using labels starting at one."""

    input_kinds = {"partitions": LABELS}
    output_kind = LABELS

    def __init__(self, name=None):
        self.name = name

    def forward(self, *partitions):
        partitions = _label_vectors(partitions)
        groups = {}
        result = []
        for memberships in zip(*partitions):
            if memberships not in groups:
                groups[memberships] = len(groups) + 1
            result.append(groups[memberships])
        return np.asarray(result, dtype=int)


class Function(Block):
    """Adapt a callable with named inputs into a block.

    Declare input_kinds by callable argument name. Custom functions must treat
    inputs as read-only and preserve sample order and sample count.
    """

    def __init__(self, function, input_kinds, output_kind=FEATURES, name=None):
        if not callable(function):
            raise TypeError("function must be callable")
        self.function = function
        self.input_kinds = dict(input_kinds)
        self.output_kind = output_kind
        self.name = name

    def _signature(self):
        return inspect.signature(self.function)

    def forward(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class AffinityToDistance(Block):
    """Explicitly convert affinity A to max(A)-A, optionally scaling to [0, 1]."""

    input_kinds = {"affinity": AFFINITY}
    output_kind = DISTANCE

    def __init__(self, normalize=True, name=None):
        self.normalize = normalize
        self.name = name

    def forward(self, affinity):
        affinity = np.asarray(affinity)
        maximum = affinity.max()
        distance = maximum - affinity.astype(float)
        if self.normalize and maximum:
            distance /= maximum
        np.fill_diagonal(distance, 0)
        return distance

    def get_params(self):
        return {"normalize": self.normalize}

    def set_params(self, **params):
        if set(params) - {"normalize"}:
            raise ValueError("Only normalize can be configured")
        self.normalize = params.get("normalize", self.normalize)
        return self
