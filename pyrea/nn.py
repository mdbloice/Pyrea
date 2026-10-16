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
    "Transform",
    "Standardize",
    "Estimator",
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

    def __init__(self, function, input_kinds, output_kind=FEATURES, name=None, inductive=False):
        if not callable(function):
            raise TypeError("function must be callable")
        if not isinstance(inductive, bool):
            raise TypeError("inductive must be a boolean")
        self.function = function
        self.input_kinds = dict(input_kinds)
        self.output_kind = output_kind
        self.name = name
        self.inductive = inductive

    def _signature(self):
        return inspect.signature(self.function)

    def forward(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    def fit_forward(self, *args, **kwargs):
        if not self.inductive:
            raise TypeError(
                "Function requires inductive=True for a stateless row-wise operation; "
                "use Transform for fitted preprocessing"
            )
        return self.forward(*args, **kwargs)

    def predict_forward(self, *args, **kwargs):
        if not self.inductive:
            raise TypeError("Function requires inductive=True for prediction")
        return self.forward(*args, **kwargs)


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

    def fit_forward(self, affinity):
        self.maximum_ = float(np.asarray(affinity).max())
        return self.forward(affinity)

    def predict_forward(self, affinity):
        # Cross-affinities need not have a diagonal. Saturate above the fitted
        # maximum, without learning a new maximum from the prediction batch.
        distance = np.maximum(self.maximum_ - np.asarray(affinity, dtype=float), 0)
        if self.normalize and self.maximum_:
            distance /= self.maximum_
        return distance

    def set_params(self, **params):
        if set(params) - {"normalize"}:
            raise ValueError("Only normalize can be configured")
        self.normalize = params.get("normalize", self.normalize)
        return self


class Transform(Block):
    """Retain a fitted sklearn-style feature transformer for inductive use.

    Eager forward fits a fresh copy, as other transductive blocks do. Graph.fit
    retains a separate fitted copy for every executable call.
    """

    input_kinds = {"data": FEATURES}
    output_kind = FEATURES

    def __init__(self, transformer, name=None):
        if not callable(getattr(transformer, "fit", None)) or not callable(
            getattr(transformer, "transform", None)
        ):
            raise TypeError("Transform requires fit() and transform()")
        self.transformer, self.name = transformer, name

    def _fit(self, data):
        estimator = copy.deepcopy(self.transformer)
        data = copy.deepcopy(data)
        if callable(getattr(estimator, "fit_transform", None)):
            output = estimator.fit_transform(data)
        else:
            output = estimator.fit(data).transform(data)
        return estimator, output

    def forward(self, data):
        return self._fit(data)[1]

    def fit_forward(self, data):
        self.transformer_, output = self._fit(data)
        return output

    def predict_forward(self, data):
        return self.transformer_.transform(copy.deepcopy(data))

    def get_params(self):
        getter = getattr(self.transformer, "get_params", None)
        return copy.deepcopy(getter()) if callable(getter) else {}

    def set_params(self, **params):
        if params:
            self.transformer.set_params(**params)
        return self


class Standardize(Transform):
    """StandardScaler with retained training statistics during Graph.predict."""

    def __init__(self, name=None, **params):
        from sklearn.preprocessing import StandardScaler

        super().__init__(StandardScaler(**params), name=name)


class Estimator(Block):
    """Adapt a clustering estimator with native fit/predict (e.g. KMeans).

    The configured estimator is copied per training call. No fitting occurs
    during prediction. Native labels must belong to its training label set.
    """

    output_kind = LABELS

    def __init__(self, estimator, input_kind=FEATURES, name=None):
        if not callable(getattr(estimator, "fit", None)) or not callable(
            getattr(estimator, "predict", None)
        ):
            raise TypeError("Estimator requires native fit() and predict()")
        if input_kind not in (FEATURES, DISTANCE, AFFINITY):
            raise ValueError("Estimator input must be features, distances, or affinities")
        self.estimator, self.input_kind, self.name = estimator, input_kind, name

    @property
    def input_kinds(self):
        """Accepted semantic kind for the estimator's data input."""
        return {"data": self.input_kind}

    def _fit(self, data):
        estimator = copy.deepcopy(self.estimator)
        data = copy.deepcopy(data)
        if callable(getattr(estimator, "fit_predict", None)):
            output = estimator.fit_predict(data)
        else:
            estimator.fit(data)
            output = getattr(estimator, "labels_", None)
            if output is None:
                output = estimator.predict(data)
        return estimator, output

    def forward(self, data):
        return self._fit(data)[1]

    def fit_forward(self, data):
        self.estimator_, output = self._fit(data)
        return output

    def predict_forward(self, data):
        return self.estimator_.predict(copy.deepcopy(data))

    def get_params(self):
        getter = getattr(self.estimator, "get_params", None)
        return copy.deepcopy(getter()) if callable(getter) else {}

    def set_params(self, **params):
        if params:
            self.estimator.set_params(**params)
        return self
