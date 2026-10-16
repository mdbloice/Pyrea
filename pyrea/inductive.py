"""Fitted execution of ensemble DAGs with frozen training references.

Non-native clustering uses explicit label transfer, not an assertion that the
underlying transductive algorithm has acquired a native out-of-sample extension.
"""

import copy
from collections.abc import Mapping

import networkx as nx
import numpy as np
from sklearn.metrics import pairwise_distances

from .graph import AFFINITY, DISTANCE, FEATURES, LABELS, Block, SampleData, _validate_value


def input_mapping(inputs, views):
    if inputs is not None:
        if views or not isinstance(inputs, Mapping):
            raise TypeError("Supply one input mapping or named views, not both")
        return dict(inputs)
    return dict(views)


def _value(value, kind, name, n_samples=None, n_train=None):
    if n_train is None or kind not in (DISTANCE, AFFINITY):
        return _validate_value(value, kind, name, n_samples)
    value = np.asarray(value)
    if value.ndim != 2 or not value.shape[0] or value.shape[1] != n_train:
        raise ValueError("{} must have shape (new samples, {})".format(name, n_train))
    if n_samples is not None and len(value) != n_samples:
        raise ValueError("{} sample count does not match other inputs".format(name))
    if not np.isfinite(value).all() or (value < 0).any():
        raise ValueError("{} requires finite, nonnegative cross-pairwise values".format(name))
    return value


def _metadata(value):
    ids, columns = None, None
    if isinstance(value, SampleData):
        ids, value = value.sample_ids, value.values
    elif hasattr(value, "index") and hasattr(value, "columns"):
        ids, columns = tuple(value.index), tuple(value.columns)
    return value, ids, columns


class _CustomState:
    def __init__(self, block):
        if type(block).predict_forward is Block.predict_forward:
            raise TypeError("Custom inductive blocks must implement predict_forward")
        self.block = block

    def fit(self, *args, **kwargs):
        return self.block.fit_forward(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.block.predict_forward(*args, **kwargs)


class _ClusterState:
    def __init__(self, block, assignment):
        self.block, self.assignment = block, assignment

    def fit(self, data):
        self.data = np.array(data, copy=True)
        self.labels = np.asarray(self.block.forward(self.data)).reshape(-1).copy()
        self.groups = list(dict.fromkeys(self.labels.tolist()))
        self.members = [np.flatnonzero(self.labels == label) for label in self.groups]
        if self.block.input_kind == FEATURES:
            params = self.block.params
            self.metric = (
                params.get("distance_metric", "euclidean")
                if (self.block.clusterer == "hierarchical")
                else params.get("metric", "euclidean")
            )
            self.metric_params = copy.deepcopy(params.get("metric_params") or {})
            if self.metric in ("minkowski", "wminkowski") and params.get("p") is not None:
                self.metric_params.setdefault("p", params["p"])
            # SciPy otherwise estimates these from X and Y together. Freeze
            # them from training data to make prediction batch-independent.
            if self.metric == "seuclidean" and "V" not in self.metric_params:
                self.metric_params["V"] = np.var(self.data, axis=0, ddof=1)
            if self.metric == "mahalanobis" and "VI" not in self.metric_params:
                self.metric_params["VI"] = np.linalg.inv(np.atleast_2d(np.cov(self.data.T)))
        if self.assignment == "medoid":
            cost = self._cost(self.data)
            self.medoids = np.array(
                [
                    members[np.argmin(cost[np.ix_(members, members)].mean(axis=1))]
                    for members in self.members
                ]
            )
        return self.labels.copy()

    def _cost(self, data):
        if self.block.input_kind == FEATURES:
            return pairwise_distances(data, self.data, metric=self.metric, **self.metric_params)
        if self.block.input_kind == AFFINITY:
            return -np.asarray(data, dtype=float)
        return np.asarray(data)

    def predict(self, data):
        cost = self._cost(data)
        if not np.isfinite(cost).all():
            raise ValueError("Assignment distances/similarities must be finite")
        if self.assignment == "nearest":
            return self.labels[np.argmin(cost, axis=1)]
        if self.assignment == "medoid":
            return np.asarray(self.groups)[np.argmin(cost[:, self.medoids], axis=1)]
        average = np.column_stack([cost[:, members].mean(axis=1) for members in self.members])
        return np.asarray(self.groups)[np.argmin(average, axis=1)]


class _FusionState:
    def __init__(self, block):
        self.block = block

    def fit(self, *partitions):
        self.partitions = [np.asarray(labels).reshape(-1).copy() for labels in partitions]
        return self.block.forward(*partitions)

    def predict(self, *partitions):
        same = [
            np.asarray(new)[:, None] == old[None, :]
            for new, old in zip(partitions, self.partitions)
        ]
        if self.block.method == "consensus":
            return np.logical_and.reduce(same).astype(int)
        values = [
            ~comparison if self.block.method == "disagreement" else comparison
            for comparison in same
        ]
        result = np.sum(values, axis=0, dtype=int)
        return result / len(partitions) if self.block.normalize else result


class _ConsensusState:
    def __init__(self, block, unseen):
        self.block, self.unseen = block, unseen

    def fit(self, *partitions):
        labels = self.block.forward(*partitions)
        self.mapping = dict(zip(zip(*partitions), labels.tolist()))
        # All partitions belonging to a learned tuple share the same label.
        self.patterns = list(self.mapping)
        self.labels = np.array(list(self.mapping.values()))
        return labels

    def predict(self, *partitions):
        result = []
        for pattern in zip(*partitions):
            if pattern in self.mapping:
                result.append(self.mapping[pattern])
            elif self.unseen == "raise":
                raise ValueError("Unseen consensus membership tuple: {!r}".format(pattern))
            else:
                mismatch = [sum(a != b for a, b in zip(pattern, known)) for known in self.patterns]
                result.append(self.labels[np.argmin(mismatch)])
        return np.asarray(result)


def _adapter(block, assignment, consensus):
    from .nn import Cluster, Consensus, Fusion

    block = copy.deepcopy(block)
    if type(block).fit_forward is not Block.fit_forward:
        return _CustomState(block)
    if isinstance(block, Cluster):
        return _ClusterState(block, assignment)
    if isinstance(block, Fusion):
        if type(block).forward is not Fusion.forward:
            raise TypeError("Custom fusion requires fit_forward and predict_forward")
        return _FusionState(block)
    if isinstance(block, Consensus):
        if type(block).forward is not Consensus.forward:
            raise TypeError("Custom consensus requires fit_forward and predict_forward")
        return _ConsensusState(block, consensus)
    raise TypeError(
        "{} has no inductive implementation; implement fit_forward and "
        "predict_forward or use nn.Transform/nn.Estimator".format(type(block).__name__)
    )


class FittedGraph:
    """Private fitted snapshot used by Graph.fit/predict/transform."""

    def __init__(self, graph, inputs, assignment, consensus):
        from .nn import Cluster

        if consensus not in ("nearest", "raise"):
            raise ValueError("consensus must be 'nearest' or 'raise'")
        choices = ("nearest", "average", "medoid")
        if isinstance(assignment, Mapping):
            cluster_names = {
                name for name, block in graph.named_blocks() if isinstance(block, Cluster)
            }
            if set(assignment) - cluster_names:
                raise ValueError("Assignment overrides must name clustering blocks")
            if any(rule not in choices for rule in assignment.values()):
                raise ValueError("assignment must be nearest, average, or medoid")
            rules = dict(assignment)
        else:
            if assignment not in choices:
                raise ValueError("assignment must be nearest, average, or medoid")
            rules = {name: assignment for name, _ in graph.named_blocks()}
        self.graph = graph
        self.schema = {}
        self.n_train = None
        self.training_ids = None
        self.order = list(nx.topological_sort(graph._graph))
        block_names = {id(block): name for name, block in graph.named_blocks()}
        self.states = {}
        for name in self.order:
            block = graph._nodes[name].block
            if block is not None:
                try:
                    self.states[name] = _adapter(
                        block, rules.get(block_names[id(block)], "nearest"), consensus
                    )
                except Exception as error:
                    raise TypeError("Cannot fit node {!r}: {}".format(name, error)) from error
        results = self._inputs(inputs, training=True)
        self.training_shapes, self.label_sets = {}, {}
        for name in self.order:
            node = graph._nodes[name]
            if node.block is not None:
                try:
                    args = [results[source] for source in node.args]
                    kwargs = {port: results[source] for port, source in node.kwargs.items()}
                    results[node] = _value(
                        self.states[name].fit(*args, **kwargs), node.kind, name, self.n_train
                    )
                except Exception as error:
                    raise RuntimeError(
                        "Fitting node {!r} failed: {}".format(name, error)
                    ) from error
            self.training_shapes[name] = results[node].shape
            if node.kind == LABELS:
                self.label_sets[name] = np.unique(results[node])
        self.training_output = copy.deepcopy(self._output(results))

    def _inputs(self, inputs, training=False):
        if set(inputs) != set(self.graph.inputs):
            raise ValueError(
                "Input mismatch: expected {}, received {}".format(
                    list(self.graph.inputs), list(inputs)
                )
            )
        results, count, reference_ids = {}, None, None
        for name, node in self.graph.inputs.items():
            raw, ids, columns = _metadata(inputs[name])
            value = _value(raw, node.kind, name, count, None if training else self.n_train)
            count = len(value)
            if ids is not None:
                if len(ids) != count or len(set(ids)) != count:
                    raise ValueError("{} requires unique sample identifiers".format(name))
                if reference_ids is not None and ids != reference_ids:
                    raise ValueError(
                        "{} sample identifiers/order do not match other inputs".format(name)
                    )
                reference_ids = ids
            if training:
                if node.kind in (DISTANCE, AFFINITY) and columns is not None and columns != ids:
                    raise ValueError(
                        "{} training matrix columns must match its sample index".format(name)
                    )
                self.schema[name] = {"shape": value.shape, "columns": columns}
                value = np.array(value, copy=True)
            else:
                schema = self.schema[name]
                if node.kind == FEATURES:
                    if value.shape[1] != schema["shape"][1]:
                        raise ValueError("{} feature count differs from training".format(name))
                    if (
                        columns is not None
                        and schema["columns"] is not None
                        and columns != schema["columns"]
                    ):
                        raise ValueError("{} feature names/order differ from training".format(name))
                elif node.kind in (DISTANCE, AFFINITY) and columns is not None:
                    if self.training_ids is not None and columns != self.training_ids:
                        raise ValueError(
                            "{} columns must follow the training sample order".format(name)
                        )
                elif node.kind == LABELS and not np.isin(value, self.label_sets[name]).all():
                    raise ValueError("{} contains labels not observed in training".format(name))
            results[node] = value
        if training:
            self.n_train, self.training_ids = count, reference_ids
        return results

    def _output(self, results):
        outputs = self.graph.outputs
        if isinstance(outputs, dict):
            return {name: results[node] for name, node in outputs.items()}
        return results[outputs]

    def label_output(self, output):
        outputs = self.graph.outputs
        if not isinstance(outputs, dict):
            if output is not None or outputs.kind != LABELS:
                raise ValueError("predict requires a label output; use transform for other outputs")
            return None
        if output is None:
            label_keys = [key for key, node in outputs.items() if node.kind == LABELS]
            output = (
                "labels"
                if "labels" in label_keys
                else (label_keys[0] if len(label_keys) == 1 else None)
            )
        if output not in outputs or outputs[output].kind != LABELS:
            raise ValueError(
                "Choose a declared label output with output=; use transform for matrices"
            )
        return output

    def transform(self, inputs, return_intermediates=False):
        results = self._inputs(inputs)
        n_new = len(next(iter(results.values())))
        for name in self.order:
            node = self.graph._nodes[name]
            if node.block is None:
                continue
            try:
                args = [results[source] for source in node.args]
                kwargs = {port: results[source] for port, source in node.kwargs.items()}
                value = _value(
                    self.states[name].predict(*args, **kwargs), node.kind, name, n_new, self.n_train
                )
                if node.kind == FEATURES and value.shape[1] != self.training_shapes[name][1]:
                    raise ValueError("Output feature count differs from training")
                if node.kind == LABELS and not np.isin(value, self.label_sets[name]).all():
                    raise ValueError("Prediction created labels not observed in training")
                results[node] = value
            except Exception as error:
                raise RuntimeError("Predicting node {!r} failed: {}".format(name, error)) from error
        output = self._output(results)
        if return_intermediates:
            return output, {name: results[self.graph._nodes[name]] for name in self.order}
        return output
