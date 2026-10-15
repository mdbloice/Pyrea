"""Composable modules and dependency-aware execution of ensemble graphs."""

import copy
import inspect
from collections import OrderedDict, deque

import networkx as nx
import numpy as np


FEATURES = "features"
LABELS = "labels"
DISTANCE = "distance"
AFFINITY = "affinity"
KINDS = (FEATURES, LABELS, DISTANCE, AFFINITY)


class Node:
    """A symbolic result of an input or a block call; contains no sample data."""

    def __init__(self, block, args=(), kwargs=None, name=None, kind=None):
        self.block = block
        self.args = tuple(args)
        self.kwargs = dict(kwargs or {})
        self.name = name
        self.kind = kind if block is None else block.output_kind
        if self.kind not in KINDS:
            raise ValueError("Unknown data kind: {!r}".format(self.kind))

    def __bool__(self):
        raise TypeError("Symbolic nodes cannot be used in data-dependent control flow")


def Input(name, kind=FEATURES):
    """Declare a named symbolic input, normally a sample-by-feature matrix."""
    if not isinstance(name, str) or not name or "__" in name:
        raise ValueError('Input names must be nonempty strings without "__"')
    return Node(None, name=name, kind=kind)


class Module:
    """Compose blocks in ``forward``; calls work with arrays or symbolic nodes.

    Subclass Module for compositions and Block for indivisible operations.
    ``compile()`` traces symbolic calls into an executable Graph.
    """

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def compile(self, **inputs):
        """Build a graph, inferring feature inputs from a fixed forward signature.

        Supply explicit Input nodes for variable signatures or other data kinds.
        """
        if not inputs:
            for name, parameter in inspect.signature(self.forward).parameters.items():
                if parameter.kind in (
                    parameter.VAR_POSITIONAL,
                    parameter.VAR_KEYWORD,
                    parameter.POSITIONAL_ONLY,
                ):
                    raise ValueError("Supply named Input nodes to compile this signature")
                if parameter.default is parameter.empty:
                    inputs[name] = Input(name)
        if not inputs or not all(
            isinstance(value, Node) and value.block is None for value in inputs.values()
        ):
            raise TypeError("compile expects named Input nodes")
        names = {}
        seen = set()

        def visit(value, path):
            if id(value) in seen:
                return
            seen.add(id(value))
            if isinstance(value, Block):
                names[value] = path or value.__class__.__name__.lower()
            elif isinstance(value, Module):
                for key, child in vars(value).items():
                    visit(child, "{}.{}".format(path, key).lstrip("."))
            elif isinstance(value, (list, tuple, dict)):
                children = value.items() if isinstance(value, dict) else enumerate(value)
                for key, child in children:
                    visit(child, "{}.{}".format(path, key).lstrip("."))

        visit(self, "")
        return Graph(inputs, self(**inputs), _block_names=names)


class Block(Module):
    """An executable operation with declared input and output kinds.

    Override forward and optionally get_params/set_params for optimization.
    input_kinds maps forward argument names to a kind (or tuple of kinds).
    A variadic argument's declaration applies to every connected input.
    """

    input_kinds = {}
    output_kind = FEATURES
    name = None

    def _signature(self):
        return inspect.signature(self.forward)

    def ports(self, args, kwargs):
        """Bind dependencies to named forward arguments, including variadic ports."""
        signature = self._signature()
        bound = signature.bind(*args, **kwargs)
        result = []
        for name, value in bound.arguments.items():
            parameter = signature.parameters[name]
            if parameter.kind == parameter.VAR_POSITIONAL:
                result.extend(
                    ("{}[{}]".format(name, i), name, item) for i, item in enumerate(value)
                )
            elif parameter.kind == parameter.VAR_KEYWORD:
                result.extend((key, key, item) for key, item in value.items())
            else:
                result.append((name, name, value))
        return result

    def __call__(self, *args, **kwargs):
        ports = self.ports(args, kwargs)
        if any(isinstance(value, Node) for _, _, value in ports):
            self._validate_ports(ports)
            return Node(self, args, kwargs)
        return self.forward(*args, **kwargs)

    def _validate_ports(self, ports):
        for port, parameter, value in ports:
            if not isinstance(value, Node):
                raise TypeError(
                    "Symbolic input {} must be a Node; configure constants "
                    "in the block constructor".format(port)
                )
            expected = self.input_kinds.get(parameter, KINDS)
            expected = (expected,) if isinstance(expected, str) else expected
            if value.kind not in expected:
                raise TypeError(
                    "{} port {} expects {}, received {}".format(
                        self.__class__.__name__, port, expected, value.kind
                    )
                )

    def get_params(self):
        return {}

    def set_params(self, **params):
        if params:
            raise ValueError("{} has no tunable parameters".format(self.__class__.__name__))
        return self


class SampleData:
    """Attach ordered sample identifiers to an array for alignment validation.

    All graph inputs must have the same sample order. DataFrame indexes are
    also recognized. Bare arrays use positional alignment.
    """

    def __init__(self, values, sample_ids):
        self.values = values
        self.sample_ids = tuple(sample_ids)


def _validate_value(value, kind, name, n_samples=None):
    value = np.asarray(value)
    if kind == LABELS:
        if value.ndim == 2 and value.shape[1] == 1:
            value = value[:, 0]
        if value.ndim != 1:
            raise ValueError("{} must contain one label per sample".format(name))
    elif value.ndim != 2:
        raise ValueError("{} must be a two-dimensional {} matrix".format(name, kind))
    if not value.shape[0] or (value.ndim == 2 and not value.shape[1]):
        raise ValueError("{} must not be empty".format(name))
    if n_samples is not None and value.shape[0] != n_samples:
        raise ValueError("{} has {} samples, expected {}".format(name, value.shape[0], n_samples))
    if kind in (AFFINITY, DISTANCE):
        if value.shape[0] != value.shape[1]:
            raise ValueError("{} must be square".format(name))
        if not np.isfinite(value).all() or (value < 0).any():
            raise ValueError("{} must contain finite, nonnegative values".format(name))
        if not np.allclose(value, value.T):
            raise ValueError("{} must be symmetric".format(name))
        if kind == DISTANCE and not np.allclose(np.diag(value), 0):
            raise ValueError("{} must have a zero distance diagonal".format(name))
    return value


class Graph(Module):
    """A NetworkX DAG compiled from symbolic block calls.

    Outputs may be a single Node or a dictionary of named Nodes. Execution is
    sequential in topological order, with shared results computed once per run.
    ``graph`` returns a frozen NetworkX copy for traversal and visualization.
    """

    def __init__(self, inputs, outputs, _block_names=None):
        self.inputs = OrderedDict(inputs)
        self.outputs = outputs.copy() if isinstance(outputs, dict) else outputs
        if not self.inputs:
            raise ValueError("A graph requires at least one input")
        for name, node in self.inputs.items():
            if not isinstance(node, Node) or node.block is not None or node.name != name:
                raise ValueError("Graph input keys must match their Input names")
        output_nodes = list(outputs.values()) if isinstance(outputs, dict) else [outputs]
        if not output_nodes or not all(isinstance(node, Node) for node in output_nodes):
            raise TypeError("Outputs must be a Node or a nonempty dictionary of Nodes")
        topology = nx.DiGraph()
        pending = deque(output_nodes)
        while pending:
            node = pending.popleft()
            if node in topology:
                continue
            topology.add_node(node)
            pending.extend(value for _, _, value in self._ports(node))
        # Build edges after discovery so shared dependencies are not skipped.
        for node in list(topology):
            for port, _, source in self._ports(node):
                if topology.has_edge(source, node):
                    topology[source][node]["ports"].append(port)
                else:
                    topology.add_edge(source, node, ports=[port])
        if not nx.is_directed_acyclic_graph(topology):
            raise ValueError("Ensemble workflows must be acyclic")
        sources = {node for node in topology if node.block is None}
        if sources != set(self.inputs.values()):
            raise ValueError("Declare every reachable Input exactly once, with no unused inputs")
        self._names = {}
        self._blocks = OrderedDict()
        used = set(self.inputs)
        block_names = {}
        for node in nx.topological_sort(topology):
            if node.block is None:
                self._names[node] = node.name
                continue
            block = node.block
            base = (
                block_names.get(block)
                or block.name
                or (_block_names or {}).get(block)
                or block.__class__.__name__.lower()
            )
            if not isinstance(base, str) or not base or "__" in base:
                raise ValueError('Block names must be nonempty strings without "__"')
            name, suffix = base, 1
            while name in used:
                name = "{}_{}".format(base, suffix)
                suffix += 1
            used.add(name)
            self._names[node] = name
            if block not in block_names:
                block_names[block] = name
                self._blocks[name] = block
        for node in topology:
            topology.nodes[node].update(block=node.block, kind=node.kind)
        self._graph = nx.relabel_nodes(topology, self._names, copy=True)
        self._nodes = {name: node for node, name in self._names.items()}

    @staticmethod
    def _ports(node):
        if node.block is None:
            return []
        ports = node.block.ports(node.args, node.kwargs)
        if not ports:
            raise TypeError("Executable nodes must depend on symbolic inputs")
        if node.block.output_kind != node.kind:
            raise ValueError("Changing output kinds requires rebuilding the graph")
        node.block._validate_ports(ports)
        return ports

    @property
    def graph(self):
        """Return a frozen NetworkX snapshot for inspection and traversal."""
        return nx.freeze(self._graph.copy())

    def named_blocks(self):
        """Return unique blocks; a reused block has shared configuration."""
        return iter(self._blocks.items())

    def get_params(self):
        """Flatten tunable configuration as ``block_name__parameter``."""
        return {
            "{}__{}".format(name, key): value
            for name, block in self.named_blocks()
            for key, value in block.get_params().items()
        }

    def with_params(self, **params):
        """Return an independent configured copy for a search trial."""
        result = copy.deepcopy(self)
        grouped = {}
        for key, value in params.items():
            name, separator, parameter = key.partition("__")
            if not separator or name not in result._blocks or not parameter:
                raise ValueError("Unknown graph parameter {!r}".format(key))
            grouped.setdefault(name, {})[parameter] = value
        for name, values in grouped.items():
            result._blocks[name].set_params(**values)
        for node in result._nodes.values():
            result._ports(node)
        return result

    def forward(self, *args, **inputs):
        if len(args) > len(self.inputs):
            raise TypeError("Too many positional inputs")
        for name, value in zip(self.inputs, args):
            if name in inputs:
                raise TypeError("Multiple values for input {!r}".format(name))
            inputs[name] = value
        return self.run(inputs)

    def run(self, inputs, return_intermediates=False):
        """Execute with named inputs, optionally returning a per-run result map.

        Treat incoming arrays as read-only in custom blocks. Arrays are not
        copied between branches. No fitted state or outputs are cached here.
        """
        if set(inputs) != set(self.inputs):
            raise ValueError(
                "Input mismatch: missing {}, unexpected {}".format(
                    sorted(set(self.inputs) - set(inputs)), sorted(set(inputs) - set(self.inputs))
                )
            )
        results = {}
        n_samples = None
        reference_ids = None
        for name, node in self.inputs.items():
            value = inputs[name]
            ids = None
            if isinstance(value, SampleData):
                ids, value = value.sample_ids, value.values
            elif hasattr(value, "index") and hasattr(value, "columns"):
                ids = tuple(value.index)
                if node.kind in (DISTANCE, AFFINITY) and tuple(value.columns) != ids:
                    raise ValueError("{} matrix columns must match its sample index".format(name))
            value = _validate_value(value, node.kind, name, n_samples)
            n_samples = value.shape[0]
            if ids is not None:
                if len(ids) != n_samples or len(set(ids)) != n_samples:
                    raise ValueError("{} requires one unique identifier per sample".format(name))
                if reference_ids is not None and ids != reference_ids:
                    raise ValueError(
                        "{} sample identifiers/order do not match other inputs".format(name)
                    )
                reference_ids = ids
            results[node] = value
        for name in nx.topological_sort(self._graph):
            node = self._nodes[name]
            if node.block is None:
                continue
            try:
                args = [results[source] for source in node.args]
                kwargs = {port: results[source] for port, source in node.kwargs.items()}
                value = node.block(*args, **kwargs)
                results[node] = _validate_value(value, node.kind, name, n_samples)
            except Exception as error:
                raise RuntimeError(
                    "Node {!r} ({}) failed: {}".format(name, node.block.__class__.__name__, error)
                ) from error
        if isinstance(self.outputs, dict):
            output = {name: results[node] for name, node in self.outputs.items()}
        else:
            output = results[self.outputs]
        if return_intermediates:
            return output, {self._names[node]: value for node, value in results.items()}
        return output
