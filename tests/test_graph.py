import networkx as nx
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from pyrea import nn


class Offset(nn.Block):
    input_kinds = {"data": nn.FEATURES}
    output_kind = nn.FEATURES

    def __init__(self, amount=1):
        self.amount = amount
        self.calls = 0

    def forward(self, data):
        self.calls += 1
        return data + self.amount

    def get_params(self):
        return {"amount": self.amount}

    def set_params(self, **params):
        self.amount = params["amount"]
        return self


def test_lazy_shared_branch_unequal_depth_and_multiple_outputs():
    x = nn.Input("x")
    shared_block = Offset(2)
    shared = shared_block(x)
    deeper = Offset(5)(shared)
    join = nn.Function(
        lambda left, right: left - right, {"left": nn.FEATURES, "right": nn.FEATURES}, name="join"
    )
    model = nn.Graph({"x": x}, {"joined": join(left=deeper, right=shared), "shared": shared})
    assert shared_block.calls == 0
    data = np.arange(12).reshape(6, 2)
    output, intermediate = model.run({"x": data}, return_intermediates=True)
    assert_array_equal(output["joined"], np.full_like(data, 5))
    assert_array_equal(output["shared"], data + 2)
    assert shared_block.calls == 1
    assert_array_equal(intermediate["join"], output["joined"])
    assert_array_equal(model(x=data * 2)["shared"], data * 2 + 2)
    assert shared_block.calls == 2
    assert isinstance(model.graph, nx.DiGraph)
    assert len(nx.descendants(model.graph, "x")) == 3
    assert list(nx.bfs_tree(model.graph, "x"))[0] == "x"
    with pytest.raises(nx.NetworkXError):
        model.graph.add_edge("join", "x")


def test_multiple_bindings_from_same_source_keep_named_ports():
    x = nn.Input("x")
    subtract = nn.Function(
        lambda left, right: left - right,
        {"left": nn.FEATURES, "right": nn.FEATURES},
        name="subtract",
    )
    model = nn.Graph({"x": x}, subtract(right=x, left=x))
    assert set(model.graph["x"]["subtract"]["ports"]) == {"left", "right"}
    assert_array_equal(model(x=np.ones((3, 2))), np.zeros((3, 2)))


def test_positional_and_keyword_model_calls():
    x, y = nn.Input("x"), nn.Input("y")
    model = nn.Graph({"x": x, "y": y}, {"x": x, "y": y})
    data = np.ones((3, 2))
    assert_array_equal(model(data, data)["x"], data)
    assert_array_equal(model(data, y=data)["y"], data)
    with pytest.raises(TypeError, match="Multiple values"):
        model(data, x=data, y=data)
    with pytest.raises(TypeError, match="Too many"):
        model(data, data, data)


def test_subclassed_nested_modules_compile_and_execute_eagerly():
    class Branch(nn.Module):
        def __init__(self):
            self.offset = Offset(3)

        def forward(self, data):
            return self.offset(data)

    class Model(nn.Module):
        def __init__(self):
            self.branch = Branch()
            self.join = nn.Function(
                lambda left, right: left + right, {"left": nn.FEATURES, "right": nn.FEATURES}
            )

        def forward(self, first, second):
            return self.join(self.branch(first), second)

    module = Model()
    model = module.compile()
    assert "branch.offset__amount" in model.get_params()
    first, second = np.ones((4, 2)), np.full((4, 2), 5)
    assert_array_equal(model(first=first, second=second), module(first, second))
    trial = model.with_params(**{"branch.offset__amount": 10})
    assert_array_equal(trial(first=first, second=second), first + second + 10)
    assert model.get_params()["branch.offset__amount"] == 3


def test_shared_block_has_shared_parameters_but_distinct_calls():
    x, y = nn.Input("x"), nn.Input("y")
    block = Offset(2)
    first, second = block(x), block(y)
    model = nn.Graph({"x": x, "y": y}, {"first": first, "second": second})
    assert len(list(model.named_blocks())) == 1
    trial = model.with_params(offset__amount=7)
    output = trial(x=np.zeros((2, 1)), y=np.ones((2, 1)))
    assert_array_equal(output["first"], [[7], [7]])
    assert_array_equal(output["second"], [[8], [8]])
    assert block.calls == 0
    trial_block = dict(trial.named_blocks())["offset"]
    assert trial_block.calls == 2


@pytest.mark.parametrize(
    "kind,block",
    [
        (nn.LABELS, nn.Hierarchical()),
        (nn.AFFINITY, nn.Hierarchical(input_kind=nn.DISTANCE)),
        (nn.DISTANCE, nn.Cluster("spectral", input_kind=nn.AFFINITY)),
        (nn.FEATURES, nn.Fusion()),
    ],
)
def test_invalid_connections_rejected_during_construction(kind, block):
    with pytest.raises(TypeError, match="expects"):
        block(nn.Input("x", kind))


def test_symbolic_constants_and_missing_ports_rejected():
    block = nn.Function(
        lambda left, right: left + right, {"left": nn.FEATURES, "right": nn.FEATURES}
    )
    with pytest.raises(TypeError, match="must be a Node"):
        block(nn.Input("x"), np.ones((2, 1)))
    with pytest.raises(TypeError, match="right"):
        block(nn.Input("x"))
    with pytest.raises(TypeError, match="data-dependent"):
        bool(nn.Input("x"))


def test_graph_rejects_cycles_and_undeclared_or_unused_inputs():
    x, unused = nn.Input("x"), nn.Input("unused")
    result = Offset()(x)
    with pytest.raises(ValueError, match="reachable Input"):
        nn.Graph({"unused": unused}, result)
    with pytest.raises(ValueError, match="unused inputs"):
        nn.Graph({"x": x, "unused": unused}, result)
    # A cycle cannot arise through ordinary symbolic calls; reject malformed nodes too.
    result.args = (result,)
    with pytest.raises(ValueError, match="acyclic"):
        nn.Graph({"x": x}, result)


def test_runtime_inputs_and_sample_counts_checked_before_execution():
    x, y = nn.Input("x"), nn.Input("y")
    block = Offset()
    model = nn.Graph({"x": x, "y": y}, {"x": block(x), "y": block(y)})
    with pytest.raises(ValueError, match="missing"):
        model(x=np.ones((3, 2)))
    with pytest.raises(ValueError, match="unexpected"):
        model(x=np.ones((3, 2)), y=np.ones((3, 2)), extra=0)
    with pytest.raises(ValueError, match="samples"):
        model(x=np.ones((3, 2)), y=np.ones((4, 2)))
    assert block.calls == 0


@pytest.mark.parametrize(
    "wrap",
    [
        lambda data, ids: nn.SampleData(data, ids),
        lambda data, ids: pd.DataFrame(data, index=ids),
    ],
)
def test_identified_inputs_reject_misalignment(wrap):
    x, y = nn.Input("x"), nn.Input("y")
    model = nn.Graph({"x": x, "y": y}, {"x": x, "y": y})
    data = np.arange(6).reshape(3, 2)
    with pytest.raises(ValueError, match="identifiers/order"):
        model(x=wrap(data, ["a", "b", "c"]), y=wrap(data, ["c", "b", "a"]))
    assert_array_equal(
        model(x=wrap(data, ["a", "b", "c"]), y=wrap(data, ["a", "b", "c"]))["x"], data
    )


def test_pairwise_dataframe_columns_must_match_index():
    x = nn.Input("x", nn.AFFINITY)
    model = nn.Graph({"x": x}, x)
    data = pd.DataFrame(np.eye(2), index=["a", "b"], columns=["b", "a"])
    with pytest.raises(ValueError, match="columns"):
        model(x=data)


@pytest.mark.parametrize(
    "data,kind,message",
    [
        (np.ones(3), nn.FEATURES, "two-dimensional"),
        (np.empty((0, 2)), nn.FEATURES, "empty"),
        (np.ones((2, 3)), nn.DISTANCE, "square"),
        (np.eye(2), nn.DISTANCE, "zero distance diagonal"),
        (np.array([[0, 1], [0, 0]]), nn.DISTANCE, "symmetric"),
        (np.array([[0, -1], [-1, 0]]), nn.DISTANCE, "nonnegative"),
        (np.full((2, 2), np.nan), nn.AFFINITY, "finite"),
        (np.ones((2, 2)), nn.LABELS, "one label"),
    ],
)
def test_input_shape_and_pairwise_semantics(data, kind, message):
    x = nn.Input("x", kind)
    model = nn.Graph({"x": x}, x)
    with pytest.raises(ValueError, match=message):
        model(x=data)


def test_custom_output_validation_and_node_context():
    x = nn.Input("x")
    block = nn.Function(
        lambda data: np.zeros(2), {"data": nn.FEATURES}, output_kind=nn.LABELS, name="broken"
    )
    with pytest.raises(RuntimeError, match="Node 'broken'.*samples"):
        nn.Graph({"x": x}, block(x))(x=np.ones((3, 2)))


def test_with_params_rejects_unknown_names_and_preserves_original():
    x = nn.Input("x")
    model = nn.Graph({"x": x}, nn.Hierarchical(name="cluster")(x))
    with pytest.raises(ValueError, match="Unknown graph parameter"):
        model.with_params(unknown__n_clusters=4)
    with pytest.raises(TypeError):
        model.with_params(cluster__misspelled_parameter=4)
    assert model.get_params()["cluster__n_clusters"] == 2
    trial = model.with_params(cluster__n_clusters=3)
    assert trial.get_params()["cluster__n_clusters"] == 3
    data = np.random.RandomState(4).normal(size=(20, 3))
    assert len(np.unique(model(x=data))) == 2
    assert len(np.unique(trial(x=data))) == 3


def test_compiling_variadic_module_requires_explicit_inputs():
    class Model(nn.Module):
        def forward(self, **views):
            return nn.Consensus()(*(nn.Hierarchical()(value) for value in views.values()))

    with pytest.raises(ValueError, match="Supply named Input"):
        Model().compile()
    model = Model().compile(x=nn.Input("x"), y=nn.Input("y"))
    assert model(x=np.arange(12).reshape(6, 2), y=np.arange(12).reshape(6, 2)).shape == (6,)


def test_keyword_only_ports_and_exception_cause():
    def join(*, left, right):
        return left + right

    x, y = nn.Input("x"), nn.Input("y")
    block = nn.Function(join, {"left": nn.FEATURES, "right": nn.FEATURES})
    model = nn.Graph({"x": x, "y": y}, block(right=y, left=x))
    assert_array_equal(model(np.ones((3, 2)), np.ones((3, 2))), np.full((3, 2), 2))

    def fail(data):
        raise ValueError("original failure")

    model = nn.Graph({"x": x}, nn.Function(fail, {"data": nn.FEATURES})(x))
    with pytest.raises(RuntimeError, match="original failure") as caught:
        model(x=np.ones((3, 2)))
    assert isinstance(caught.value.__cause__, ValueError)


def test_custom_parameter_trials_revalidate_connections():
    class Configurable(Offset):
        def get_params(self):
            return {"output_kind": self.output_kind}

        def set_params(self, **params):
            self.output_kind = params["output_kind"]
            return self

    source = nn.Input("data")
    block = Configurable()
    model = nn.Graph({"data": source}, block(source))
    with pytest.raises(ValueError, match="Changing output kinds"):
        model.with_params(configurable__output_kind=nn.LABELS)
    assert block.output_kind == nn.FEATURES
