"""Genetic configuration search for fixed-topology ensemble graphs.

Evolution uses DEAP fitness/population utilities and a local random generator.
No global DEAP creator classes or Python/NumPy random seeds are modified.
"""

import copy
import math
import numbers
import pickle
import random
from collections.abc import Mapping

from deap import base, tools

from .graph import Graph


__all__ = ["Choice", "Int", "Float", "SearchResult", "SearchError", "genetic_search"]


class _Domain:
    def sample(self, rng):
        raise NotImplementedError


class Choice(_Domain):
    """Choose from a nonempty sequence, copying the chosen value for each trial."""

    def __init__(self, values):
        if isinstance(values, (str, bytes)):
            raise TypeError("Choice expects a sequence of values, not a string")
        self.values = tuple(copy.deepcopy(list(values)))
        if not self.values:
            raise ValueError("Choice requires at least one value")

    def sample(self, rng):
        return copy.deepcopy(rng.choice(self.values))


class Int(_Domain):
    """Uniform integer sampling between inclusive lower and upper bounds."""

    def __init__(self, low, high):
        if any(
            isinstance(value, bool) or not isinstance(value, numbers.Integral)
            for value in (low, high)
        ):
            raise TypeError("Int bounds must be integers")
        if low > high:
            raise ValueError("Int requires low <= high")
        self.low, self.high = int(low), int(high)

    def sample(self, rng):
        return rng.randint(self.low, self.high)


class Float(_Domain):
    """Uniform or log-uniform real sampling within finite inclusive bounds."""

    def __init__(self, low, high, log=False):
        if any(
            isinstance(value, bool) or not isinstance(value, numbers.Real) for value in (low, high)
        ):
            raise TypeError("Float bounds must be real numbers")
        if not math.isfinite(low) or not math.isfinite(high) or low > high:
            raise ValueError("Float requires finite bounds with low <= high")
        if not isinstance(log, bool):
            raise TypeError("log must be a boolean")
        if log and low <= 0:
            raise ValueError("Log-uniform bounds must be positive")
        self.low, self.high, self.log = float(low), float(high), log

    def sample(self, rng):
        if self.log:
            value = math.exp(rng.uniform(math.log(self.low), math.log(self.high)))
        else:
            value = rng.uniform(self.low, self.high)
        return min(self.high, max(self.low, value))


class SearchResult:
    """Result of a search, including an unexecuted copy of the best graph.

    best_params contains active overrides for the original graph. history has
    one row per evaluation request (including cache hits), and logbook has one
    row per generation, including generation zero. n_evaluations excludes cache
    hits and elites carried forward without reevaluation.
    """

    def __init__(self, best_graph, best_params, best_score, history, logbook, n_evaluations):
        self.best_graph = best_graph
        self.best_params = copy.deepcopy(best_params)
        self.best_score = best_score
        self.history = history
        self.logbook = logbook
        self.n_evaluations = n_evaluations

    def __repr__(self):
        return "SearchResult(best_score={!r}, n_evaluations={})".format(
            self.best_score, self.n_evaluations
        )


class SearchError(RuntimeError):
    """Search failure with evaluation history and completed generation summaries."""

    def __init__(self, message, history, logbook):
        super().__init__(message)
        self.history = copy.deepcopy(history)
        self.logbook = copy.deepcopy(logbook)


class _MaxFitness(base.Fitness):
    weights = (1.0,)


class _MinFitness(base.Fitness):
    weights = (-1.0,)


class _Individual(list):
    def __init__(self, genes, maximize):
        super().__init__(genes)
        self.fitness = _MaxFitness() if maximize else _MinFitness()
        self.score = None


def _domain(value):
    if isinstance(value, _Domain):
        return copy.deepcopy(value)
    if isinstance(value, (list, tuple)):
        return Choice(value)
    return Choice([value])


def _space(graph, search_space):
    if not isinstance(search_space, Mapping) or not search_space:
        raise ValueError("search_space must map at least one block name to its parameters")
    blocks = dict(graph.named_blocks())
    normalized = []
    for name, definition in search_space.items():
        if name not in blocks:
            raise ValueError("Unknown block {!r}; available blocks: {}".format(name, list(blocks)))
        cases = [definition] if isinstance(definition, Mapping) else definition
        if not isinstance(cases, (list, tuple)) or not cases:
            raise ValueError("Each block needs a parameter mapping or a nonempty list of cases")
        normalized_cases = []
        for case in cases:
            if not isinstance(case, Mapping) or not case:
                raise ValueError("Each conditional case must be a nonempty parameter mapping")
            if not all(isinstance(key, str) and key for key in case):
                raise ValueError("Parameter names must be nonempty strings")
            normalized_cases.append({key: _domain(value) for key, value in case.items()})
        normalized.append((name, normalized_cases))
    return normalized


def _sample_gene(cases, rng, case=None):
    case = rng.randrange(len(cases)) if case is None else case
    return {
        "case": case,
        "params": {key: domain.sample(rng) for key, domain in cases[case].items()},
    }


def _params(individual, space):
    return {
        "{}__{}".format(name, key): copy.deepcopy(value)
        for (name, _), gene in zip(space, individual)
        for key, value in gene["params"].items()
    }


def _mate(left, right, rng):
    for i, (a, b) in enumerate(zip(left, right)):
        if a["case"] != b["case"]:
            if rng.random() < 0.5:
                left[i], right[i] = b, a
        else:
            for key in a["params"]:
                if rng.random() < 0.5:
                    a["params"][key], b["params"][key] = b["params"][key], a["params"][key]


def _mutate(individual, space, rng):
    i = rng.randrange(len(space))
    cases = space[i][1]
    gene = individual[i]
    if len(cases) > 1 and rng.random() < 0.5:
        other = rng.choice([case for case in range(len(cases)) if case != gene["case"]])
        individual[i] = _sample_gene(cases, rng, case=other)
    else:
        key = rng.choice(list(gene["params"]))
        gene["params"][key] = cases[gene["case"]][key].sample(rng)


def _integer(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError("{} must be an integer".format(name))
    if value < minimum:
        raise ValueError("{} must be at least {}".format(name, minimum))


def genetic_search(
    graph,
    inputs,
    search_space,
    fitness,
    population_size=20,
    generations=10,
    crossover_probability=0.7,
    mutation_probability=0.2,
    tournament_size=3,
    elitism=1,
    random_state=None,
    maximize=True,
    on_error="raise",
    cache=False,
):
    """Optimize block configurations while keeping graph connectivity fixed.

    search_space maps names from graph.named_blocks() to parameter mappings.
    A list of mappings defines conditional cases for a block. Parameter values
    can be Choice/Int/Float domains, lists/tuples of choices, or fixed scalars.
    Each case remains intact when crossing or mutating clustering families.

    fitness receives the graph's output (a value or named-output dictionary)
    and must return a finite real scalar. Scores are maximized by default.
    Use maximize=False for losses. No labels or built-in objective are imposed.

    random_state seeds a local evolutionary RNG, not clustering estimators or
    the fitness callable. Set their own seeds for reproducible evaluations.
    on_error='raise' aborts on a candidate error, preserving its cause/history;
    'penalize' records it and assigns worst fitness. An entirely failed search
    raises SearchError instead of returning an unusable graph.

    cache=True reuses scores for identical active parameter overrides and needs
    deterministic graph operations/fitness and picklable parameter values.
    By default duplicate candidates are reevaluated. Elites keep their recorded
    scores. history records failures and cache hits explicitly. Every evaluated
    candidate and the returned best_graph are separate copies of the template.
    Inputs and outputs must be treated as read-only by custom blocks/fitness.
    """
    if not isinstance(graph, Graph):
        raise TypeError("graph must be a compiled Graph")
    if not isinstance(inputs, Mapping) or set(inputs) != set(graph.inputs):
        raise ValueError("inputs must match the graph's declared input names")
    if not callable(fitness):
        raise TypeError("fitness must be callable")
    for name, value, minimum in [
        ("population_size", population_size, 2),
        ("generations", generations, 0),
        ("tournament_size", tournament_size, 2),
        ("elitism", elitism, 0),
    ]:
        _integer(name, value, minimum)
    if elitism >= population_size:
        raise ValueError("elitism must be smaller than population_size")
    for name, value in [
        ("crossover_probability", crossover_probability),
        ("mutation_probability", mutation_probability),
    ]:
        if (
            isinstance(value, bool)
            or not isinstance(value, numbers.Real)
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError("{} must be a probability between zero and one".format(name))
    if not isinstance(maximize, bool) or not isinstance(cache, bool):
        raise TypeError("maximize and cache must be booleans")
    if on_error not in ("raise", "penalize"):
        raise ValueError("on_error must be 'raise' or 'penalize'")
    if random_state is not None:
        _integer("random_state", random_state, 0)
        random_state = int(random_state)
    rng = random.Random(random_state)
    template = copy.deepcopy(graph)
    space = _space(template, search_space)
    history, logbook, cached = [], tools.Logbook(), {}
    n_evaluations = 0
    best_params, best_score = None, None
    toolbox = base.Toolbox()
    toolbox.register(
        "individual",
        lambda: _Individual([_sample_gene(cases, rng) for _, cases in space], maximize),
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual, generation, index):
        nonlocal n_evaluations, best_params, best_score
        params = _params(individual, space)
        key = None
        if cache:
            try:
                key = pickle.dumps(sorted(params.items()), protocol=4)
            except Exception as error:
                raise TypeError("Score caching requires picklable parameter values") from error
        hit = cache and key in cached
        error_object = None
        if hit:
            score, phase, error_message = cached[key]
        else:
            n_evaluations += 1
            phase, error_message, score = "configuration", None, None
            try:
                candidate = template.with_params(**params)
                phase = "execution"
                output = candidate.run(inputs)
                phase = "fitness"
                value = fitness(output)
                if isinstance(value, bool) or not isinstance(value, numbers.Real):
                    raise TypeError("fitness must return a real scalar")
                score = float(value)
                if not math.isfinite(score):
                    raise ValueError("fitness must return a finite score")
                phase = None
            except Exception as error:
                score = None
                error_object = error
                error_message = "{}: {}".format(type(error).__name__, error)
            if cache:
                cached[key] = (score, phase, error_message)
        history.append(
            {
                "generation": generation,
                "individual": index,
                "params": copy.deepcopy(params),
                "score": score,
                "status": "ok" if score is not None else "failed",
                "cached": bool(hit),
                "phase": phase,
                "error": error_message,
            }
        )
        if error_object is not None and on_error == "raise":
            raise SearchError(
                "Candidate failed during {}: {}".format(phase, error_message), history, logbook
            ) from error_object
        individual.score = score
        individual.fitness.values = (
            score if score is not None else (-math.inf if maximize else math.inf),
        )
        if score is not None and (
            best_score is None or (score > best_score if maximize else score < best_score)
        ):
            best_score, best_params = score, copy.deepcopy(params)

    def summarize(population, generation, previous_evaluations, previous_history):
        scores = [individual.score for individual in population if individual.score is not None]
        logbook.record(
            generation=generation,
            evaluations=n_evaluations - previous_evaluations,
            cache_hits=sum(row["cached"] for row in history[previous_history:]),
            valid=len(scores),
            failed=len(population) - len(scores),
            generation_best=(max(scores) if maximize else min(scores)) if scores else None,
            best_score=best_score,
            mean_score=sum(scores) / len(scores) if scores else None,
        )

    population = toolbox.population(n=population_size)
    for index, individual in enumerate(population):
        evaluate(individual, 0, index)
    summarize(population, 0, 0, 0)
    for generation in range(1, generations + 1):
        previous_evaluations, previous_history = n_evaluations, len(history)
        elites = [
            toolbox.clone(individual)
            for individual in sorted(population, key=lambda item: item.fitness, reverse=True)[
                :elitism
            ]
        ]
        children = []
        for _ in range(population_size - elitism):
            contenders = [rng.choice(population) for _ in range(tournament_size)]
            children.append(toolbox.clone(max(contenders, key=lambda item: item.fitness)))
        for i in range(0, len(children) - 1, 2):
            if rng.random() < crossover_probability:
                _mate(children[i], children[i + 1], rng)
        for index, child in enumerate(children):
            if rng.random() < mutation_probability:
                _mutate(child, space, rng)
            evaluate(child, generation, index)
        population = elites + children
        summarize(population, generation, previous_evaluations, previous_history)
    if best_params is None:
        first_error = history[0]["error"]
        raise SearchError(
            "No candidate produced a valid fitness score. First failure: {}".format(first_error),
            history,
            logbook,
        )
    return SearchResult(
        template.with_params(**best_params),
        best_params,
        best_score,
        history,
        logbook,
        n_evaluations,
    )
