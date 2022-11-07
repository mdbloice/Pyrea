import random
import numpy as np

# from deap import benchmarks
from deap import base
from deap import creator
from deap import tools
from deap import algorithms

# Following the example shown in: https://www.linkedin.com/pulse/hyper-parameter-optimisation-using-genetic-algorithms-conor-rothwell

# List all parameters and their types
# linkage/method:   string
# fusion method:    string
# n_clusters:       int
# n_ensembles:      int
# height:           int
# Hence one chromosome might look as follows:
#['single', 'disagreement', 2, 2, 4]

# Create a fitness metric and one individual using this fitness metric
creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # Maximise the fitness function value
creator.create("Individual", list, fitness=creator.FitnessMax)

# Create a new toolbox to add genes:
toolbox = base.Toolbox()

# Now define all possible values
LINKAGES = ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']
FUSION_METHODS = ['agreement', 'consensus', 'disagreement']
N_CLUSTERS_MIN, N_CLUSTERS_MAX = 1, 20
N_ENSEMBLES_MIN, N_ENSEMBLES_MAX = 1, 20
OPTION_OR_NONE = ['balanced', None]

# Define how each gene will be generated (e.g. criterion is a random choice from the criterion list).
# You can think of these as custom types. For example if two attributes shared the
# same characteristics, e.g. float between 0 and 1, then you only need to create this
# once, e.g. attr_float, random.random and use it twice when registering an
# individual below
toolbox.register("attr_linkage", random.choice, LINKAGES)
toolbox.register("attr_fusion_method", random.choice, FUSION_METHODS)
toolbox.register("attr_n_clusters", random.randint, N_CLUSTERS_MIN, N_CLUSTERS_MAX)
toolbox.register("attr_n_ensembles", random.randint, N_ENSEMBLES_MIN, N_CLUSTERS_MAX)

# This is the order in which genes will be combined to create a chromosome
N_CYCLES = 1
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_linkage, toolbox.attr_fusion_method, toolbox.attr_n_clusters, toolbox.attr_n_ensembles), n=N_CYCLES)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Sometimes it is a requirement to build a custom mutate function, we will do that here
def mutate(individual):

    # We have 4 possible genes to mutate, so let's choose a random one here
    gene_selector = random.randint(0, 3)

    # 0: linkage
    # 1: fusion method
    # 2: n clusters
    # 3: n ensembles

    if gene_selector == 0:
        individual[gene_selector] = random.choice(LINKAGES)
    elif gene_selector == 1:
        individual[gene_selector] = random.choice(FUSION_METHODS)
    elif gene_selector == 2:
        individual[gene_selector] = random.randint(N_CLUSTERS_MIN, N_CLUSTERS_MAX)
    elif gene_selector == 3:
        individual[gene_selector] = random.randint(N_ENSEMBLES_MIN, N_ENSEMBLES_MAX)

    return individual

# The evaluate function is almost always defined by the user.
def evaluate(individual):
    return 'xyz'

# Now the functions we have defined above need to be registered with the toolbox
# and some other functions need to be set
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selTournament, tournsize=2)  # this param can be adjusted
toolbox.register("evaluate", evaluate)

# Now a number of params need to be set, these depend on the problem and will need
# to be adjusted.
population_size = 1000
crossover_probability = 0.7
mutation_probability = 0.01
number_of_generations = 20

pop = toolbox.population(n=population_size)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)
pop, log = algorithms.eaSimple(pop, toolbox, cxpb=crossover_probability, stats = stats,
                               mutpb = mutation_probability, ngen=number_of_generations, halloffame=hof,
                               verbose=True)

best_parameters = hof[0] # save the optimal set of parameters