import random
import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

# Read input data
file_path = "rpop_data_1.txt"
data = []
with open(file_path, "r") as file:
    for line in file:
        if not line.startswith("#"):
            data.append(line.strip())

target_return = np.array(data[0].split(","), dtype=float)
expected_returns = np.array(data[1].split(","), dtype=float)
cov_matrix = np.array([row.split(",") for row in data[2:]], dtype=float)
n_assets = len(expected_returns)

# GA parameters
n_generations = 100
restart_interval = 10  # number of generations before a restart
pop_size = 300
crossover_rate = 0.7
mutation_rate = 0.2  # prob. of mutation for each individual
gene_mutation_rate = 0.2  # prob. of mutation for each gene
tourn_size = 3
hof_size = 2
penalty = 100  # has to be large to discourage infeasible sols, but not too large


# Define the evaluation function with soft constraints and penalty cost
def evaluate_portfolio(individual):
    # Convert the individual (list) to a numpy array
    individual = np.array(individual)
    # Calculate portfolio return: x^T * R
    portfolio_return = np.dot(individual, expected_returns)
    # Calculate portfolio risk using matrix notation: x^T * C * x
    portfolio_risk = np.dot(np.dot(individual, cov_matrix), individual)
    # Constraint: sum of weights has to be approximately 1
    sum_of_weights_cons = 0  # if 0, no violation occurs
    if np.sum(individual) > 1.0:
        sum_of_weights_cons = 1
    # Constraint: ensure portfolio return is at least the target return
    return_cons = 0
    if portfolio_return < target_return:
        return_cons = -1
    # Constraint: the weight of each asset is bounded between 0.00 and 1.00
    weight_cons = 0
    if np.any(individual < 0.0) or np.any(individual > 1.0):
        weight_cons = 1
    # Apply a proportional penalty if some constraints are not met
    n_violations = sum_of_weights_cons + return_cons + weight_cons
    portfolio_risk += n_violations * penalty
    return (portfolio_risk,)


# Create the DEAP types and toolbox for single-objective optimization
creator.create("FitnessMinimize", base.Fitness, weights=(-1.0,))  # min risk
creator.create("Individual", list, fitness=creator.FitnessMinimize)
toolbox = base.Toolbox()

# Register gene initialization
toolbox.register("attr_float", random.uniform, 0, 1)
toolbox.register(
    "individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_assets
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=gene_mutation_rate)
toolbox.register("select", tools.selTournament, tournsize=tourn_size)
toolbox.register("evaluate", evaluate_portfolio)

stats = tools.Statistics(lambda ind: ind.fitness.values[0])
stats.register("min", np.min)

# Create a population
population = toolbox.population(n=pop_size)

# Create a Hall of Fame to store the best solutions
hof = tools.HallOfFame(hof_size)

# Run the evolution with restart using the standard genetic algorithm with statistics and Hall of Fame
gen_best_fitness = []  # List of best fitness values at each generation
for gen in range(1, n_generations + 1):
    # check if it is a restart point
    if gen % restart_interval == 0:
        # generate new individuals for the entire population except HoF
        new_population = []
        for _ in range(pop_size - hof_size):
            new_individual = toolbox.individual()
            new_population.append(new_individual)

        new_population.extend(hof)
        hof.update(population)
        population = new_population
        crossover_rate = np.random.uniform(0.5, 1)
        mutation_rate = np.random.uniform(0.0, 0.5)
        gene_mutation_rate = random.uniform(0.0, 0.5)

    pop, log = algorithms.eaSimple(
        population,
        toolbox,
        cxpb=crossover_rate,
        mutpb=mutation_rate,
        ngen=1,
        stats=stats,
        halloffame=hof,
        verbose=False,
    )
    gen_best_fitness.append(log.select("min"))

plt.figure()
best_so_far = penalty
plotted_best_fitness = []
for gen_stats in gen_best_fitness:
    min_fitness = min(gen_stats)
    best_so_far = min(best_so_far, min_fitness)
    if best_so_far < penalty:
        plotted_best_fitness.append(best_so_far)
    else:
        plotted_best_fitness.append(None)

plt.plot(plotted_best_fitness, label="Best Fitness So Far")
plt.xlabel("Generations")
plt.ylabel("Fitness (Risk)")
plt.legend()
plt.title("Evolution of Best Fitness So Far")
plt.show()

best_solution = hof[0]
print("BestSol Weights:", [f"{weight:.3f}" for weight in best_solution])
print("Sum of Weights:", f"{np.sum(best_solution):.4f}")
print("BestSol Return:", f"{np.dot(best_solution, expected_returns):.4f}")
print("BestSol Risk:", f"{best_solution.fitness.values[0]:.6f}")
