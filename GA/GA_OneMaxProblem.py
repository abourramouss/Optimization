from deap import base, creator, tools
import random
import matplotlib.pyplot as plt

# Problem constants
ONE_MAX_LENGTH = 100  # Length of bit string to be optimized


# Fitness function definition
def oneMaxFitness(individual):
    return (sum(individual),)  # Return a tuple


# Run GA with given parameters
def run_ga(POPULATION_SIZE, P_CROSSOVER, P_MUTATION, MAX_GENERATIONS, TOURNAMENT_SIZE):
    # Check if the class already exists and create only if it doesn't
    if not "FitnessMax" in creator.__dict__:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not "Individual" in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("zeroOrOne", random.randint, 0, 1)
    toolbox.register(
        "individualCreator",
        tools.initRepeat,
        creator.Individual,
        toolbox.zeroOrOne,
        ONE_MAX_LENGTH,
    )
    toolbox.register(
        "populationCreator", tools.initRepeat, list, toolbox.individualCreator
    )
    toolbox.register("evaluate", oneMaxFitness)
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / ONE_MAX_LENGTH)

    population = toolbox.populationCreator(n=POPULATION_SIZE)
    fitnessValues = list(map(toolbox.evaluate, population))
    for individual, fitnessValue in zip(population, fitnessValues):
        individual.fitness.values = fitnessValue

    generationCounter = 0
    maxFitnessValues = []
    meanFitnessValues = []

    while generationCounter < MAX_GENERATIONS:
        generationCounter += 1
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < P_CROSSOVER:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < P_MUTATION:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        freshIndividuals = [ind for ind in offspring if not ind.fitness.valid]
        freshFitnessValues = list(map(toolbox.evaluate, freshIndividuals))
        for individual, fitnessValue in zip(freshIndividuals, freshFitnessValues):
            individual.fitness.values = fitnessValue

        population[:] = offspring
        fitnessValues = [ind.fitness.values[0] for ind in population]

        maxFitness = max(fitnessValues)
        meanFitness = sum(fitnessValues) / len(population)
        maxFitnessValues.append(maxFitness)
        meanFitnessValues.append(meanFitness)

    return maxFitnessValues, meanFitnessValues


# Define the parameter range for brute-forcing
population_sizes = [200, 500, 1000]
crossover_probs = [0.7, 0.8, 0.9]
mutation_probs = [0.01, 0.05, 0.1]
max_generations = [50, 100]
tournament_sizes = [3, 5, 7]

# Store the best scenario, its fitness evolution data, and all fitness evolutions
best_scenario = None
best_fitness = 0
best_fitness_evolution = None
all_fitness_evolutions = []

# Iterate over all possible combinations of parameters
for pop_size in population_sizes:
    for crossover_prob in crossover_probs:
        for mutation_prob in mutation_probs:
            for max_gen in max_generations:
                for tournament_size in tournament_sizes:
                    print(
                        f"Testing with Pop: {pop_size}, Crossover: {crossover_prob}, Mutation: {mutation_prob}, Generations: {max_gen}, Tournament: {tournament_size}"
                    )
                    maxFitnessValues, meanFitnessValues = run_ga(
                        pop_size,
                        crossover_prob,
                        mutation_prob,
                        max_gen,
                        tournament_size,
                    )
                    all_fitness_evolutions.append(
                        (
                            pop_size,
                            crossover_prob,
                            mutation_prob,
                            max_gen,
                            tournament_size,
                            maxFitnessValues,
                        )
                    )
                    max_fitness = max(maxFitnessValues)
                    print(f"Result -> Max Fitness: {max_fitness}")

                    # Compare to previous best and update if necessary
                    if max_fitness > best_fitness:
                        best_fitness = max_fitness
                        best_scenario = {
                            "pop_size": pop_size,
                            "crossover": crossover_prob,
                            "mutation": mutation_prob,
                            "generations": max_gen,
                            "tournament": tournament_size,
                        }
                        best_fitness_evolution = maxFitnessValues

# Output the best scenario
print(f"Best scenario: {best_scenario}")
print(f"Best max fitness: {best_fitness}")

# Plotting the fitness evolution of each combination and highlighting the best solution
for (
    pop_size,
    crossover_prob,
    mutation_prob,
    max_gen,
    tournament_size,
    fitness_evolution,
) in all_fitness_evolutions:
    if (pop_size, crossover_prob, mutation_prob, max_gen, tournament_size) == (
        best_scenario["pop_size"],
        best_scenario["crossover"],
        best_scenario["mutation"],
        best_scenario["generations"],
        best_scenario["tournament"],
    ):
        plt.plot(
            fitness_evolution,
            label=f"Best: Pop={pop_size}, CX={crossover_prob}, Mut={mutation_prob}, Gen={max_gen}, Tour={tournament_size}",
            linewidth=2,
            color="red",
        )
    else:
        plt.plot(fitness_evolution, color="grey", alpha=0.5)

plt.title("Fitness Evolution for Different Parameter Combinations")
plt.xlabel("Generation")
plt.ylabel("Max Fitness")
plt.legend()
plt.show()
