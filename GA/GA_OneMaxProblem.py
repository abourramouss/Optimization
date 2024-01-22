from deap import base
from deap import creator
from deap import tools
import random
import matplotlib.pyplot as plt

# problem constants

# length of bit string to be optimized
ONE_MAX_LENGTH = 100

# Genetic Algorithm constants:

# number of individuals in population
POPULATION_SIZE = 200

# probability for crossover
P_CROSSOVER = 0.9

# probability for mutating
P_MUTATION = 0.1

# max number of generations for stopping condition
MAX_GENERATIONS = 50


# Fitness function definition
def oneMaxFitness(individual):
    return (sum(individual),)  # return a tuple


def main():
    toolbox = base.Toolbox()
    toolbox.register("zeroOrOne", random.randint, 0, 1)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    # Individual Constructor register
    toolbox.register(
        "individualCreator",
        tools.initRepeat,
        creator.Individual,
        toolbox.zeroOrOne,
        ONE_MAX_LENGTH,
    )
    # Population Constructor register
    toolbox.register(
        "populationCreator", tools.initRepeat, list, toolbox.individualCreator
    )
    # Evaluation function register
    toolbox.register("evaluate", oneMaxFitness)
    # Selection function register
    toolbox.register("select", tools.selTournament, tournsize=3)
    # Crossover function register
    toolbox.register("mate", tools.cxOnePoint)
    # Mutation function register
    toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / ONE_MAX_LENGTH)

    generationCounter = 0
    maxFitnessValues = []
    meanFitnessValues = []

    population = toolbox.populationCreator(n=POPULATION_SIZE)
    fitnessValues = list(map(toolbox.evaluate, population))
    # Assigning initial fitness to individual
    for individual, fitnessValue in zip(population, fitnessValues):
        individual.fitness.values = fitnessValue
    fitnessValues = [individual.fitness.values[0] for individual in population]

    while max(fitnessValues) < ONE_MAX_LENGTH and generationCounter < MAX_GENERATIONS:
        generationCounter = generationCounter + 1
        # New generation
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        # Crossover and delete invalid fitness
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < P_CROSSOVER:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        # Mutation and delete invalid fitness
        for mutant in offspring:
            if random.random() < P_MUTATION:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Take individual with invalid fitness
        freshIndividuals = [ind for ind in offspring if not ind.fitness.valid]
        # COmpute new fitness
        freshFitnessValues = list(map(toolbox.evaluate, freshIndividuals))
        # Assign new fitness
        for individual, fitnessValue in zip(freshIndividuals, freshFitnessValues):
            individual.fitness.values = fitnessValue

        population[:] = offspring
        # Take all fitness
        fitnessValues = [ind.fitness.values[0] for ind in population]
        # Compute max and mean
        maxFitness = max(fitnessValues)
        meanFitness = sum(fitnessValues) / len(population)
        maxFitnessValues.append(maxFitness)
        meanFitnessValues.append(meanFitness)
        print(
            "- Generation {}: Max Fitness = {}, Avg Fitness = {}".format(
                generationCounter, maxFitness, meanFitness
            )
        )
        # In addition, we locate the index of the (first) best individual using the max fitness
        best_index = fitnessValues.index(max(fitnessValues))
        print("Best Individual = ", *population[best_index], "\n")

    plt.plot(maxFitnessValues, color="red")
    plt.plot(meanFitnessValues, color="green")
    plt.xlabel("Generation")
    plt.ylabel("Max / Average Fitness")
    plt.title("Max and Average fitness over Generations")
    plt.show()


if __name__ == "__main__":
    main()
