import random
import matplotlib.pyplot as plt

# a. Input data for the Knapsack problem
max_capacity = 400  # max weight allowed in knapsack
filename = "tourist.txt"
items = []
with open(filename, "r") as file:
    for line in file:
        weight, value = map(int, line.strip().split())
        items.append({"weight": weight, "value": value})

# b. Set GA parameters
population_size = 50
num_generations = 200
mutation_rate = 0.1  # probability of mutation
isRoulette = False  # if False, it uses the tournament selection procedure
tournament_size = 3  # in case tournament method is used
isOnePoint = False  # if False, it uses the two-point crossover
elite_count = int(0.05 * population_size)  # number of elite solutions to preserve

# c. Initialize key variables
best_sol = None  # configuration of the best overall sol
best_value = 0  # value of the best overall sol
average_values = []  # list of average values at each generation
best_values = []  # list of best values at each generation

# a. Initialize population (list of sols, each sol being a k-dim vector of 0s and 1s)
population = [random.choices([0, 1], k=len(items)) for _ in range(population_size)]


# 1b. Evaluate fitness for each sol
fitness_values = []
for sol in population:
    total_weight = sum(items[i]["weight"] for i, bit in enumerate(sol) if bit)
    if total_weight > max_capacity:  # infeasible sol
        fitness_values.append(0)
    else:
        fitness_values.append(
            sum(items[i]["value"] for i, bit in enumerate(sol) if bit)
        )


# 2. Define selection process as well as crossover and mutation operators
def selection(population, fitness_values, k, isRoulette):
    if isRoulette:  # select k parents with prob. selection proportional to fitness
        parents = random.choices(population, weights=fitness_values, k=k)
        return parents
    else:  # select k parents using a tournament selection process
        parents = []
        for _ in range(k):
            tournament_candidates = random.sample(
                range(len(population)), tournament_size
            )
            tournament_fitness = [fitness_values[i] for i in tournament_candidates]
            winner_index = tournament_candidates[
                tournament_fitness.index(max(tournament_fitness))
            ]
            parents.append(population[winner_index])
        return parents


def crossover(parent1, parent2):
    if isOnePoint:  # one-point crossover
        split_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:split_point] + parent2[split_point:]
        child2 = parent2[:split_point] + parent1[split_point:]
        return child1, child2
    else:  # two-point crossover
        point1 = random.randint(1, len(parent1) - 1)
        point2 = random.randint(point1, len(parent1) - 1)
        child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
        return child1, child2


def mutate(chromosome):
    mutated_chromosome = list(chromosome)
    for i in range(len(mutated_chromosome)):
        if random.random() < mutation_rate:
            mutated_chromosome[i] = 1 - mutated_chromosome[i]
    return mutated_chromosome


# 3. Evolve the population for specified number of generations
for generation in range(num_generations):
    # 3a. Select k parents for crossover, either using roulette or tournament
    # select elite solutions
    elite_indices = sorted(
        range(len(fitness_values)), key=lambda i: fitness_values[i], reverse=True
    )[:elite_count]
    elite_solutions = [population[i] for i in elite_indices]
    # select parents for crossover and mutation
    non_elite_parents = selection(
        population, fitness_values, population_size - elite_count, isRoulette
    )
    # get the entire population of parents
    parents = elite_solutions + non_elite_parents

    # 3b. Create new generation through crossover and mutation
    # Create new generation through crossover and mutation
    children = elite_solutions  # Preserve elite solutions
    for i in range(0, population_size - elite_count, 2):
        child1, child2 = crossover(parents[i], parents[i + 1])
        children.append(mutate(child1))
        children.append(mutate(child2))

    # 3c. Replace old population with new generation
    population = children

    # 3d. Evaluate fitness for each sol
    fitness_values = []
    for sol in population:
        total_weight = sum(items[i]["weight"] for i, bit in enumerate(sol) if bit)
        if total_weight > max_capacity:  # infeasible sol
            fitness_values.append(0)
        else:
            fitness_values.append(
                sum(items[i]["value"] for i, bit in enumerate(sol) if bit)
            )

    # 3e. Return the best sol in the current generation
    best_index = fitness_values.index(max(fitness_values))
    gen_best_sol = population[best_index]
    gen_best_value = fitness_values[best_index]
    print(f"Generation {generation+1}: Best value in gen = {gen_best_value}")

    # 3f. Update best global sol if appropriate
    if gen_best_value > best_value:
        best_sol = population[best_index]
        best_value = gen_best_value

    # 3g. Compute average value of the current generation and update lists
    average_value = sum(fitness_values) / len(fitness_values)
    average_values.append(average_value)
    best_values.append(best_value)

# 4. Print configuration and value of the best sol found during the entire search
print(f"Best sol overall: {best_sol}")
print(f"Best value overall: {best_value}")

# 5. Plot the average value and best value for each generation
generations = list(range(1, num_generations + 1))
plt.plot(generations, average_values, label="Average Value")
plt.plot(generations, best_values, label="Best Value")
plt.xlabel("Generation")
plt.ylabel("Value")
plt.title("Evolution of Average and Best Values")
plt.legend()
plt.ylim(bottom=0)
plt.show()
