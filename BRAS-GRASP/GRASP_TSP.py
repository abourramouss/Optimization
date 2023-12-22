from Shared import berlin52, stochasticTwoOpt, tourCost, euclideanDistance
import random, time, math
import matplotlib.pyplot as plt


# Aux Funct to Apply a Local Search
def localSearch(aSol, aCost, maxIter):
    count = 0
    while count < maxIter:
        newSol = stochasticTwoOpt(aSol)
        newCost = tourCost(newSol)
        if newCost < aCost:  # Restart the search when we find an improvement
            aSol = newSol
            aCost = newCost
            count = 0
        else:
            count += 1
    # return solution and cost
    return aSol, aCost


def gaussianProbability(x, mu, sigma):
    return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )


def biasedRandomSelectionGaussian(distances):
    mu = sum(distances) / len(distances)
    variance = sum((x - mu) ** 2 for x in distances) / len(distances)
    sigma = math.sqrt(variance)

    # Ensure sigma is not zero to avoid division by zero
    if sigma == 0:
        sigma = 0.0001  # A small positive number
    probabilities = [gaussianProbability(d, mu, sigma) for d in distances]
    total = sum(probabilities)
    probabilities = [p / total for p in probabilities]

    return random.choices(range(len(probabilities)), probabilities, k=1)[0]


def constructGreedySolutionWithBRAs(perm):
    emergingSol = [perm[random.randrange(0, len(perm))]]
    problemSize = len(perm)

    while len(emergingSol) < problemSize:
        notInSolNodes = [node for node in perm if node not in emergingSol]
        distances = [euclideanDistance(emergingSol[-1], node) for node in notInSolNodes]

        nextNodeIndex = biasedRandomSelectionGaussian(distances)
        emergingSol.append(notInSolNodes[nextNodeIndex])

    newCost = tourCost(emergingSol)
    return emergingSol, newCost


def constructGreedySolution(perm, alpha):
    # Select one node randomly and incorporate it to the emerging sol
    emergingSol = [perm[random.randrange(0, len(perm))]]
    problemSize = len(perm)
    # While sol size is not equal to the original permutation size
    while len(emergingSol) < problemSize:
        # Get all nodes not already in the emerging sol
        notInSolNodes = [node for node in perm if node not in emergingSol]
        # For each node not in emergingSol, compute distance w.r.t. last element
        costs = []
        emergingSolSize = len(emergingSol)
        for node in notInSolNodes:
            costs.append(euclideanDistance(emergingSol[emergingSolSize - 1], node))
        # Determining the max cost and min cost from the feature set
        maxCost, minCost = max(costs), min(costs)
        # Build the RCL by adding the nodes satisfying the condition
        rcl = []
        for index, cost in enumerate(costs):
            if cost <= minCost + alpha * (maxCost - minCost):
                # Add it to the RCL
                rcl.append(notInSolNodes[index])
        # Select random feature from RCL and add it to the solution
        emergingSol.append(rcl[random.randrange(0, len(rcl))])
    # Calculate the final tour cost before returning the new solution
    newCost = tourCost(emergingSol)
    # return solution and cost
    return emergingSol, newCost


algorithmName = "GRASP"
print("Best Sol by " + algorithmName + " ...")

inputsTSP = berlin52
maxIterations = 100
maxNoImprove = 50
greedinessFactor = 0.30  # In the range [0,1]. 0 is more greedy and 1 less greedy


def runGaussianBRAsGRASP(maxIterations, maxNoImprove, greedinessFactor):
    # Problem configuration

    start = time.time()

    # Main Loop
    bestCost = float("inf")  # infinity
    while maxIterations > 0:
        maxIterations -= 1
        # Construct greedy solution
        newSol, newCost = constructGreedySolutionWithBRAs(inputsTSP)
        # refine it using a local search heuristic
        newSol, newCost = localSearch(newSol, newCost, maxNoImprove)
        if newCost < bestCost:
            bestSol = newSol
            bestCost = newCost
        print("Cost = %.2f; Iter = %d" % (bestCost, maxIterations))

    stop = time.time()
    exec_time = stop - start
    print("BestCost GAUSSIAN = %.2f; Elapsed = %.2fs" % (bestCost, exec_time))
    print("BestSol = %s" % bestSol)
    return bestCost, exec_time


def runOriginalGRASP(maxIterations, maxNoImprove, greedinessFactor):
    # Main Loop
    start = time.time()
    bestCost = float("inf")  # infinity
    while maxIterations > 0:
        maxIterations -= 1
        # Construct greedy solution
        newSol, newCost = constructGreedySolution(inputsTSP, greedinessFactor)
        # refine it using a local search heuristic
        newSol, newCost = localSearch(newSol, newCost, maxNoImprove)
        if newCost < bestCost:
            bestSol = newSol
            bestCost = newCost
        print("Cost = %.2f; Iter = %d" % (bestCost, maxIterations))

    stop = time.time()
    exec_time = stop - start
    print("BestCost Greedy = %.2f; Elapsed = %.2fs" % (bestCost, exec_time))
    print("BestSol = %s" % bestSol)
    return bestCost, exec_time


num_runs = 30  # Number of runs for each version
original_costs = []
original_times = []
gaussian_costs = []
gaussian_times = []

for _ in range(num_runs):
    cost, exec_time = runOriginalGRASP(maxIterations, maxNoImprove, greedinessFactor)
    original_costs.append(cost)
    original_times.append(exec_time)

    cost, exec_time = runGaussianBRAsGRASP(
        maxIterations, maxNoImprove, greedinessFactor
    )
    gaussian_costs.append(cost)
    gaussian_times.append(exec_time)


# Plot for Solution Quality
plt.figure()
plt.boxplot([original_costs, gaussian_costs], labels=["Original", "Gaussian BRAs"])
plt.title("Solution Quality Comparison")
plt.ylabel("Tour Cost")
plt.savefig("solution_quality_comparison.png")

# Plot for Computational Efficiency
plt.figure()
plt.boxplot([original_times, gaussian_times], labels=["Original", "Gaussian BRAs"])
plt.title("Computational Efficiency Comparison")
plt.ylabel("Execution Time (s)")
plt.savefig("computational_efficiency_comparison.png")
