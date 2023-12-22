from Shared import berlin52, stochasticTwoOpt, tourCost, euclideanDistance
import random, time


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

# Problem configuration
inputsTSP = berlin52
maxIterations = 10000
maxNoImprove = 50
greedinessFactor = 0.14  # In the range [0,1]. 0 is more greedy and 1 less greedy
start = time.time()

# Main Loop
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
print("BestCost = %.2f; Elapsed = %.2fs" % (bestCost, stop - start))
print("BestSol = %s" % bestSol)
