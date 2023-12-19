import time, random
from Shared import (
    berlin52,
    bayg29,
    bays29,
    tourCost,
    stochasticTwoOpt,
    constructInitialSolution,
)


def perturbation(aSol):
    newSol = doubleBridgeMove(aSol)
    newSolCost = tourCost(newSol)
    return newSol, newSolCost


def doubleBridgeMove(perm):
    sliceLength = len(perm) // 4
    p1 = 1 + random.randrange(0, sliceLength)
    p2 = p1 + 1 + random.randrange(0, sliceLength)
    p3 = p2 + 1 + random.randrange(0, sliceLength)

    return perm[0:p1] + perm[p3:] + perm[p2:p3] + perm[p1:p2]


def tripleBridgeMove(perm):
    size = len(perm)
    sliceLength = size // 6  # Shorter slices, more segments
    p1 = 1 + random.randrange(0, sliceLength)
    p2 = p1 + 1 + random.randrange(0, sliceLength)
    p3 = p2 + 1 + random.randrange(0, sliceLength)
    p4 = p3 + 1 + random.randrange(0, sliceLength)
    p5 = p4 + 1 + random.randrange(0, sliceLength)

    # Rearrange the segments to introduce more change
    newPerm = (
        perm[0:p1] + perm[p4:p5] + perm[p2:p3] + perm[p5:] + perm[p1:p2] + perm[p3:p4]
    )
    return newPerm


def dynamicPerturbation(aSol, iter_since_improvement, threshold):
    if iter_since_improvement > threshold:
        newSol = tripleBridgeMove(aSol)
    else:
        newSol = doubleBridgeMove(aSol)
    newSolCost = tourCost(newSol)
    return newSol, newSolCost


def localSearch(aSol, aCost, maxIter):
    while maxIter > 0:
        maxIter -= 1
        newSol = stochasticTwoOpt(aSol)
        newCost = tourCost(newSol)
        if newCost < aCost:
            aSol, aCost = newSol, newCost
    return aSol, aCost


algorithmName = "ILS"
print(f"Best Sol by {algorithmName}")

inputsTSP = bays29
maxIterations = 10000
maxNoImprove = 50

start = time.time()

bestSol = constructInitialSolution(inputsTSP)
bestCost = tourCost(bestSol)

bestSol, bestCost = localSearch(bestSol, bestCost, maxNoImprove)

while maxIterations > 50:
    maxIterations -= 1
    newSol, newCost = perturbation(bestSol)

    newSol, newCost = localSearch(newSol, newCost, maxNoImprove)
    if newCost < bestCost:
        bestSol, bestCost = newSol, newCost
        print(bestCost, maxIterations)

stop = time.time()

print(f"Cost = {bestCost}, sol = {bestSol}, elapsed = {stop - start}")
