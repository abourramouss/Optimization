# Input data (nodes) for the TSP
# The optimal solution (using real numbers for distances) is 7544.3659 according to:
# https://www.researchgate.net/figure/The-optimal-solution-of-Berlin52-fig2_221901574
import random
import math

berlin52 = [
    [565, 575],
    [25, 185],
    [345, 750],
    [945, 685],
    [845, 655],
    [880, 660],
    [25, 230],
    [525, 1000],
    [580, 1175],
    [650, 1130],
    [1605, 620],
    [1220, 580],
    [1465, 200],
    [1530, 5],
    [845, 680],
    [725, 370],
    [145, 665],
    [415, 635],
    [510, 875],
    [560, 365],
    [300, 465],
    [520, 585],
    [480, 415],
    [835, 625],
    [975, 580],
    [1215, 245],
    [1320, 315],
    [1250, 400],
    [660, 180],
    [410, 250],
    [420, 555],
    [575, 665],
    [1150, 1160],
    [700, 580],
    [685, 595],
    [685, 610],
    [770, 610],
    [795, 645],
    [720, 635],
    [760, 650],
    [475, 960],
    [95, 260],
    [875, 920],
    [700, 500],
    [555, 815],
    [830, 485],
    [1170, 65],
    [830, 610],
    [605, 625],
    [595, 360],
    [1340, 725],
    [1740, 245],
]


def locateBestNewSol(newSols):
    newSols.sort(key=lambda c: c["cost"])
    bestNewSol = newSols[0]
    return bestNewSol


# Creates a random solution (permutation) from an initial permutation by shuffling its elements
def constructInitialSolution(initPerm):
    # Randomize the initial permutation
    permutation = initPerm[:]  # make a copy of the initial permutation
    size = len(permutation)

    for index in range(size):
        # shuffle the values of the initial permutation randomly
        # get a random index and exchange values
        shuffleIndex = random.randrange(
            index, size
        )  # randrange would exclude the upper bound
        permutation[shuffleIndex], permutation[index] = (
            permutation[index],
            permutation[shuffleIndex],
        )

    return permutation


# Evaluates the total length of a TSP solution (permutation of nodes)
def tourCost(perm):
    # Tour cost is the sum of the euclidean distance between consecutive points in the path
    totalDistance = 0.0
    size = len(perm)

    for index in range(size):
        startNode = perm[index]
        # select the end point point for calculating the segment length
        if index == size - 1:
            # In order to complete the 'tour' we need to reach the starting point
            endNode = perm[0]
        else:
            # select the next point
            endNode = perm[index + 1]

        totalDistance += euclideanDistance(startNode, endNode)

    return totalDistance


# Calculates the euclidean distance between two points
def euclideanDistance(xNode, yNode):
    sum = 0.0
    # use Zip to iterate over the two vectors (nodes)
    for xi, yi in zip(xNode, yNode):
        sum += pow((xi - yi), 2)
    return math.sqrt(sum)


def stochasticTwoOptWithEdges(perm):
    result = perm[:]  # make a copy
    size = len(result)
    # select indices of two random points in the tour
    p1, p2 = random.randrange(0, size), random.randrange(0, size)
    # do this so as not to overshoot tour boundaries
    exclude = set([p1])
    if p1 == 0:
        exclude.add(size - 1)
    else:
        exclude.add(p1 - 1)
    if p1 == size - 1:
        exclude.add(0)
    else:
        exclude.add(p1 + 1)

    while p2 in exclude:
        p2 = random.randrange(0, size)

    # to ensure we always have p1<p2
    if p2 < p1:
        p1, p2 = p2, p1

    # now reverse the tour segment between p1 and p2
    result[p1:p2] = reversed(result[p1:p2])

    return result, [[perm[p1 - 1], perm[p1]], [perm[p2 - 1], perm[p2]]]


def generateNewSol(baseSol, bestSol, tabuList):
    newPermutation, edges, newSol = None, None, {}
    # Generates a new permutation that does not include any edge from the tabu list
    while newPermutation is None or isTabu(newPermutation, tabuList):
        newPermutation, edges = stochasticTwoOptWithEdges(baseSol["permutation"])
        # If the move generates a solution with a better cost than the best one, always accept it (regardless of if it is tabu or not)
        if tourCost(newPermutation) < bestSol["cost"]:
            break

    newSol["permutation"] = newPermutation
    newSol["cost"] = tourCost(newSol["permutation"])
    newSol["edges"] = edges
    return newSol


# Returns True if aPermutation contains an edge in tabuList; returns False otherwise
def isTabu(aPermutation, tabuList):
    size = len(aPermutation)
    for index, node in enumerate(aPermutation):
        if index == size - 1:
            nextNode = aPermutation[0]
        else:
            nextNode = aPermutation[index + 1]
        edge = [node, nextNode]
        if edge in tabuList:
            return True
    return False


# ALGORITHM FRAMEWORK
algorithmName = "TABU SEARCH"
print("Best Sol by " + algorithmName + "...")

inputsTSP = berlin52  # TSP instance
maxNewSols = 40  # max number of newSols to generate
maxIterations = 5000  # number of iterations in the main loop
maxEdgesInTabuList = 10  # permutation edges in the tabu list
k = 5  # multiplier for the demon-based acceptance criterion

# a sol as a dictionary including the permutation of nodes, its associated cost, and two edges if
# sol has been generated in a 2-opt local search
bestSol = {}

# construct an initial random sol (permutation of nodes, each one defined by two coordinates)
bestSol["permutation"] = constructInitialSolution(
    inputsTSP
)  # generate a random permutation
bestSol["cost"] = tourCost(bestSol["permutation"])  # computes its associated cost
bestSol["edges"] = None  # this sol has not been generated by a 2-opt local search
baseSol = bestSol

credit = 0  # credit for the demon-based acceptance criterion
tabuList = []  # list of tabu edges

while maxIterations > 0:
    # Generates newSols in the neighborhood of bestSol by using stochastic 2-opt; the two edges employed
    # in the 2-opt local search are added to the tabu list, so they cannot appear in future newSols
    # Then, chooses the best newSol in newSols
    newSols = []
    for index in range(0, maxNewSols):
        # Generate a newSol that does not use any tabu edge and add it to a list of newSols
        # In the first iteration, the tabu list is empty
        newSol = generateNewSol(baseSol, bestSol, tabuList)
        newSols.append(newSol)

    # Locate the bestNewSol among the newSols
    bestNewSol = locateBestNewSol(newSols)

    # Compare with current baseSol and update if necessary
    # Worse solutions are accepted as the baseSol according to an acceptable
    # worsening margin (using delta and credit variables)
    delta = bestNewSol["cost"] - baseSol["cost"]
    if delta <= 0:
        credit = -1 * delta
        baseSol = bestNewSol

        if bestNewSol["cost"] < bestSol["cost"]:
            bestSol = bestNewSol
            print("iter:", maxIterations, "cost: %.2f" % bestSol["cost"])

            for edge in bestNewSol["edges"]:
                tabuList.append(edge)
                if len(tabuList) > maxEdgesInTabuList:
                    del tabuList[0]
    else:
        if delta <= k * credit:
            credit = 0
            baseSol = bestNewSol

    maxIterations -= 1
