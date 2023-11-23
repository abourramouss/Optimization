from shared import basinFunction, basinFunction1
import random, time


def randomSolution(searchSpace, problemSize):
    min = searchSpace[0]
    max = searchSpace[1]
    inputValues = []
    for i in range(0, problemSize):
        inputValues.append(min + (max - min) * random.random())
    return inputValues


algorithmName = "Random search"
searchSpace = [-5, 5]
maxIterations = 1000
problemSize = 2
print("Best Sol by" + algorithmName + "...")

start = time.time()
bestCost = float("inf")  # infinity

while maxIterations > 0:
    maxIterations -= 1
    newSol = randomSolution(searchSpace, problemSize)
    newCost = basinFunction1(newSol)

    if newCost < bestCost:
        bestSol = newSol
        bestCost = newCost

stop = time.time()
print("Cost =", bestCost)
print("Sol =", bestSol)
print("Elapsed =", stop - start)
