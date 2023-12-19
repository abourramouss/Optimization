from PFSP_elements import Job, Solution
import operator
import time

""" NEH HEURISTIC FOR THE PERMUTATION FLOW-SHOP PROBLEM (PFSP) """

""" Read instance data from txt file """
instanceName = "tai117_500_20"  # name of the instance
# txt file with the PFSP instance data for each job-machine
fileName = "data/" + instanceName + "_inputs.txt"

with open(fileName) as instance:
    i = -3  # we start at -3 so that the first job is job 0
    jobs = []
    for line in instance:
        if i == -3:
            pass  # line 0 contains a symbol #
        elif i == -2:
            nJobs = int(line.split()[0])
            nMachines = int(line.split()[1])
        elif i == -1:
            pass
        else:
            # array data with processing time of job i in machine j
            data = [float(x) for x in line.split("\t")]
            TPT = sum(data)  # total processing time of this job in the system
            aJob = Job(i, data, TPT)
            jobs.append(aJob)
        i += 1


def calcEMatrix(aSol, k):
    nRows = k
    nCols = nMachines
    e = [[0 for j in range(nCols)] for i in range(nRows)]
    for i in range(k):
        for j in range(nMachines):
            if i == 0 and j == 0:
                e[i][j] = aSol.jobs[0].processingTimes[0]
            elif j == 0:
                e[i][j] = e[i - 1][j] + aSol.jobs[i].processingTimes[j]
            elif i == 0:
                e[i][j] = e[i][j - 1] + aSol.jobs[i].processingTimes[j]
            else:
                maxTime = max(e[i - 1][j], e[i][j - 1])
                e[i][j] = maxTime + aSol.jobs[i].processingTimes[j]
    return e


def calcQMatrix(aSol, k):
    nRows = k + 1
    nCols = nMachines
    q = [[0 for j in range(nCols)] for i in range(nRows)]
    for i in range(k, -1, -1):
        for j in range(nMachines - 1, -1, -1):
            if i == k:
                q[k][j] = 0  # dummy file to make possible fMatrix + qMatrix
            elif i == k - 1 and j == nMachines - 1:
                q[k - 1][nMachines - 1] = aSol.jobs[k].processingTimes[nMachines - 1]
            elif j == nMachines - 1:
                q[i][nMachines - 1] = (
                    q[i + 1][nMachines - 1]
                    + aSol.jobs[i].processingTimes[nMachines - 1]
                )
            elif i == k - 1:
                q[k - 1][j] = q[k - 1][j + 1] + aSol.jobs[k].processingTimes[j]
            else:
                maxTime = max(q[i + 1][j], q[i][j + 1])
                q[i][j] = maxTime + aSol.jobs[i].processingTimes[j]
    return q


def calcFMatrix(aSol, k, e):
    nRows = k + 1
    nCols = nMachines
    f = [[0 for j in range(nCols)] for i in range(nRows)]
    for i in range(nCols):
        for j in range(k + 1):
            if i == 0 and j == 0:
                f[0][0] = aSol.jobs[k].processingTimes[0]
            elif j == 0:
                f[i][0] = e[i - 1][0] + aSol.jobs[k].processingTimes[0]
            elif i == 0:
                f[0][j] = f[0][j - 1] + aSol.jobs[k].processingTimes[j]
            else:
                maxTime = max(e[i - 1][j], f[i][j - 1])
                f[i][j] = maxTime + aSol.jobs[k].processingTimes[j]
    return f


def improveByShiftingJobToLeft(aSol, k):
    """
    Employs Taillard's accelerations to compute the best position for a job,
    optimizes the total time; also updates makespan if k == nJobs - 1
    """
    # Implements Taillard's acceleration, where k is the position of the job on the right that's to the left, makespan if k == nJobs - 1
    bestPosition = k
    minMakespan = float("inf")
    eMatrix = calcEMatrix(aSol, k)
    qMatrix = calcQMatrix(aSol, k)
    fMatrix = calcFMatrix(aSol, eMatrix)
    # compute bestPosition (0...k) and minMakespan (mVector)
    for i in range(k, -1, -1):
        maxSum = 0.0
        for j in range(aSol.nMachines):
            newSum = fMatrix[i][j] + qMatrix[i][j]
            if newSum > maxSum:
                maxSum = newSum
            newMakespan = maxSum
        # TIE ISSUE No.2 - In case of tie, do swap (it might affect the final result)
        if newMakespan <= minMakespan:
            minMakespan = newMakespan
            bestPosition = i
    # update solution with bestPosition and minMakespan
    if bestPosition < k:  # if i == k do nothing
        auxJob = aSol.jobs[k]
        for i in range(k, bestPosition, -1):
            aSol.jobs[i] = aSol.jobs[i - 1]
        aSol.jobs[bestPosition] = auxJob
    if k == aSol.nJobs - 1:
        aSol.makespan = minMakespan
    return aSol
