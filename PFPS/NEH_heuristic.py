from PFPS_elements import Job, Solution
import operator
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

""" NEH HEURISTIC FOR THE PERMUTATION FLOW-SHOP PROBLEM (PFSP) """


def calcEMatrix(aSol, k, nMachines):
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


def calcQMatrix(aSol, k, nMachines):
    nRows = k + 1
    nCols = nMachines
    q = [[0 for j in range(nCols)] for i in range(nRows)]
    for i in range(k, -1, -1):
        for j in range(nMachines - 1, -1, -1):
            if i == k:
                q[k][j] = 0  # dummy file to make possible fMatrix + qMatrix
            elif i == k - 1 and j == nMachines - 1:
                q[k - 1][nMachines - 1] = aSol.jobs[k - 1].processingTimes[
                    nMachines - 1
                ]
            elif j == nMachines - 1:
                q[i][nMachines - 1] = (
                    q[i + 1][nMachines - 1]
                    + aSol.jobs[i].processingTimes[nMachines - 1]
                )
            elif i == k - 1:
                q[k - 1][j] = q[k - 1][j + 1] + aSol.jobs[k - 1].processingTimes[j]
            else:
                maxTime = max(q[i + 1][j], q[i][j + 1])
                q[i][j] = maxTime + aSol.jobs[i].processingTimes[j]
    return q


def calcFMatrix(aSol, k, e, nMachines):
    nRows = k + 1
    nCols = nMachines
    f = [[0 for j in range(nCols)] for i in range(nRows)]
    for i in range(k + 1):
        for j in range(nMachines):
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


def improveByShiftingJobToLeft(aSol, k, nMachines):
    """
    Employs Taillard's accelerations to compute the best position for a job,
    optimizes the total time; also updates makespan if k == nJobs - 1
    """
    # Implements Taillard's acceleration, where k is the position of the job on the right that's to the left, makespan if k == nJobs - 1
    bestPosition = k
    minMakespan = float("inf")
    eMatrix = calcEMatrix(aSol, k, nMachines)
    qMatrix = calcQMatrix(aSol, k, nMachines)
    fMatrix = calcFMatrix(aSol, k, eMatrix, nMachines)
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


def run_NEH(instance_name, file_name):
    with open(file_name) as instance:
        jobs = []
        nJobs, nMachines = 0, 0  # Initialize these to ensure they are defined
        for i, line in enumerate(instance):
            if i == 1:
                nJobs, nMachines = map(int, line.split())
            elif i > 2:
                data = [float(x) for x in line.split("\t")]
                TPT = sum(data)
                jobs.append(Job(i - 3, data, TPT))

    tStart = time.time()
    jobs.sort(key=operator.attrgetter("TPT"), reverse=True)
    sol = Solution(nJobs, nMachines)
    sol.jobs.append(jobs[0])

    for i in range(1, nJobs):
        sol.jobs.append(jobs[i])
        sol = improveByShiftingJobToLeft(sol, i, nMachines)

    tEnd = time.time()
    return {
        "instance": instance_name,
        "makespan": sol.makespan,
        "computational_time": tEnd - tStart,
        "nJobs": nJobs,  # Include number of jobs
        "nMachines": nMachines,  # Include number of machines
        "solution": [job.ID for job in sol.jobs],
    }


def analyze_instances(instance_names, base_path="data/"):
    results = []
    for instance_name in instance_names:
        file_name = f"{base_path}{instance_name}_inputs.txt"
        result = run_NEH(instance_name, file_name)
        results.append(result)
    return results


# List of instances to analyze
instance_names = [
    "tai117_500_20",
    "tai004_20_5",
    "tai018_20_10",
    "tai026_20_20",
    "tai034_50_5",
    "tai046_50_10",
    "tai063_100_5",
    "tai108_200_20",
]
results = analyze_instances(instance_names)

# Convert the results list to a DataFrame
df = pd.DataFrame(results)

# Adding number of jobs and machines as columns
df["nJobs"] = [result["nJobs"] for result in results]
df["nMachines"] = [result["nMachines"] for result in results]

# Computational Time per Job
df["time_per_job"] = df["computational_time"] / df["nJobs"]

# Setting the aesthetic style of the plots
# Setting the aesthetic style of the plots
sns.set_style("whitegrid")

# Makespan vs Number of Jobs
plt.figure(figsize=(10, 6))  # Adjust the figure size
sns.scatterplot(x="nJobs", y="makespan", data=df, hue="instance")
plt.xlabel("Number of Jobs")
plt.ylabel("Makespan")
plt.title("Makespan vs Number of Jobs for Each Instance")
plt.legend(title="Instance", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.savefig(
    "makespan_vs_jobs.png", bbox_inches="tight"
)  # Save with bbox_inches="tight"
plt.show()

# Computational Time per Job Plot
plt.figure(figsize=(12, 6))  # Adjust the figure size
sns.barplot(x="instance", y="time_per_job", data=df)
plt.xticks(rotation=45)
plt.xlabel("Instance")
plt.ylabel("Computational Time per Job (seconds)")
plt.title("Computational Time per Job Across Instances")
plt.savefig(
    "comp_time_per_job.png", bbox_inches="tight"
)  # Save with bbox_inches="tight"
plt.show()

# Statistical Summary
summary = df[["makespan", "computational_time"]].describe()
print("Statistical Summary:\n", summary)

# Box Plot for Makespan and Computational Time
plt.figure(figsize=(8, 6))  # Adjust the figure size
sns.boxplot(data=df[["makespan", "computational_time"]])
plt.title("Box Plot for Makespan and Computational Time")
plt.savefig(
    "boxplot_analysis.png", bbox_inches="tight"
)  # Save with bbox_inches="tight"
plt.show()
