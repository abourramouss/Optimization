class Job:
    def __init__(self, ID, processingTimes, TPT):
        self.ID = ID  # job ID
        self.processingTimes = processingTimes  # list of processing times
        self.TPT = TPT  # total processing time of job in system


class Solution:
    last_ID = -1  # counts the number of solution objects generated, starting with ID 0

    def __init__(self, nJobs, nMachines):
        Solution.last_ID += 1
        self.ID = Solution.last_ID
        self.nJobs = nJobs
        self.nMachines = nMachines
        self.jobs = []  # list of sorted jobs
        self.makespan = 0.0  # makespan of this solution

    def calcMakespan(self):
        # computes makespan, used only to confirm the value provided by Taillard acceleration matrices
        nRows = self.nJobs
        nCols = self.nMachines
        times = [[0 for j in range(nCols)] for i in range(nRows)]
        for column in range(nCols):
            for row in range(nRows):
                if column == 0 and row == 0:
                    times[row][column] = self.jobs[0].processingTimes[0]
                elif column == 0:
                    times[row][column] = (
                        times[row - 1][column] + self.jobs[row].processingTimes[column]
                    )
                elif row == 0:
                    times[row][column] = (
                        times[row][column - 1] + self.jobs[row].processingTimes[column]
                    )
                else:
                    maxTime = max(times[row - 1][column], times[row][column - 1])
                    times[row][column] = (
                        maxTime + self.jobs[row].processingTimes[column]
                    )
        return times[nRows - 1][nCols - 1]
