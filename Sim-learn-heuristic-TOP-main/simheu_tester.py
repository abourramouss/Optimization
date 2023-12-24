''' Reviewed by Angel A. Juan 2023.06 for the TOP with stochastic / dynamic travel times '''

import os
import random
import numpy as np

from aux_functions import read_tests, read_instance, printRoutes
from simheu import algorithm

# Read the tests2run.txt file and build the list of instances (tests) to run
file_name = "tests" + os.sep + "tests2run.txt"
tests = read_tests(file_name)

# For each instance (test), read inputs, create nodes, and execute it
for test in tests:
    # set the seed in the RNG for reproducibility purposes
    random.seed(test.seed) # Python default RNG, used during BR
    np.random.seed(test.seed) # Numpy RNG, used during simulation
    # print basic instance info
    print('\nInstance: ', test.instanceName)
    print('Var level (k in Var = k*mean):', test.varLevel)
    # read input data from instance file
    file_name = "data" + os.sep + test.instanceName + ".txt"
    fleetSize, routeMaxCost, nodes = read_instance(file_name)
    # execute the algorithm and obtain the different Our Best solutions, where
    # OBD = Our Best Deterministic sol
    # OBS = Our Best Stochastic sol
    OBD, OBS = algorithm(test, fleetSize, routeMaxCost, nodes)
    # Print summary results
    print('Reward for OBD sol in a Det. env. =', OBD.reward)
    print('Reward for OBD sol in a Stoch. env. =', OBD.reward_sim)
    print('Reward for OBS sol in a Stoch. env. =', OBS.reward_sim)
    print('Routes for OBD sol')
    printRoutes(OBD)
    print('Routes for OBS sol')
    printRoutes(OBS)
