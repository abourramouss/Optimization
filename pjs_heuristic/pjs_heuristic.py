""" PANADERO & JUAN SAVINGS HEURISTIC FOR THE TEAM ORIENTEERING PROBLEM (TOP) """

import networkx as nx
from routing_objects import Node, Edge, Route, Solution
import math
import operator
import time
import matplotlib.pyplot as plt

""" Set algorithm parameters """
# alpha is used to compute the edge efficiency (enriched savings), its best value
# might depend on the specific instance as explained in Panadero et al. (2020)
alpha = 0.7

""" Read instance data from txt file """
instanceName = "p7.4.e"  # name of the instance
# txt file with the TOP instance data
fileName = "data/" + instanceName + ".txt"

with open(fileName) as instance:
    i = -3  # we start at -3 so that the first node is node 0
    nodes = []
    for line in instance:
        if i == -3:
            pass  # Line 0 contains the number of nodes, not needed
        elif i == -2:
            fleetSize = int(line.split(";")[1])
        elif i == -1:
            routeMaxCost = float(line.split(";")[1])
        else:
            # array data with node data: x, y, demand (reward in TOP)
            data = [float(x) for x in line.split(";")]
            aNode = Node(i, data[0], data[1], data[2])
            nodes.append(aNode)
        i += 1

startTime = time.time()
""" Construct edges with costs and efficiency list from nodes """

start = nodes[0]  # first node is the start depot
finish = nodes[-1]  # last node is the finish depot

for node in nodes[1:-1]:  # excludes both depots
    snEdge = Edge(start, node)  # creates the (start, node) edge (arc)
    nfEdge = Edge(node, finish)  # creates the (node, finish) edge (arc)
    # compute the Euclidean distance as cost
    snEdge.cost = math.sqrt((node.x - start.x) ** 2 + (node.y - start.y) ** 2)
    nfEdge.cost = math.sqrt((node.x - finish.x) ** 2 + (node.y - finish.y) ** 2)

    # save in node a reference to the (depot, node) edge (arc)
    node.dnEdge = snEdge
    node.ndEdge = nfEdge

efficiencyList = []

for i in range(1, len(nodes) - 2):  # excludes the start and finish depots
    iNode = nodes[i]
    for j in range(i + 1, len(nodes) - 1):
        jNode = nodes[j]
        ijEdge = Edge(iNode, jNode)  # creates the (i, j) edge
        jiEdge = Edge(jNode, iNode)  # creates the inverse edge (arc)
        ijEdge.invEdge = jiEdge
        jiEdge.invEdge = ijEdge  # sets the inverse edge (arc)
        # compute the Euclidean distance as cost
        ijEdge.cost = math.sqrt((jNode.x - iNode.x) ** 2 + (jNode.y - iNode.y) ** 2)
        jiEdge.cost = ijEdge.cost  # assume symmetric costs
        # compute efficiency as proposed by Panadero et al. (2020)
        ijSavings = iNode.ndEdge.cost + jNode.dnEdge.cost - ijEdge.cost
        edgeReward = iNode.demand + jNode.demand
        ijEdge.savings = ijSavings
        ijEdge.efficiency = alpha * ijSavings + (1 - alpha) * edgeReward
        jiSavings = jNode.ndEdge.cost + iNode.dnEdge.cost - jiEdge.cost
        jiEdge.savings = jiSavings
        jiEdge.efficiency = alpha * jiSavings + (1 - alpha) * edgeReward
        # save both edges in the efficiency list
        efficiencyList.append(ijEdge)
        efficiencyList.append(jiEdge)

# sort the list of edges from higher to lower efficiency
efficiencyList.sort(key=operator.attrgetter("efficiency"), reverse=True)
""" Construct the dummy solution """

sol = Solution()
for node in nodes[1:-1]:  # excludes the start and finish depots
    snEdge = node.dnEdge  # get the (start, node) edge
    nfEdge = node.ndEdge  # get the (node, finish) edge
    snfRoute = Route()  # construct the route (start, node, finish)
    snfRoute.edges.append(snEdge)
    snfRoute.demand += node.demand
    snfRoute.cost += snEdge.cost
    snfRoute.edges.append(nfEdge)
    snfRoute.cost += nfEdge.cost
    node.inRoute = snfRoute  # save in node a reference to its current route
    node.isLinkedToStart = True  # this node is currently linked to start depot
    node.isLinkedToFinish = True  # this node is currently linked to finish depot
    sol.routes.append(snfRoute)  # add this route to the solution
    sol.cost += snfRoute.cost
    sol.demand += snfRoute.demand  # total reward in route

""" Perform the edge-selection & routing-merging iterative process """


def checkMergingConditions(iNode, jNode, iRoute, jRoute, ijEdge):
    # condition 1: iRoute and jRoute are not the same route object
    if iRoute == jRoute:
        return False
    # condition 2: jNode has to be linked to start and i node to finish
    if iNode.isLinkedToFinish == False or jNode.isLinkedToStart == False:
        return False
    # condition 3: cost after merging does not exceed maxTime (or maxCost)
    if routeMaxCost < iRoute.cost + jRoute.cost - ijEdge.savings:
        return False
    # else, merging is feasible
    return True


while len(efficiencyList) > 0:  # List is not empty
    index = 0  # the index of the list to select the next edge from the list
    ijEdge = efficiencyList.pop(index)  # select the next edge from the list
    # determine the nodes that define the edge
    iNode = ijEdge.origin
    jNode = ijEdge.end
    # determine the routes associated with each node
    iRoute = iNode.inRoute
    jRoute = jNode.inRoute
    # check if merge is possible
    isMergeFeasible = checkMergingConditions(iNode, jNode, iRoute, jRoute, ijEdge)
    # if all necessary conditions are satisfied, merge and delete edge (j, i)
    if isMergeFeasible == True:
        # If still in list, delete edge (j, i) since it will not be used
        jiEdge = ijEdge.invEdge
        if jiEdge in efficiencyList:
            efficiencyList.remove(jiEdge)
        # Remove iEdge from iRoute and update iRoute cost
        iEdge = iRoute.edges[-1]  # iEdge is (i, finish)
        iRoute.edges.remove(iEdge)
        iRoute.cost -= iEdge.cost
        # Node i will not be linked to finish depot anymore
        iNode.isLinkedToFinish = False
        # Remove jEdge from jRoute and update jRoute cost
        jEdge = jRoute.edges[0]  # jEdge is (j, start)
        jRoute.edges.remove(jEdge)
        jRoute.cost -= jEdge.cost
        # Node j will not be linked to start depot anymore
        jNode.isLinkedToStart = False
        # add Edge to iRoute (end)
        iRoute.edges.append(ijEdge)
        iRoute.cost += ijEdge.cost
        iRoute.demand += jNode.demand
        jNode.inRoute = iRoute
        # add jRoute to new route
        for edge in jRoute.edges:
            iRoute.edges.append(edge)
            iRoute.cost += edge.cost
            iRoute.demand += edge.end.demand
            edge.end.inRoute = iRoute
        # delete jRoute from emerging solution
        sol.cost -= ijEdge.savings
        sol.routes.remove(jRoute)
# sort the list of routes in sol by demand (reward) and delete extra routes
sol.routes.sort(key=operator.attrgetter("demand"), reverse=True)

for route in sol.routes[fleetSize:]:
    sol.demand -= route.demand  # update reward
    sol.cost -= route.cost  # update cost
    sol.routes.remove(route)  # delete extra route

endTime = time.time()

""" Print the PJS Solution """
print("Instance: ", instanceName)
print("Reward obtained with PJS heuristic sol: ", "{:.2f}".format(sol.demand, 2))
print("Computational time: ", "{:.2f}".format(endTime - startTime, 2), "sec.")
for route in sol.routes:
    s = str(0)
    for edge in route.edges:
        s += " - " + str(edge.end.ID)
    print(
        "Route: "
        + s
        + " || Reward = "
        + "{:.2f}".format(route.demand, 2)
        + " || Cost / Time = "
        + "{:.2f}".format(route.cost, 2)
    )

G = nx.Graph()
G.add_node(start.ID, coord=(start.x, start.y))
for route in sol.routes:
    for edge in route.edges:
        G.add_edge(edge.origin.ID, edge.end.ID)
        G.add_node(edge.end.ID, coord=(edge.end.x, edge.end.y))
coord = nx.get_node_attributes(G, "coord")
nx.draw_networkx(G, coord, node_color="pink")
# Save the plot to a file
plt.axis("off")  # Turn off the axis
plt.savefig("network_graph.png", bbox_inches="tight")
