#
# Modify long_and_simple_path
# to build and return the path
#
from copy import deepcopy


# Given this function
def make_link(G, node1, node2):
    if node1 not in G:
        G[node1] = {}
    (G[node1])[node2] = 1
    if node2 not in G:
        G[node2] = {}
    (G[node2])[node1] = 1
    return G


# given
def break_link(G, node1, node2):
    # removes edge between two nodes in a graph if it exists and returns the graph
    if node1 not in G:
        print("error: breaking link in a non-existent node")
        return
    if node2 not in G:
        print("error: breaking link in a non-existent node")
        return
    if node2 not in G[node1]:
        print("error: breaking non-existent link")
        return
    if node1 not in G[node2]:
        print("error: breaking non-existent link")
        return
    del G[node1][node2]
    del G[node2][node1]
    return G


# given
def all_perms(seq):
    # creates a list of all permutations of the given sequence of nodes
    if len(seq) == 0:
        return [[]]
    if len(seq) == 1:
        return [seq, []]
    most = all_perms(seq[1:])
    first = seq[0]
    rest = []
    for perm in most:
        for i in range(len(perm) + 1):
            rest.append(perm[0:i] + [first] + perm[i:])
    return most + rest


# given
def check_path(G, path):
    # check if each node in the path is connected to the next node in the path, if any are not
    # then it is not a valid path in the graph, as a connection is missing
    for i in range(len(path) - 1):
        if path[i + 1] not in G[path[i]]:
            return False
    return True


# given - commented it
def long_and_simple_decision(G, u, v, l):
    if l == 0:
        return False
    # creates all possible permutations of node paths in the graph, even if they arent actual paths in the graph
    # example: Tree rooted at 1, with children 2,3: [1,2] (valid), [1,2,3] (invalid - no edge bweteen 2 and 3)
    perms = all_perms(list(G.keys()))
    # check the permutations to see if there are any paths in the graph of length l (all permutations are simple paths)
    for perm in perms:
        # check permutation is correct length, is an actual path in the graph, starts with u, and ends with v
        if (len(perm) >= l and check_path(G, perm) and perm[0] == u
                and perm[len(perm) - 1] == v):
            return True
    return False


def testo():
    # initialise graph
    flights = [(1, 2), (1, 3), (2, 3), (2, 6), (2, 4), (2, 5), (3, 6), (4, 5)]
    G = {}
    for (x, y) in flights:
        make_link(G, x, y)

    # run simple tests
    print("graph:", G)
    test1 = long_and_simple_path(G, 1, 4, 9)
    print("test1:", test1)
    test2 = long_and_simple_path(G, 1, 4, 6)
    print("test2:", test2)
    assert test1 is False
    assert test2 == [1, 3, 6, 2, 5, 4]


def long_and_simple_path(G, u, v, l):
    """
    G: Graph
    u: starting node
    v: ending node
    l: minimum length of path
    """

    if not long_and_simple_decision(G, u, v, l):
        return False

    # break every edge and test if the path still holds. If yes, then leave out this edge, if not, then
    # restore the edge as it is one of the paths edges. After the algorithm finishes, the graph contains
    # only edges in the path we want, so we can return that path by following the edges starting at node u

    # used a copy of the graph, else get a runtime error - iterating over graph as we change it is apparently a problem
    # TODO: can use list(G[node].keys()) to get a list of edges instead of iterating over the dict for that node
    reduced_graph = deepcopy(G)
    for node in G:
        for edge in G[node]:
            # break link (both ways)
            broken = break_link(reduced_graph, node, edge)
            # make sure break link has actually broken link, else we create duplicate links
            if broken and not long_and_simple_decision(reduced_graph, u, v, l):
                make_link(reduced_graph, node, edge)
    print("reduced graph:", reduced_graph)
    # run through the reduced graph, starting at node u, appending the next node in the path to a list
    # until we reach node v, at which point the list contains the full path, in order
    path = list()
    path.append(u)
    prev_node = u
    node = u
    while node != v:
        # should be max 2 edges for each node, with u,v having only 1 edge each
        # next node in path is the neighbour which is not the previous node
        for neighbour in reduced_graph[node]:
            if neighbour != prev_node:
                path.append(neighbour)
                prev_node = node
                print("prev_node:", prev_node)
                node = neighbour
                print("node:", node)
                break
    return path
