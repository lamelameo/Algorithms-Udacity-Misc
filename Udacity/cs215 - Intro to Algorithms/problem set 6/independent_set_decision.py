# In the lecture, we described how a solution to k_clique_decision(G, k)
# can be used to solve independent_set_decision(H,s).
# Write a Python function that carries out this transformation.


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
# Returns a list of all the subsets of a list of size k
def k_subsets(lst, k):
    if len(lst) < k:
        return []
    if len(lst) == k:
        return [lst]
    if k == 1:
        return [[i] for i in lst]
    return k_subsets(lst[1:], k) + map(lambda x: x + [lst[0]], k_subsets(lst[1:], k - 1))


# given
# Checks if the given list of nodes forms a clique in the given graph.
def is_clique(G, nodes):
    for pair in k_subsets(nodes, 2):
        if pair[1] not in G[pair[0]]:
            return False
    return True


# given
# Determines if there is clique of size k or greater in the given graph.
def k_clique_decision(G, k):
    nodes = G.keys()
    for i in range(k, len(nodes) + 1):
        for subset in k_subsets(nodes, i):
            if is_clique(G, subset):
                return True
    return False


# This function should use the k_clique_decision function
# to solve the independent set decision problem
def independent_set_decision(H, s):
    # intialise the inverse graph
    G = {}
    # if s is 1, then there is always an independent set, given the graph has at least 1 node
    if s == 1 and len(G) > 0:
        return True
    # check all nodes and see if they have an edge with other nodes,
    # if not make an edge to create the inverse graph
    for node in H:
        for other_nodes in H:
            if other_nodes != node and other_nodes not in H[node]:
                make_link(G, node, other_nodes)
    # check if inverse graph, G, has a clique of size, s, then that set of nodes is an independant set in graph, H
    print(H)
    return k_clique_decision(G, s)
