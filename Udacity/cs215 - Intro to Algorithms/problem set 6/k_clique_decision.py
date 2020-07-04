# Decision problems are often just as hard as actually returning an answer.
# Show how a k-clique can be found using a solution to the k-clique decision
# problem.  Write a Python function that takes a graph G and a number k
# as input, and returns a list of k nodes from G that are all connected
# in the graph.  Your function should make use of "k_clique_decision(G, k)",
# which takes a graph G and a number k and answers whether G contains a k-clique.
# We will also provide the standard routines for adding and removing edges from a graph.


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


def k_clique(G, k):
    # check if there is a clique first
    if not k_clique_decision(G, k):
        return False

    # if k is 1, then as long as graph has a node, then the graph has a clique consisting of any 1 node
    if k == 1 and len(G):
        return [G.keys()[0]]

    # if there is then we can remove all edges, one by one, and check if a clique is still present
    # if yes, leave removed, else, we must repair the link. After all edges are checked, only the clique remains.
    for node in G.keys():
        for edge in G[node].keys():
            break_link(G, node, edge)
            # edge is part of the clique, must re-make it
            if not k_clique_decision(G, k):
                make_link(G, node, edge)
    # return a list of the nodes which still have edges
    return [node for node in G if len(G[node])]
