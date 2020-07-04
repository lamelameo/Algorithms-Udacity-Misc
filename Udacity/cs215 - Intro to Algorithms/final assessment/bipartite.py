#
# Write a function, `bipartite` that
# takes as input a graph, `G` and tries
# to divide G into two sets where
# there are no edges between elements of the
# the same set - only between elements in
# different sets.
# If two sets exists, return one of them
# or `None` otherwise
# Assume G is connected
#

import timeit


# Given this function
def make_link(G, node1, node2):
    if node1 not in G:
        G[node1] = {}
    (G[node1])[node2] = 1
    if node2 not in G:
        G[node2] = {}
    (G[node2])[node1] = 1
    return G


def bipartite(G):
    # TODO: Essentially a breadth first search of the graph...without using a queue
    # Start at arbitrary node in graph. Append it to set1 and mark it as so, using a dict. Check edges for unmarked
    # nodes. Any found must be in the opposite set for it to be valid. Now we must do the same for the group of nodes we
    # placed in set 2. Check edges for nodes, following same procedure for unmarked nodes. For marked neighbours, if
    # they are in same set as the node, that means the graph is invalid, as it was placed there to move it away from
    # another node in opposite set and cannot move it there. Mark ths graph as invalid, but continue on to produce a
    # graph anyway (could be trimmed of bad edges if we mark them as bad to get valid graph).
    # This procedure starts after intialising a start node, then continues checking each new group of nodes, alternating
    # between sets till all nodes have been placed in either set. If it is possible to create a bipartite graph,
    # this algorithm will create a valid form, else it will mark the attempt as invalid. This is because we have checked
    # every edge in the graph (twice) to make sure the nodes are in opposite sets, and done so in a sequential manner,
    # meaning each nodes placement relies on the chain of nodes before it.

    # Initialise start node as first node in dict keys list (doesnt matter where we start)
    reference_node = list(G)[0]
    sets = [[reference_node], []]
    set_index = 1
    slice_index = [0, 0]
    # Add marker for each node in the graph so we know which set it has been put it, if any. 0=set1, 1=set2
    markers = {reference_node: 0}
    invalid_graph = False
    break_loop = False
    # Loop till we have placed all nodes into either set
    while True:
        # All nodes placed into a set, just have to check edges of the last group of nodes before we break loop,
        # in case they are connected to other nodes in same set. If we placed at end of loop, would terminate
        # early, as we mark before checking edges, leading to potentially returning an invalid graph.
        if len(markers) == len(G):
            break_loop = True
        # alternate between set1 and set2: 1-0 = 1, 1-1 = 0
        set_index = 1 - set_index
        current_set = sets[set_index]
        other_set = sets[1 - set_index]
        # Check edges of each node in the current, and then place the connected nodes in the other set.
        for node in current_set[slice_index[set_index]:]:  # only search new nodes added to set
            # increment slice index for next loop for each new node found
            slice_index[set_index] += 1
            for neighbour in G[node]:
                # If neighbour is marked, check if its in same group, if so, we cannot create a bipartite graph.
                if neighbour in markers:
                    if markers[neighbour] != 1 - set_index:
                        # print("edge between nodes:", node, ",", neighbour, "in set",
                        #       set_index + 1, "- graph cannot be transformed into bipartite form")
                        invalid_graph = True
                else:  # New node found, add to other set, marking it
                    markers[neighbour] = 1 - set_index
                    other_set.append(neighbour)

        # All nodes are placed into either set and edges have been checked, graph is transformed into valid bipartite
        # graph, or marked as invalid. Must now break while loop.
        if break_loop:
            break

    if invalid_graph:
        return None
    else:
        # print(sets)
        return set(sets[0])


def test_bipartite():
    edges = [(1, 2), (2, 3), (1, 4), (2, 5),
             (3, 8), (5, 6)]
    bi_edges = [(1, 5), (2, 5), (2, 6), (7, 2), (3, 7), (3, 5), (7, 4)]
    bad_edges = [(7, 2), (2, 5), (1, 5), (3, 4), (2, 6), (7, 4), (3, 7), (3, 5)]
    G = {}
    for n1, n2 in edges:
        make_link(G, n1, n2)
    g1 = bipartite(G)
    assert (g1 == set([1, 3, 5]) or
            g1 == set([2, 4, 6, 8]))
    edges = [(1, 2), (1, 3), (2, 3)]
    G = {}
    for n1, n2 in edges:
        make_link(G, n1, n2)
    g1 = bipartite(G)
    assert g1 == None


setup2 = "from __main__ import bipartite\nfrom __main__ import make_link"
code2 = "bi_edges = [(1, 5), (2, 5), (2, 6), (7, 2), (3, 7), (3, 5), (7, 4)]\n" \
        "bad_edges = [(7, 2), (2, 5), (1, 5), (3, 4), (2, 6), (7, 4), (3, 7), (3, 5)]\n" \
        "b1 = {}\nfor n1, n2 in bi_edges:\n    make_link(b1, n1, n2)\n" \
        "b2 = {}\nfor n1, n2 in bad_edges:\n    make_link(b2, n1, n2)\n" \
        "bipartite(b1); bipartite(b2)"
time = timeit.timeit(setup=setup2, stmt=code2, number=10000)
print(time)
