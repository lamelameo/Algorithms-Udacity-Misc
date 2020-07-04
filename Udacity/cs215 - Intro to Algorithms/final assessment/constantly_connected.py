#
# Design and implement an algorithm that can preprocess a
# graph and then answer the question "is x connected to y in the
# graph" for any x and y in constant time Theta(1).
#
# `process_graph` will be called only once on each graph.  If you want,
# you can store whatever information you need for `is_connected` in
# global variables
#


def process_graph1(G):
    # DFS or BFS to find all nodes connected to an arbitrary starting node. Marking each node when first seen and
    # add to a sub graph list, containing all the nodes we find in this search.
    # Repeat process starting with any remaining node until all nodes are marked and we have a list of sub graphs.
    # In our processed graph, add edges between every node in a sub graph, and we can then simply check if a node
    # is connected to another by checking the processed graph for that edge.
    global processed_graph
    processed_graph = {}
    sub_graphs = []  # make lists for all connected, then can iterate backwards and add new edges in G (where necessary)
    unseen = {str(x): True for x in G}
    # keep doing searches to find connected sub graphs until all nodes have been placed into a list
    while unseen:
        start_node, _ = unseen.popitem()
        queue = [start_node]
        sub_graph = [start_node]
        # keep searching till no more neighbours of nodes are found for this sub graph
        while queue:
            current_node = queue.pop()
            for neighbour in G[current_node]:
                if neighbour in unseen:
                    del unseen[neighbour]
                    queue.append(neighbour)
                    sub_graph.append(neighbour)
        # done with this sub graph
        sub_graphs.append(sub_graph)

    # can just add edges to graph, or make new graph
    for x in G:
        processed_graph[x] = {}

    # connect nodes in each sub graph
    for graph in sub_graphs:
        while graph:
            # pop last node and connect to all other nodes, continue till graph list is empty
            curr_node = graph.pop(-1)
            for node in graph:
                processed_graph[curr_node][node] = 1
                processed_graph[node][curr_node] = 1


# faster than above attempt...
def process_graph(G):
    # DFS or BFS to find all nodes connected to an arbitrary starting node. Unmarking each node when first seen and
    # adding a sub graph identifier, which will be the same for all the nodes we find in this search.
    # Repeat process starting with any remaining node until all nodes are unmarked and we have identified all nodes
    # We can simply check if a node is connected to another by checking if they have the same sub graph identifier.
    sub_graph = 0
    global processed_graph
    processed_graph = {}
    unseen = {str(x): True for x in G}
    # keep doing searches to find connected sub graphs until all nodes have been placed into a sub graph
    while unseen:
        start_node, _ = unseen.popitem()
        stack = [start_node]
        # keep searching till no more neighbours of nodes are found for this sub graph
        while stack:
            current_node = stack.pop()
            # give all nodes in this sub graph same identifier
            processed_graph[current_node] = sub_graph
            for neighbour in G[current_node]:
                if neighbour in unseen:
                    del unseen[neighbour]
                    stack.append(neighbour)
        # update sub graph identifier for next loop/sub graph (if any)
        sub_graph += 1


#
# When being graded, `is_connected` will be called
# many times so this routine needs to be quick
#
def is_connected(i, j):
    # check for an edge between the two nodes in the processed graph
    # if i in processed_graph[j]:
    if processed_graph[i] == processed_graph[j]:
        return True
    else:
        return False


#######
# Testing
#
def test_process():
    G = {'a': {'b': 1},
         'b': {'a': 1},
         'c': {'d': 1},
         'd': {'c': 1},
         'e': {}}
    process_graph(G)

    assert is_connected('a', 'b') is True
    assert is_connected('a', 'c') is False

    G = {'a': {'b': 1, 'c': 1},
         'b': {'a': 1},
         'c': {'d': 1, 'a': 1},
         'd': {'c': 1},
         'e': {}}
    process_graph(G)
    assert is_connected('a', 'b') is True
    assert is_connected('a', 'c') is True
    assert is_connected('a', 'e') is False


# test_process()
