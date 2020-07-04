# Take a weighted graph representing a social network where the weight
# between two nodes is the "love" between them.  In this "feel the
# love of a path" problem, we want to find the best path from node `i`
# and node `j` where the score for a path is the maximum love of an
# edge on this path. If there is no path from `i` to `j` return
# `None`.  The returned path doesn't need to be simple, ie it can
# contain cycles or repeated vertices.
#
# Devise and implement an algorithm for this problem.


def feel_the_love(G, i, j):
    # return a path (a list of nodes) between `i` and `j`,
    # with `i` as the first node and `j` as the last node,
    # or None if no path exists

    # Do a breadth first search from j to find all nodes reachable from it, saving the max edge weight seen so far along
    # with that node. As we find a node's neighbours, save in a dict with value being that node, forming a linked list
    # from i to any node (shortest path in terms of num of nodes). If a neighbour already has a link, it is not added to
    # the queue but its link must be overridden if this edge is the max weight, otherwise the max edge may not appear in
    # the spanning tree of links. NOTE: max node will always point to the other node in max edge, so if max node is in a
    # path, it will contain the max edge. If a path to j from i exists, j will be in the link dict, and then we must
    # check the max weight edge in the tree. If it is in the path, return the path using the links from j. Else, find
    # where the max edge path intersects the path to j, and splice the two paths together to create our final path.
    # TODO: could get shortest path (num nodes) by doing 2nd BFS from the max edge node, which terminates once at j
    # TODO: or BFS to find all edges, then Dijkstra's to max edge, then 2nd call from here to j (for weighted path)

    from collections import deque
    # Breadth first search to visit all possible nodes from start node, i
    queue = deque([i])
    links = {i: "start"}
    max_weight = (None, 0)
    while queue:
        current_node = queue.popleft()
        # mark current node so we dont return to it, have checked all its edges
        for neighbour in G[current_node]:
            # If we have linked this neighbour, it will be on the queue already, or has been checked already
            if neighbour not in links:
                queue.append(neighbour)
                links[neighbour] = current_node
            # check weight, if find new high, save the node and weight, and override neighbours link
            if G[current_node][neighbour] > max_weight[1]:
                max_weight = (neighbour, G[current_node][neighbour])
                # override the neighbour's link, as we must cross this edge, in case it is the max weight
                # this will cut an already checked edge - which will be a lower weight so it doesnt matter
                links[neighbour] = current_node

    # if we made it to node, j, backtrack to determine that path
    if j in links:
        path_to_j = [j]
        node = j
        # backtrack from j to get path from i, creating a list of nodes in the path
        while node != i:
            node = links[node]
            path_to_j.insert(0, node)
        max_node = max_weight[0]
        # check if max node/edge is in path to j, if not - insert path to max into path to j
        if max_node not in path_to_j:
            # backtrack from the max edge node towards i until we find a node that is in the path to j, this is the
            # backtrack path from max node to path to j. Must then add the forward path, excluding the first and last
            # nodes, as they appear in either list already. Then we join all the paths to get the final path
            path_to_max = [max_node]
            node = max_node
            while node not in path_to_j:
                node = links[node]
                path_to_max.append(node)
            # determine index of the node of intersection, then insert the forward path to max at this point
            insert_index = path_to_j.index(path_to_max[-1]) + 1
            for num, node in enumerate(reversed(path_to_max[1:-1])):
                path_to_j.insert(insert_index + num, node)
            # now add the reverse path, updating the insert index, taking into account the forward paths length
            insert_index += len(path_to_max) - 2
            for num, node in enumerate(path_to_max):
                path_to_j.insert(insert_index + num, node)
        return path_to_j
    else:  # didnt make it to j, return None
        return None


# given
def score_of_path(G, path):
    max_love = -float('inf')
    for n1, n2 in zip(path[:-1], path[1:]):
        love = G[n1][n2]
        if love > max_love:
            max_love = love
    return max_love


def test_love():
    G = {'a': {'c': 1},
         'b': {'c': 1},
         'c': {'a': 1, 'b': 1, 'e': 1, 'd': 1},
         'e': {'c': 1, 'd': 2},
         'd': {'e': 2, 'c': 1},
         'f': {}}
    love_graph = {"i": {"k": 1}, "k": {"m": 1, "l": 1, "i": 1}, "l": {"j": 1, "p": 1, "k": 1}, "j": {"l": 1},
                  "p": {"l": 1, "o": 1, "q": 1}, "o": {"r": 1, "n": 2, "p": 1}, "n": {"o": 2, "m": 1},
                  "m": {"k": 1, "n": 1}, "r": {"q": 1, "o": 1}, "q": {"p": 1, "o": 1}}
    lg2 = {"i": {"k": 1, "s": 1}, "k": {"l": 1, "i": 1}, "l": {"j": 1, "p": 1, "k": 1}, "j": {"l": 1, "s": 1},
           "p": {"l": 1, "o": 1, "q": 1}, "o": {"r": 1, "n": 2, "p": 1}, "n": {"o": 2, "m": 1},
           "m": {"n": 1}, "r": {"q": 3, "o": 1}, "q": {"p": 1, "o": 1, "r": 3}, "s": {"j": 1, "i": 1, "t": 1, "u": 1},
           "t": {"s": 1}, "u": {"s": 1}}
    path = feel_the_love(love_graph, "i", "j")
    assert score_of_path(love_graph, path) == 2

    path = feel_the_love(lg2, "i", "j")
    assert score_of_path(lg2, path) == 3

    path = feel_the_love(G, 'a', 'b')
    assert score_of_path(G, path) == 2

    path = feel_the_love(G, 'a', 'f')
    assert path is None
