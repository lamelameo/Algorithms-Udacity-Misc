# Find Eulerian Tour
#
# Write a function that takes in a graph
# represented as a list of tuples
# and return a list of nodes that
# you would follow on an Eulerian Tour
#
# For example, if the input graph was
# [(1, 2), (2, 3), (3, 1)]
# A possible Eulerian tour would be [1, 2, 3, 1]

import timeit


def find_eulerian_tour(graph):
    # Finds an Eulerian tour by finding smaller loops in the graph and removing edges as we go. Then inserting loops
    # into one another to create the full tour as one big loop.

    # Initialise
    start_node = graph[0][0]
    current_node = start_node
    sub_loops = []
    sub_loop = [start_node]

    # Continue searching to find loops till we have searched all edges. At this point we have to try insert
    # each loop into each other till we have a single loop, which is the tour.
    while graph:
        # check edges till we find one that contains the current_node
        for index, edge in enumerate(graph):
            if current_node in edge:
                # next node index depends on current index: curr = 1|0, next = 1 - (1|0) = 0|1
                next_node_index = 1 - (edge.index(current_node))
                next_node = edge[next_node_index]
                # else we have a valid path, add the edge and nodes to the sub loop and update current node
                sub_loop.append(next_node)
                # remove edge from graph, so we dont revisit it, TODO: alternatively use a dict to mark edges
                del graph[index]
                current_node = next_node
                break
        # If all edges are checked and rejected, we have completed a sub loop, ie used all edges for starting node
        # and returned back to it. Must now check the graph for other loops.
        else:
            # Initialise starting node for next loop, to find any other loops
            current_node = graph[0][0]
            sub_loops.append(sub_loop)
            sub_loop = []

    # last sub loop will not have been appended yet, as we terminate the while loop before reaching the else clause
    sub_loops.append(sub_loop)

    # Must now find a node in each sub loop which is in another loop and then shift the sub loop to end at that
    # node, retaining the correct sequence of nodes. Then we can insert each node from this loop into the other,
    # after the index where that node appears in that loop.
    # TODO: some use of dict would help here... with a key for each sub loop, and a value for each node
    # TODO: what is the best way to insert to do least indexing?

    # Nested function to insert a loop into another. Removes the last loop in the list of loops we have found and for
    # each node in that loop, it tries to find a loop in the list which also contains that node. When successful, the
    # loop to be inserted is shifted to end on that node and is inserted into the found loop where the node was found.
    # Now the list of loops is 1 shorter, and we must call it till we have a single loop, which is the tour.
    # Used a function simply for ease of breaking multiple for-loops by returning once we have completed an insertion.
    def join_loops():
        current_loop = sub_loops.pop()
        # BRUTE FORCE through each node and each other loop to find the shifting/insertion points
        for node in current_loop:  # check all nodes of loop to be inserted
            for insert_loop in sub_loops:  # check other loops
                for insert_index, insert_node in enumerate(insert_loop):  # check for insertion point
                    if insert_node == node:
                        # TODO: if in first half of list, shift left, else shift right for lower amount of shifting
                        # shift end node to start till loop ends with the given node and then insert into other loop
                        while current_loop[-1] != node:  # shift
                            current_loop.insert(0, current_loop.pop())
                        for num, item in enumerate(current_loop):  # insert
                            insert_loop.insert(insert_index + 1 + num, item)
                        return  # return from function to potentially be called again

    # Call join function till we have our tour, which will be the only item in our loops list and then return it
    while len(sub_loops) > 1:
        join_loops()
    return sub_loops.pop()


# Recursive algorithm to find eulerian tour, first creates a dictionary form of graph for convenience then follows
# same loop finding concept as above
def find_eulerian_tour_recursive(graph):
    # Images of graphs g2, g3, g4, gmega and how this algorithm finds the tour at: https://puu.sh/DPi0P/60102f6c0c.png
    # TODO: This algorithm could be used to find Eulerian Paths too, simply by starting with an odd degree node
    # TODO: using this algorithm, starting with the node with the most edges may perform best, as it can find a
    # first loop consisting of multiple sub loops, reducing the amount of recursion/joining sub loops needed??
    # Possible to end up finding a smaller amount of edges than starting at another node, but its a good bet anyway?

    # Create a dictionary to hold keys corresponding to nodes, with values being a dict containing all the nodes it is
    # connected to. This will help in backtracking to find the right path to determine an Eulerian tour.
    graph_map = {}
    for edge in graph:
        node1, node2 = edge[0], edge[1]
        if node1 not in graph_map:
            graph_map[node1] = {node2: 0}
        else:
            graph_map[node1][node2] = 0
        if node2 not in graph_map:
            graph_map[node2] = {node1: 0}
        else:
            graph_map[node2][node1] = 0

    # Initialise with first edge in dict
    start_node = graph[0][0]

    def find_tour(starting_node):
        current_node = starting_node
        sub_loop = []
        insert_index = -1
        # Continue recursing to find loops till we have found/added all loops and thus searched all edges
        # We manually break while loops when we finish a sub loop.
        while True:
            insert_index += 1
            # traverse edges, deleting as we go until we run out of edges to check
            for neighbour in graph_map[current_node]:
                next_node = neighbour
                sub_loop.append(next_node)
                del graph_map[current_node][next_node]
                del graph_map[next_node][current_node]
                current_node = next_node
                break
            # If all edges are checked and rejected, we have completed a sub loop, ie used all edges for starting node
            # and returned back to it. Must now check other nodes in that sub loop to find new loops to connect to.
            # When we find one, recursively call this function, which will find all sub loops which we store in a list.
            # Once we have found all connected sub loops, insert each of them at their starting node in this loop.
            # Then return this loop, so the parent loop can repeat the same process, till we reach the initial loop.
            # which will then insert all found loops, and the final tour will be complete and the graph empty.
            else:
                # find nodes in sub loop to contain unvisited edges, using the queue we stored multi edge nodes in
                cumulative_index = 0
                recursive_loops = []
                for index, node in enumerate(sub_loop):
                    if len(graph_map[node]) != 0:
                        recursive_loop = find_tour(node)
                        recursive_loops.append((index + 1, recursive_loop))  # +1 to index - insert is after given index
                # Insert each sub loop, found by recursion, to the current loop keeping track of the changing index
                # Then break while loop to return this loop up to above depth.
                for index, loop in recursive_loops:
                    for num, node in enumerate(loop):
                        sub_loop.insert(index + num + cumulative_index, node)
                    cumulative_index += len(loop)
                break
        # Once we have searched all nodes in a sub loop and found no unvisited edges, we will exit the while loop at
        # that recursion depth. Must then return the sub loop so it can be appended to the parent loop
        return sub_loop
    # Shift tour to start at the 2nd node, only have to append that node to the end, instead of inserting first
    # node at the start of the tour which is more expensive
    tour = find_tour(start_node)
    tour.append(tour[0])
    return tour


# g2,g3,g4 are same graph with different edge orders to give different starting points and paths for different tests
g2 = [(0, 1), (1, 5), (1, 7), (4, 5),
      (4, 8), (1, 6), (3, 7), (5, 9),
      (2, 4), (0, 4), (2, 5), (3, 6), (8, 9)]
# starts on node with 2 edges, will find all 3 sub loops
g4 = [(0, 1), (1, 5), (1, 7), (4, 5),
      (4, 0), (1, 6), (3, 7), (5, 9),
      (2, 4), (8, 4), (2, 5), (3, 6), (8, 9)]
# starts on node with 4 edges, will find two loops (by linking two sub loops)
g5 = [(1, 5), (4, 5), (1, 7), (0, 1),
      (4, 0), (1, 6), (3, 7), (5, 9),
      (2, 4), (8, 4), (2, 5), (3, 6), (8, 9)]
# recursive loop finding idea graph, red, green, orange loops, purple tour
g3 = [(0, 1), (1, 5), (4, 5), (8, 9),
      (4, 0), (5, 9), (8, 4), (2, 5),
      (8, 7), (10, 8), (10, 12), (11, 10),
      (1, 3), (1, 6), (11, 3), (6, 11), (6, 13),
      (12, 6), (12, 13), (12, 11), (7, 10), (2, 4)]
# large interconnected grid with lots of recursion possible
gmega = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (0, 7),
         (1, 14), (14, 16), (15, 16), (1, 15), (1, 23), (23, 19), (19, 22), (1, 22),
         (16, 20), (17, 20), (16, 17),
         (9, 17), (6, 9), (6, 8), (3, 8), (3, 9), (9, 18), (17, 18),
         (15, 20), (15, 21), (20, 21),
         (19, 25), (25, 26), (24, 26), (19, 24),
         (4, 13), (11, 13), (5, 11), (5, 10), (10, 12), (4, 12)]

# print(find_eulerian_tour_recursive(gmega))
# print(find_eulerian_tour(gmega))

setup = "from __main__ import find_eulerian_tour_recursive"
code = "gmega = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (0,7)," \
       "(1,14), (14,16), (15,16), (1,15), (1,23), (23,19), (19,22), (1,22)," \
       "(16,20), (17,20), (16,17)," \
       "(9,17), (6,9), (6,8), (3,8), (3,9), (9,18), (17,18)," \
       "(15,20), (15,21), (20,21)," \
       "(19,25), (25,26), (24,26), (19,24)," \
       "(4,13), (11,13), (5,11), (5,10), (10,12), (4,12)]" \
       "\nfind_eulerian_tour_recursive(gmega)"
time = timeit.timeit(setup=setup, stmt=code, number=10000)
print(time)
