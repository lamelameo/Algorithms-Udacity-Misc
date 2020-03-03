""" udacity tutorials and stuff """
import timeit


# Eulerian Tour Ver 1
#
# Write a function, `create_tour` that takes as
# input a list of nodes
# and outputs a list of tuples representing
# edges between nodes that have an Eulerian tour.
#

def create_tour(nodes):
    # If there are less than 3 nodes, then there is no Eulerian tour possible - but we are ignoring this...
    # An Eulerian tour can be created by linking each node to the next in the graph, till we reach the end, and then
    # link the last node back to the first node, creating a chain visiting all nodes and ending at the start
    tour = []
    for index, node in enumerate(nodes):
        if index + 1 == len(nodes):
            tour.append((node, nodes[0]))
        else:
            tour.append((node, nodes[index+1]))
    print(tour)
    return tour


# given
def get_degree(tour):
    degree = {}
    for x, y in tour:
        degree[x] = degree.get(x, 0) + 1
        degree[y] = degree.get(y, 0) + 1
    return degree


# given
def check_edge(t, b, nodes):
    """
    t: tuple representing an edge
    b: origin node
    nodes: set of nodes already visited

    if we can get to a new node from `b` following `t`
    then return that node, else return None
    """
    if t[0] == b:
        if t[1] not in nodes:
            return t[1]
    elif t[1] == b:
        if t[0] not in nodes:
            return t[0]
    return None


# given
def connected_nodes(tour):
    """return the set of nodes reachable from
    the first node in `tour`"""
    a = tour[0][0]
    nodes = set([a])
    explore = set([a])
    while len(explore) > 0:
        # see what other nodes we can reach
        b = explore.pop()
        for t in tour:
            node = check_edge(t, b, nodes)
            if node is None:
                continue
            nodes.add(node)
            explore.add(node)
    return nodes


# given
def is_eulerian_tour(nodes, tour):
    # all nodes must be even degree
    # and every node must be in graph
    degree = get_degree(tour)
    for node in nodes:
        try:
            d = degree[node]
            if d % 2 == 1:
                print("Node %s has odd degree" % node)
                return False
        except KeyError:
            print("Node %s was not in your tour" % node)
            return False
    connected = connected_nodes(tour)
    if len(connected) == len(nodes):
        return True
    else:
        print("Your graph wasn't connected")
        return False


def test():
    nodes = [20, 21, 22, 23, 24, 25]
    tour = create_tour(nodes)
    return is_eulerian_tour(nodes, tour)


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
# time = timeit.timeit(setup=setup, stmt=code, number=10000)
# print(time)


# Given this function
def make_link(G, node1, node2):
    if node1 not in G:
        G[node1] = {}
    (G[node1])[node2] = 1
    if node2 not in G:
        G[node2] = {}
    (G[node2])[node1] = 1
    return G


def create_combo_lock(nodes):
    # nodes is a list of integers
    G = {}
    first_node = nodes[0]
    prev_node = None
    for node in nodes:
        # exclude first node, as it has no previous node
        if prev_node is not None:
            # make link between each node to form chain, and add link to first node for each, to make loops
            make_link(G, prev_node, node)
            make_link(G, node, first_node)
        # update prev node to current for next loop
        prev_node = node
    return G


def centrality(G, v):
    distance_from_start = {}
    open_list = [v]
    distance_from_start[v] = 0
    while len(open_list) > 0:  # while queue is not empty, check first in queue for neighbours
        current = open_list[0]
        del open_list[0]
        for neighbour in G[current].keys():
            if neighbour not in distance_from_start:
                distance_from_start[neighbour] = distance_from_start[current] + 1
                open_list.append(neighbour)
    return max(distance_from_start.values())


def mark_component(G, node, marked):
    marked[node] = True
    total_marked = 1
    open_list = [node]  # initialise stack to contain only the chosen node
    while open_list:  # while stack is not empty, check each node in the stack
        curr_node = open_list.pop()
        for neighbor in G[curr_node]:
            if neighbor not in marked:  # add unmarked neighbours to the stack, mark them and increment counter
                open_list.append(neighbor)
                marked[neighbor] = True
                total_marked += 1
    return total_marked


def create_rooted_spanning_tree(G, root):
    # initialize stuff
    S = {}
    node_stack = [root]
    S[root] = {}
    # TODO: can replace not in with dict.get() which returns None if no value for that key, is faster?
    # iterate through Graph nodes starting at the given root, going depth first through neighbours using a stack
    while node_stack:  # will end when stack is empty
        curr_node = node_stack.pop()
        # check all neighbours of current node and add them to the tree graph as connections to this node
        for neighbour in G[curr_node]:
            # add neighbours to stack and tree graph if it has not been seen already
            if neighbour not in S:
                node_stack.append(neighbour)
                S[neighbour] = {}
                # add green connection to this edge in tree graph
                (S[curr_node])[neighbour] = "green"
                (S[neighbour])[curr_node] = "green"
            # if neighbour is already part of the tree but connection has not been made, then we have a red edge,
            # as we are looping back around onto the current branch in the tree
            elif neighbour not in S[curr_node]:
                (S[neighbour])[curr_node] = "red"
                (S[curr_node])[neighbour] = "red"
    return S


def test_create_rooted_spanning_tree():
    G = {'a': {'c': 1, 'b': 1},
         'b': {'a': 1, 'd': 1},
         'c': {'a': 1, 'd': 1},
         'd': {'c': 1, 'b': 1, 'e': 1},
         'e': {'d': 1, 'g': 1, 'f': 1},
         'f': {'e': 1, 'g': 1},
         'g': {'e': 1, 'f': 1}
         }
    S = create_rooted_spanning_tree(G, "a")
    assert S == {'a': {'c': 'green', 'b': 'green'},
                 'b': {'a': 'green', 'd': 'red'},
                 'c': {'a': 'green', 'd': 'green'},
                 'd': {'c': 'green', 'b': 'red', 'e': 'green'},
                 'e': {'d': 'green', 'g': 'green', 'f': 'green'},
                 'f': {'e': 'green', 'g': 'red'},
                 'g': {'e': 'green', 'f': 'red'}
                 }


def post_order(S, root):
    # return mapping between nodes of S and the post-order value
    # of that node
    index = 1
    order = {}
    # the return type, (order, ordered_list), is not allowed for the quiz...try other hack,
    # by manually counting through dict values to find right order or something else
    ordered_list = []  # used this to maintain the order of the nodes, which is lost in a dict
    unordered_stack = [root]  # stack
    marked = {root: True}
    # TODO: could change to use a stack for each branch, and a queue to use breadth first search to
    # TODO: alternate between branches, till we find the shortest branch, then number the end and
    # TODO: travel back up the branch stack and order it, then move to queue to resume other branches

    # loop till stack is empty, will traverse the tree branch by branch (no specific order)
    while unordered_stack:
        curr_node = unordered_stack.pop()
        # check all neighbours of the current node in the branch to determine if it is an ending
        seen_all_child = True
        curr_node_in_stack = False
        for neighbour in S[curr_node]:
            # not at the end of this branch, have to add this node to the stack for future ordering
            if S[curr_node][neighbour] == "green" and neighbour not in marked:
                seen_all_child = False
                marked[neighbour] = True
                # add the current node to stack before any of the neighbours, but make sure only once
                if curr_node_in_stack is False:
                    unordered_stack.append(curr_node)
                    curr_node_in_stack = True
                unordered_stack.append(neighbour)
        # this node was the end of a branch, can give it the current index and increment that value
        if seen_all_child:
            order[curr_node] = index
            ordered_list.append(curr_node)
            index += 1
    return order


# This is just one possible solution
# There are other ways to create a
# spanning tree, and the grader will
# accept any valid result.
# feel free to edit the test to
# match the solution your program produces
def test_post_order():
    S = {'a': {'c': 'green', 'b': 'green'},
         'b': {'a': 'green', 'd': 'red'},
         'c': {'a': 'green', 'd': 'green'},
         'd': {'c': 'green', 'b': 'red', 'e': 'green'},
         'e': {'d': 'green', 'g': 'green', 'f': 'green'},
         'f': {'e': 'green', 'g': 'red'},
         'g': {'e': 'green', 'f': 'red'}
         }
    po = post_order(S, 'a')
    assert po == {'b': 1, 'f': 2, 'g': 3, 'e': 4, 'd': 5, 'c': 6, 'a': 7}


def number_of_descendants(S, root):
    D = {}
    stack = [root]
    # TODO: could use post order to check for "root" neighbours
    S[root]["root"] = None
    while stack:  # loop till stack is empty, which is when we have determined values for the whole tree
        node = stack.pop()
        # flags to use in for loop to evaluate the node after all neighbours are checked
        has_children = False
        children_numbered = False
        node_on_stack = False
        sum_children = 0
        for neighbour in S[node]:
            # check for green neighbours that arent root, and flag if find any, else we can number this node
            if S[node]["root"] != neighbour and S[node][neighbour] == "green":  # node has child
                has_children = True
                S[neighbour]["root"] = node  # mark childs root as node - will add unnecessary repetitions
                if neighbour in D:  # check if child is numbered and add to sum
                    sum_children += D[neighbour]
                    children_numbered = True
                else:  # there is an unnumbered branch we have to check before we can number this node
                    children_numbered = False
                    if not node_on_stack:  # add node (if not already) then child to stack
                        stack.append(node)
                        node_on_stack = True
                    stack.append(neighbour)
        # have checked for children, now can number, or move on to number children
        if has_children is False:  # if the node has no children, then it is an end, and its number is 1
            D[node] = 1
        else:  # has children, if they are all numbered, take the child sum and add one to get its number
            if children_numbered:
                D[node] = sum_children + 1

    # clean root entries in spanning tree
    for node in S:
        del S[node]["root"]

    # use prev algorithm to search through branches, but determine length as we go, and when we find an end can give
    # its length, and then decrement length for previous items on same branch, but this means we must track branching
    return D


def test_number_of_descendants():
    S =  {'a': {'c': 'green', 'b': 'green'},
          'b': {'a': 'green', 'd': 'red'},
          'c': {'a': 'green', 'd': 'green'},
          'd': {'c': 'green', 'b': 'red', 'e': 'green'},
          'e': {'d': 'green', 'g': 'green', 'f': 'green'},
          'f': {'e': 'green', 'g': 'red'},
          'g': {'e': 'green', 'f': 'red'}
          }
    nd = number_of_descendants(S, 'a')
    assert nd == {'a' :7, 'b' :1, 'c' :5, 'd' :4, 'e' :3, 'f' :1, 'g' :1}


def lowest_post_order(S, root, po):
    # return a mapping of the nodes in S
    # to the lowest post order value
    # below that node
    # (and you're allowed to follow 1 red edge)

    # use post order, to select nodes, 1-7, this will start at branch ends moving inwards, ensuring any non ends
    # will be connected to a node which has been given a number already on the same branch
    # if we cross a red to get to a neighbour, then we can only take its post_order not lowest post order
    # if we take a green neighbour's lowest post order, then it is a valid pathway for this node too and the
    # lowest post order down that branch regardless of sub branches

    lpo = {}
    # make list length of po dict, then insert each key into it in the order based on its po value
    ordered_po = ["" for _ in range(len(po))]
    for node in po:
        # replace the list item at the index of the node's post order value
        ordered_po[po[node] - 1] = node
    # loop through all nodes in post order using the ordered list
    for curr_node in ordered_po:
        node_lpo = po[curr_node]
        # check all neighbours
        for neighbour in S[curr_node]:
            # check green neighbours only if they are children of node
            if S[curr_node][neighbour] == "green" and po[neighbour] < po[curr_node]:
                # check child's lpo (will be set already), if it is lower than the current lpo, then update
                if lpo[neighbour] < node_lpo:
                    node_lpo = lpo[neighbour]
            # any red neighbour, check if its po is lower than current lpo, then update
            elif S[curr_node][neighbour] == "red" and po[neighbour] < node_lpo:
                node_lpo = po[neighbour]
        lpo[curr_node] = node_lpo  # add lowest post order of node to a dict to be returned
    return lpo


def test_lowest_post_order():
    S = {'a': {'c': 'green', 'b': 'green'},
         'b': {'a': 'green', 'd': 'red'},
         'c': {'a': 'green', 'd': 'green'},
         'd': {'c': 'green', 'b': 'red', 'e': 'green'},
         'e': {'d': 'green', 'g': 'green', 'f': 'green'},
         'f': {'e': 'green', 'g': 'red'},
         'g': {'e': 'green', 'f': 'red'}
         }
    po = post_order(S, 'a')
    l = lowest_post_order(S, 'a', po)
    print(l)
    assert l == {'a':1, 'b':1, 'c':1, 'd':1, 'e':2, 'f':2, 'g':2}


def highest_post_order(S, root, po):
    # return a mapping of the nodes in S
    # to the highest post order value
    # below that node
    # (and you're allowed to follow 1 red edge)

    hpo = {}

    # make list length of po dict, then insert each key into it in the order based on its po value
    ordered_po = ["" for _ in range(len(po))]
    for node in po:
        # replace the list item at the index of the node's post order value
        ordered_po[po[node] - 1] = node

    # use sorted list to run through nodes in post order
    for node in ordered_po:
        node_hpo = po[node]
        # check all neighbours
        for neighbour in S[node]:
            # check green neighbours only if they are children of node
            if S[node][neighbour] == "green" and po[neighbour] < po[node]:
                # check child's hpo (will be set already), if it is higher than the current hpo, then update
                if hpo[neighbour] > node_hpo:
                    node_hpo = hpo[neighbour]
            # any red neighbour, check if its po is higher than current hpo, then update
            elif S[node][neighbour] == "red" and po[neighbour] > node_hpo:
                node_hpo = po[neighbour]
        hpo[node] = node_hpo  # add highest post order of node to a dict to be returned
    return hpo


def test_highest_post_order():
    S = {'a': {'c': 'green', 'b': 'green'},
         'b': {'a': 'green', 'd': 'red'},
         'c': {'a': 'green', 'd': 'green'},
         'd': {'c': 'green', 'b': 'red', 'e': 'green'},
         'e': {'d': 'green', 'g': 'green', 'f': 'green'},
         'f': {'e': 'green', 'g': 'red'},
         'g': {'e': 'green', 'f': 'red'}
         }
    po = post_order(S, 'a')
    h = highest_post_order(S, 'a', po)
    assert h == {'a': 7, 'b': 5, 'c': 6, 'd': 5, 'e': 4, 'f': 3, 'g': 3}


def bridge_edges(G, root):
    # use the four functions above
    # and then determine which edges in G are bridge edges
    # return them as a list of tuples ie: [(n1, n2), (n4, n5)]

    bridges = []
    # functions to get spanning tree graph, post order, num descendants, lowest post order, highest post order
    st = create_rooted_spanning_tree(G, root)
    po = post_order(st, root)
    nd = number_of_descendants(st, root)
    lpo = lowest_post_order(st, root, po)
    hpo = highest_post_order(st, root, po)
    print("tree: ", st)
    print("post order: ", po)
    print("num dec: ", nd)
    print("low po: ", lpo)
    print("high po: ", hpo)
    # check all nodes in the tree
    for node in st:
        # if these relationships are both true, then the green edge that ends on this node is a bridge
        if (hpo[node] <= po[node]) and (lpo[node] > abs(nd[node] - po[node])):
            # get the node's neighbour on the other end of this bridge edge, using post order
            for neighbour in st[node]:
                # the neighbour will have higher post order, and be green
                if po[neighbour] > po[node] and st[node][neighbour] == "green":
                    bridges.append((neighbour, node))  # add the edge and break this inner loop
                    break
    return bridges


def test_bridge_edges():
    G = {'a': {'c': 1, 'b': 1},
         'b': {'a': 1, 'd': 1},
         'c': {'a': 1, 'd': 1},
         'd': {'c': 1, 'b': 1, 'e': 1},
         'e': {'d': 1, 'g': 1, 'f': 1},
         'f': {'e': 1, 'g': 1},
         'g': {'e': 1, 'f': 1}
         }
    bridges = bridge_edges(G, 'a')
    print(bridges)
    assert bridges == [('d', 'e')]


# Write partition to return a new array with
# all values less then `v` to the left
# and all values greater then `v` to the right
def partition(L, v):
    P = []
    higher_list = []
    num_v = 0
    for item in L:
        if v > item:
           P.append(item)  # adds item to lesser side of new list (unordered)
        elif v < item:
            higher_list.append(item)  # add to higher side list
        elif v == item:
            num_v += 1
    # what if the item is equal to v? I have just added them all to middle of two partitions...
    for _ in range(num_v):
        P.append(v)
    P += higher_list
    print(P)
    return P


# given this function - I have commented on it
import random as rand
def top_k(L, k):
    # a recursive algorithm to find the top/bottom k values (unsorted) in a list using randomly chosen pivots
    v = L[rand.randrange(len(L))]
    # TODO: does not handle doubled values, must look to change here and partition
    (left, middle, right) = partition(L, v)
    # if lower list = k, then we have found all k values, with v being 1 above that
    if len(left) == k:
        return left
    # if lower list + 1 = k then, the lower list + v constitutes all k values
    if len(left)+1 == k:
        return left+[v]
    # if lower list is larger than k, then we have too many values, and recursively call this function
    # still using value k to partition the lower list to make it smaller, till it reaches length k
    if len(left) > k:
        return top_k(left, k)
    # if lower list is smaller than k, then we have too few values, and recursively call the function
    # using a k value lowered by the length of the left list to start a new partition on the right list
    # till the length of the new lower list added to length of original lower list = k
    return left+[v]+top_k(right, k-len(left)-1)


def median(L):
    sums = []
    square_sums = []
    for item in L:
        sum1 = 0
        square = 0
        for item2 in L:
            sum1 += abs(item - item2)
            square += (item - item2)**2
        sums.append(sum1)
        square_sums.append(square)
    print("L", L)
    print("sorted", sorted(L))
    print("sums", sums)
    print("squares", square_sums)

# median([2,5,9,19,24,5,9,10,54,87,2,13,21,32,44,4,16,18,19,26,25,39,47,56,71])
# print(sorted([2,5,9,19,24,5,9,10,54,87,2,13,21,32,44,4,16,18,19,26,25,39,47,56,71]))
# median([2,2,3,4,2])


from math import ceil
def minimize_absolute(L):
    # this is same as finding the median in the sorted list, which we can get by partitioning to
    # find lowest k values, where k = list size/2, essentially splitting list in half,
    # then we search the list for the highest value which will be the median
    # this takes n operations for the top_k part and 1/2n operations to find max in the return list

    # if list is even, will take the lower of two middle values, if odd, ceil gives middle value
    length_list = len(L)
    half_vals = ceil(length_list / float(2))  # used float as udacity seems to not cast this to a float automatically
    lower_half = top_k(L, half_vals)
    median = lower_half[0]
    # get highest value in list, which is the median
    for val in lower_half:
        if val > median:
            median = val
    return median


#
# Given a list of numbers, L, find a number, x, that
# minimizes the sum of the square of the difference
# between each element in L and x: SUM_{i=0}^{n-1} (L[i] - x)^2
#
# Your code should run in Theta(n) time
#

def minimize_square(L):
    # f(x) = sum(to n) (L[i] - x)**2
    # g(x) = x**2, h(x) = L[i] - x
    # g'(x) = 2x, h'(x) = -1
    # f'(x) = sum (g'(h(x)) . h'(x))
    #       = sum 2(L[i] - x).(-1)
    #       = sum -2(L[i] - x)
    # minimum f(x) is when f'(x) = 0, or minimise f(x) as f'(x) approaches 0
    # therefore: 0 = -2 sum(L[i] - x)
    #            0 = sum(L[i) - nx  (sum is same as adding all elements and subtracting x, n times)
    #           nx = sum L[i]
    #            x = (sum L[i])/n
    # This is the average value of the list, so to minimise f(x), find x that is closest to the average
    # can do this in two loops, 2n operations, Theta(n) time
    average = 0
    for item in L:
        average += item
    average /= len(L)
    print("average", average)
    smallest_diff = abs(average - L[0])
    x = 0
    for item in L:
        diff = abs(average - item)
        if diff < smallest_diff:
            x = item
            smallest_diff = diff
    return x


#
# Given a list L of n numbers, find the mode
# (the number that appears the most times).
# Your algorithm should run in Theta(n).
# If there are ties - just pick one value to return
#

def mode(L):
    # use a dictionary to save the frequencies of each item using the item as the key
    # increasing frequency by 1 if the key already exists, or making a value of 1 if not
    # then check each key in the dictionary (which could be same length as list or a lot smaller)
    # keep track of the key for highest value seen so far, once checked all frequencies, then the key
    # we have is the item (or one of multiple) in the list with most frequency ie the mode

    frequencies = {}
    # TODO: could keep track of highest frequency as we go through the list, then we only need one pass to create the
    # dict of frequencies and return the max value...
    mode = 0
    for item in L:
        if item not in frequencies:
            frequencies[item] = 1
        else:
            frequencies[item] += 1
            if frequencies[item] > mode:
                mode = frequencies[item]
    return mode

    # TODO: checks all frequencies, to find the highest, slower but could list all highest or top x highest etc..
    # initialise the mode key arbitrarily to first item in list (it will be updated if it isnt the mode)
    # mode_key = L[0]
    # for value in frequencies:
    #     # determine if the current highest is lower than the value we are currently checking
    #     if frequencies[value] > frequencies[mode_key]:
    #         mode_key = value
    # return mode_key


def up_heapify1(L):
    # up heapify all values, iterate backwards through the list ie start with bottom node and move up to top of heap
    reverse_index = 0
    length = len(L)
    for _ in range(length):
        # check nodes children (min 0, max 2)
        # normal index(i) = length + reverse index
        # child1 index = 2i + 1, child2_ind = 2i + 2
        # reverse child1 index = child1 index - length
        #                      = 2i + 1 - length
        #                      = 2(length + reverse index) + 1 - length
        #                   c1 = length + 2*reverse index + 1, c2 = length + 2*reverse index + 2
        reverse_index -= 1
        child1_ind = length + 2 * reverse_index + 1
        child2_ind = length + 2 * reverse_index + 2
        print("rev index:", reverse_index)
        print(child1_ind)
        print(child2_ind)

        if child1_ind > -1:  # any node that has no children will have child index outside the range ie 0 or higher
            continue  # ignore these nodes and continue up the heap list

        # check if each child's value is less than the node, if so then swap the values
        # taking account of 1st change, if any, for the 2nd child
        node = L[reverse_index]
        child1 = L[child1_ind]
        child2 = L[child2_ind]
        if node > child1:  # must swap child and node
            print("node", node)
            print("child1", child1)
            L[reverse_index] = child1
            L[child1_ind] = node

        if child2_ind > -1:  # could have a left child but no right child, move to next
            continue
        if node > child2:  # check
            print("node", node)
            print("child2", child2)
            L[reverse_index] = child2
            L[child2_ind] = node
    return L


#
# write up_heapify, an algorithm that checks if
# node i and its parent satisfy the heap
# property, swapping and recursing if they don't
#
# L should be a heap when up_heapify is done
#
def up_heapify(L, i):
    # if parent of node i has a greater value than i, swap the values and recurse, else return the List as it is a heap
    # if node i reaches top of heap, the parent will be itself, and fail the comparison and return the valid heap
    # TODO: fails in udacity as it runs python 2, added an if statement to return L once top of heap is reached...
    if i == 0:
        return L
    node = L[i]
    print("parent", int(parent(i)))
    parent_index = int(parent(i))
    parent_node = L[parent_index]
    print(node, parent_node)
    if node < parent_node:
        print(node, parent_node)
        L[parent_index] = node
        L[i] = parent_node
        return up_heapify(L, parent_index)
    else:
        return L


def dijk_up_heapify(L, values, indexes, i):
    # takes a heap as a list of node keys, a dict of the corresponding values for the keys, a dict containing the
    # indexes of those nodes in the heap, and a reference node index and makes appropriate swaps for node i and parents
    # recursively till list is a heap, returning the list and dict of indexes which may have been altered.
    # The indexes must be updated as the nodes are moved in the heap, since we must know
    # their position to up heapify correctly if a new shorter path is found and its value is updated
    if i == 0:
        return L, indexes
    print("i", i)
    print("values", values)
    print("indexes", indexes)
    node = L[i]
    parent_index = int(parent(i))
    print("parent index: ", parent_index)
    parent_node = L[parent_index]
    if values[node] < values[parent_node]:
        # swap keys in heap
        L[parent_index] = node
        L[i] = parent_node
        # swap heap index in dict
        indexes[node] = parent_index
        indexes[parent_node] = i
        return dijk_up_heapify(L, values, indexes, parent_index)
    else:
        return L, indexes


# given each of these one line functions
def parent(i):
    return (i - 1) / 2


def left_child(i):
    return 2 * i + 1


def right_child(i):
    return 2 * i + 2


def is_leaf(L, i):
    return (left_child(i) >= len(L)) and (right_child(i) >= len(L))


def one_child(L, i):
    return (left_child(i) < len(L)) and (right_child(i) >= len(L))


# given this function
def make_link_weighted(G, node1, node2, w):
    if node1 not in G:
        G[node1] = {}
    if node2 not in G[node1]:
        (G[node1])[node2] = 0
    (G[node1])[node2] += w
    if node2 not in G:
        G[node2] = {}
    if node1 not in G[node2]:
        (G[node2])[node1] = 0
    (G[node2])[node1] += w
    return G


def down_heapify(L, values, indexes, i):
    # Is used to create a heap given a List which is a heap except for a node i and its direct children.
    # Recursively check if node i is larger than its children and swap with the smaller of either till
    # the node is in its correct position and the list is now a heap. Returns the updated heap and indexes.
    # Takes a list of node keys, a dict of corresponding values, a dict of node indexes in the heap, and index of
    # the node to down heapify from. Must update the node indexes as they are changed.

    # if i is a leaf, it is a heap
    if is_leaf(L, i):
        return L, indexes
    # if i has one child check heap property, and swap values if necessary
    if one_child(L, i):
        if values[L[i]] > values[L[left_child(i)]]:
            # swap values
            (L[i], L[left_child(i)]) = (L[left_child(i)], L[i])
            # swap indexes
            (indexes[L[i]], indexes[L[left_child(i)]]) = (indexes[L[left_child(i)]], indexes[L[i]])
        return L, indexes
    # if i has two direct children check if the smaller is higher than i, if it is then return, if not
    # then we have to swap the smaller of two values
    if min(values[L[left_child(i)]], values[L[right_child(i)]]) >= values[L[i]]:
        return L, indexes
    # check for smaller child and swap with i
    if values[L[left_child(i)]] < values[L[right_child(i)]]:
        # swap values and indexes
        (L[i], L[left_child(i)]) = (L[left_child(i)], L[i])
        (indexes[L[i]], indexes[L[left_child(i)]]) = (indexes[L[left_child(i)]], indexes[L[i]])
        down_heapify(L, values, indexes, left_child(i))
        return L, indexes
    # right child is smaller
    (L[i], L[right_child(i)]) = (L[right_child(i)], L[i])
    (indexes[L[i]], indexes[L[right_child(i)]]) = (indexes[L[right_child(i)]], indexes[L[i]])
    down_heapify(L, values, indexes, right_child(i))
    return L, indexes


def shortest_dist_node(dist):
    best_node = 'undefined'
    best_value = 1000000
    for v in dist:
        if dist[v] < best_value:
            (best_node, best_value) = (v, dist[v])
    return best_node


def dijkstra(G, v):
    # Implementation of Dijkstra's algorithm to find shortest path to each node in graph, G, from node, v.
    # Initialise a min heap with node v, using a dict to store its value and a dict for its index in the heap for quick
    # access of either value, necessary during this function. Top value (min) of the heap is taken as the current node
    # then replaced by the bottom value, and heap down heapified to reform the heap with the next shortest value now at
    # the top of the heap. The neighbours of the current node are checked, and if they arent finished already, we can
    # add or update their distance so far then add to the heap if necessary then up heapify from that node to maintain
    # the heap. After all neighbours are checked, the heap now contains to shortest path so far as the top value and the
    # loop can continue finding the next shortest path till all nodes shortest paths are found

    dist_so_far_heap = [v]  # use heap with initial node key as root or top of heap as it has min distance
    unfinished_nodes = {v: 0}
    heap_indexes = {v: 0}
    final_dist = {}
    # continue till all nodes final shortest path value is determined
    while len(final_dist) < len(G):
        print()
        print("heap", dist_so_far_heap)
        print("distances", unfinished_nodes)
        print("indexes", heap_indexes)
        # find shortest dist neighbour (will be top of heap), and use that as the next current node
        curr_node = dist_so_far_heap[0]
        # set the final distance for this current node and delete from dist_so_far dict
        final_dist[curr_node] = unfinished_nodes[curr_node]
        print("curr node:", curr_node)
        print("final dist:", final_dist)
        # replace top of heap with the value at bottom of heap (removing the bottom value)
        # then down heapify to ensure a new valid heap is formed so we can find next shortest path
        dist_so_far_heap[0] = dist_so_far_heap[len(dist_so_far_heap)-1]
        heap_indexes[dist_so_far_heap[0]] = 0
        del dist_so_far_heap[len(dist_so_far_heap)-1]
        del unfinished_nodes[curr_node]
        del heap_indexes[curr_node]
        dist_so_far_heap, heap_indexes = down_heapify(dist_so_far_heap, unfinished_nodes, heap_indexes, 0)
        print("down heapify: ", dist_so_far_heap)
        print()
        # check neighbours of current node to see if the distance to them from curr node is shortest path
        for x in G[curr_node]:
            if x not in final_dist:  # neighbour is a child not a parent ie hasnt got a final distance yet
                if x not in unfinished_nodes:  # havent given distance so far yet, calculate and set it
                    unfinished_nodes[x] = final_dist[curr_node] + G[curr_node][x]
                    # add to end of heap for sorting using up heapify
                    print("new node: ", x)
                    dist_so_far_heap.append(x)
                    heap_indexes[x] = len(dist_so_far_heap) - 1
                    dist_so_far_heap, heap_indexes = dijk_up_heapify(dist_so_far_heap, unfinished_nodes, heap_indexes,
                                                                     len(dist_so_far_heap) - 1)
                # has a distance already, check if this path is shorter, if so, update its value
                elif final_dist[curr_node] + G[curr_node][x] < unfinished_nodes[x]:
                    unfinished_nodes[x] = final_dist[curr_node] + G[curr_node][x]
                    print("new dist", x, ":", unfinished_nodes[x])
                    print("heap index: ", heap_indexes[x])
                    # must try up heapify the changed value as heap may not be valid now
                    dist_so_far_heap, heap_indexes = dijk_up_heapify(dist_so_far_heap, unfinished_nodes, heap_indexes,
                                                                     heap_indexes[x])
    return final_dist


def test_dijk():
    (a,b,c,d,e,f,g) = ('A', 'B', 'C', 'D', 'E', 'F','G')
    # triples = ((a,c,4),(c,b,1),(a,b,6),(d,b,5),(c,d,7),(d,f,10),(d,e,7),(e,f,2))
    triples = ((a,c,3),(c,b,10),(a,b,15),(d,b,9),(a,d,4),(d,f,7),(d,e,3),
               (e,g,1),(e,f,5),(f,g,2),(b,f,1))
    G = {}
    for (i,j,k) in triples:
        make_link_weighted(G, i, j, k)

    dist = dijkstra(G, a)
    print(dist)


#
# Modify long_and_simple_path
# to build and return the path
#

# Find me that path!
def long_and_simple_path(G, u, v, l):
    """
    G: Graph
    u: starting node
    v: ending node
    l: minimum length of path
    """

    from copy import deepcopy

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


# In the lecture, we described how a solution to k_clique_decision(G, k)
# can be used to solve independent_set_decision(H,s).
# Write a Python function that carries out this transformation.

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


# Decision problems are often just as hard as actually returning an answer.
# Show how a k-clique can be found using a solution to the k-clique decision
# problem.  Write a Python function that takes a graph G and a number k
# as input, and returns a list of k nodes from G that are all connected
# in the graph.  Your function should make use of "k_clique_decision(G, k)",
# which takes a graph G and a number k and answers whether G contains a k-clique.
# We will also provide the standard routines for adding and removing edges from a graph.
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
# time = timeit.timeit(setup=setup2, stmt=code2, number=10000)
# print(time)


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
        print(current_node)
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
    print("links", links)
    print("max weight", max_weight)
    # if we made it to node, j, backtrack to determine that path
    if j in links:
        path_to_j = [j]
        node = j
        # backtrack from j to get path from i, creating a list of nodes in the path
        while node != i:
            node = links[node]
            path_to_j.insert(0, node)
        print("path to j", path_to_j)
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
            print("path to max", path_to_max)
            # determine index of the node of intersection, then insert the forward path to max at this point
            insert_index = path_to_j.index(path_to_max[-1]) + 1
            for num, node in enumerate(reversed(path_to_max[1:-1])):
                path_to_j.insert(insert_index + num, node)
            # now add the reverse path, updating the insert index, taking into account the forward paths length
            insert_index += len(path_to_max) - 2
            for num, node in enumerate(path_to_max):
                path_to_j.insert(insert_index + num, node)
        print("max path to j", path_to_j)
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

# test_love()


# Modify Dijkstras algorithm to find shortest path, on a graph with lexicographically ordered weightings
def find_best_flights(flights, origin, destination):
    from collections import deque

    # TODO: we have to search all possibilities? taking lowest cost changes time, and edges are time
    # dependent. So taking lowest cost at each step, cuts options from next destination, which we could check, but then
    # we have to check all combinations, as each time change compounds to next destination,
    # TODO: BFS/DFS searches all nodes, save best path so far at each node along with cost, then change if find better,
    # so for example: b=ab1,$1, c=ab1bc1,$4, d=ab1bc1cd1,$5 -> b=same, c=ab2bc2,$3, d=ab2bc2cd1=$4?????

    # run Dijkstra on each flight for origin given, where it saves best path + cost in best_to_country, so at
    # each step we have to calculate the path and cost, see if its better than the best saved for destination country
    # and update when necessary, for instance, a-b we calculate on init, as the flight we select, and the lowest cost
    # flight will be best to b, but a-c will be calculated from combinations of flights from a-b and b-c eg.
    # ab8->bc11 = $3,3h a-c, while ab12->bc15 = $4,4.5h
    # Can only take a flight from destination country if arrival time(curr) < departure time(next)
    # Add new flights to queue
    # This connects flights to other flights, given the country and time are compatible, without explicitly making
    # edges between flight nodes

    best_paths = {}
    flights_graph = {}
    # save all flights from country in dict with country as key for ease of access
    for flight in flights:
        make_flights(flights_graph, flight, best_paths)

    # make a queue for each flight from origin, then run a modified Dijkstra's algorithm starting at each flight
    # each item in the queue is a list containing, the country, flight number, and cost, time and path so far

    # TODO: check useage of bests in graph vs queue, are both needed?
    # TODO: could use a min heap, with min being cost, to take least cost path first - liek Dijkstra
    queue = deque([[origin, x, 0, 0, []] for x in flights_graph[origin]])
    # run till we have exhausted the queue, move onto next starting flight
    while queue:
        # pop next flight to be checked, and initialise the details associated with the path in the form:
        # [origin, flight num, best cost, best time, best path]
        flight_data = queue.popleft()
        current_country, flight_num, current_cost, current_time, current_path = \
            flight_data[0], flight_data[1], flight_data[2], flight_data[3], flight_data[4]
        # [origin, destination, depart, arrive, cost, best path, best cost, best time]
        current_flight = flights_graph[current_country][flight_num]

        # We found a better path to this flight, after adding current path to queue, so discontinue this path
        # TODO: can this be done another way, maybe by just checking the current flight best path ?
        if current_country != origin and current_flight[5] != current_path:
            # print('found a better path to this flight before we popped this')
            continue

        # add the flight's cost and time to the current, update current country to flight destination
        current_cost += current_flight[4]
        current_time += current_flight[3] - current_flight[2]
        # must now extend the current path to include this flight
        current_path.append(flight_num)
        arrival = current_flight[3]
        # must update current country, as we have taken flight
        current_country = current_flight[1]

        # check if updating best path to each country is necessary - could keep track of destination only
        if best_paths[current_country] is None or best_paths[current_country][1:] > [current_cost, current_time]:
            best_paths[current_country] = [current_path, current_cost, current_time]

        # reached destination country, end this path
        if current_country == destination:
            # print("reached destination")
            continue
        # Break search early if we go above the highest found cost for a path to our destination...
        if best_paths[destination] and best_paths[destination][1] < current_cost:
            # print("current cost greater than best to destination...terminate search")
            continue

        # flight data: [origin, destination, depart, arrive, cost, best path, best cost, best time]
        # Continuing the path...search destination country for connectable flights, if in time for it and lower cost
        # than that flights best, add to queue with the current path
        for next_flight in flights_graph[current_country]:
            flight_data = flights_graph[current_country][next_flight]
            waiting_time = flight_data[2] - arrival  # time between arrival and next flight
            # make sure departure time is after current and check best cost so we only consider cheaper paths to the
            # next flight. TODO: reject cyclical paths, ie already seen countries in this path - howto? use next flight?
            if arrival > flight_data[2] or flight_data[-2] < current_cost\
                    or (flight_data[-2] == current_cost and flight_data[-1] < current_time + waiting_time):
                # print("flight rejected")
                continue
            else:  # flight can be connected, add to queue with relevant data and update bests for this new flight
                flight_data[-3], flight_data[-2], flight_data[-1] = \
                    current_path, current_cost, (current_time + waiting_time)
                # TODO: could just use best cost+time, without path, then if either are different to best when we pop
                # off queue, we know its a different path,
                # have to use copy of current path, as it was updating the path in the queue as we looped
                queue.append([flight_data[0], next_flight, current_cost, (current_time+waiting_time), current_path.copy()])

    # return the best path to destination country
    if best_paths[destination]:
        return best_paths[destination][0]
    else:
        return


# Makes a graph full of nodes, but no edges, we assume edges based on the countries in the flight and the timing
def make_flights(G, flight, best_paths):
    flight_num, origin, destination, depart, arrive, cost = \
        flight[0], flight[1], flight[2], flight[3], flight[4], flight[5]
    # convert time to minutes as an integer for ease of handling
    depart = int(depart[:2]) * 60 + int(depart[3:])
    arrive = int(arrive[:2]) * 60 + int(arrive[3:])
    if origin not in G:
        G[origin] = {}
    if origin not in best_paths:
        best_paths[origin] = None
    # in case destination has no outgoing flights...still needs to be in graph
    if destination not in G:
        G[destination] = {}
    if destination not in best_paths:
        best_paths[destination] = None
    if flight_num not in G[origin]:
        # save flight with all data and add a list and 2ints for best path+cost+time taken
        (G[origin])[flight_num] = [origin, destination, depart, arrive, cost, [], 1000000, 1000000]
    return G


# Each tuple contains six items:
#   Flight Number, Origin, Destination, Departure Time, Arrival Time, Cost
# (Don't worry about any time zone issues; assume everything happens
# in the same time zone)
# Also note that overnight layovers are not allowed.
all_flights = [(523, 'Broome', 'Derby', '07:17', '08:57', 60),
               (526, 'Broome', 'Derby', '08:41', '10:30', 50),
               (527, 'Broome', 'Derby', '11:46', '13:24', 200),
               (530, 'Broome', 'Derby', '14:23', '15:59', 50),
               (540, 'Broome', 'Derby', '17:49', '19:40', 50),
               (546, 'Broome', 'Derby', '20:34', '22:09', 20),
               (547, 'Broome', 'Perth', '06:41', '08:44', 30),
               (549, 'Broome', 'Perth', '17:16', '19:18', 100),
               (559, 'Carnarvon', 'Geraldton', '09:05', '10:57', 50),
               (561, 'Carnarvon', 'Geraldton', '11:14', '13:03', 30),
               (578, 'Carnarvon', 'Geraldton', '14:56', '16:48', 150),
               (582, 'Carnarvon', 'Geraldton', '17:05', '18:46', 50),
               (598, 'Carnarvon', 'Geraldton', '22:08', '23:49', 20),
               (599, 'Carnarvon', 'Perth', '07:04', '09:46', 200),
               (100, 'Carnarvon', 'Perth', '10:53', '13:38', 60),
               (604, 'Carnarvon', 'Perth', '14:50', '17:16', 200),
               (612, 'Carnarvon', 'Perth', '19:54', '22:38', 50),
               (107, 'Derby', 'Broome', '08:44', '10:36', 160),
               (108, 'Derby', 'Broome', '21:18', '23:04', 30),
               (622, 'Derby', 'Fitzroy Crossing', '13:59', '15:04', 60),
               (112, 'Derby', 'Fitzroy Crossing', '19:24', '20:15', 60),
               (113, 'Derby', 'Geraldton', '07:00', '08:10', 20),
               (115, 'Derby', 'Geraldton', '10:00', '11:07', 200),
               (118, 'Derby', 'Geraldton', '13:24', '14:31', 50),
               (121, 'Derby', 'Geraldton', '14:41', '15:52', 50),
               (122, 'Derby', 'Geraldton', '17:05', '18:09', 60),
               (635, 'Derby', 'Geraldton', '18:59', '20:18', 60),
               (638, 'Fitzroy Crossing', 'Derby', '09:18', '10:08', 50),
               (131, 'Fitzroy Crossing', 'Derby', '13:59', '14:51', 160),
               (226, 'Fitzroy Crossing', 'Derby', '14:34', '15:34', 110),
               (139, 'Fitzroy Crossing', 'Derby', '18:43', '19:36', 50),
               (654, 'Fitzroy Crossing', 'Halls Creek', '07:55', '09:48', 180),
               (143, 'Fitzroy Crossing', 'Halls Creek', '09:45', '11:39', 20),
               (280, 'Fitzroy Crossing', 'Halls Creek', '15:10', '17:07', 110),
               (660, 'Fitzroy Crossing', 'Halls Creek', '18:41', '20:24', 30),
               (661, 'Fitzroy Crossing', 'Halls Creek', '20:35', '22:19', 200),
               (663, 'Geraldton', 'Carnarvon', '08:30', '10:24', 30),
               (152, 'Geraldton', 'Carnarvon', '12:52', '14:42', 50),
               (153, 'Geraldton', 'Carnarvon', '15:24', '17:15', 30),
               (154, 'Geraldton', 'Carnarvon', '18:07', '19:53', 180),
               (671, 'Geraldton', 'Derby', '06:01', '07:10', 120),
               (676, 'Geraldton', 'Derby', '10:46', '12:09', 20),
               (165, 'Geraldton', 'Derby', '11:29', '12:45', 30),
               (683, 'Geraldton', 'Derby', '14:17', '15:23', 50),
               (174, 'Geraldton', 'Derby', '16:45', '17:58', 180),
               (175, 'Geraldton', 'Derby', '18:31', '19:47', 20),
               (179, 'Halls Creek', 'Fitzroy Crossing', '06:32', '08:22', 200),
               (187, 'Halls Creek', 'Fitzroy Crossing', '13:19', '15:03', 200),
               (702, 'Halls Creek', 'Fitzroy Crossing', '14:04', '15:45', 20),
               (192, 'Halls Creek', 'Fitzroy Crossing', '20:08', '21:59', 160),
               (195, 'Halls Creek', 'Kalbarri', '06:43', '09:01', 110),
               (709, 'Halls Creek', 'Kalbarri', '08:45', '11:04', 200),
               (199, 'Halls Creek', 'Kalbarri', '13:21', '15:39', 20),
               (209, 'Halls Creek', 'Kalbarri', '15:45', '18:01', 100),
               (723, 'Halls Creek', 'Kalbarri', '16:04', '18:10', 50),
               (724, 'Halls Creek', 'Kalbarri', '19:52', '22:07', 160),
               (216, 'Kalbarri', 'Halls Creek', '06:15', '08:34', 100),
               (217, 'Kalbarri', 'Halls Creek', '14:57', '17:14', 200),
               (730, 'Kalbarri', 'Halls Creek', '21:05', '23:24', 20),
               (731, 'Kalbarri', 'Perth', '06:18', '08:50', 50),
               (734, 'Kalbarri', 'Perth', '12:23', '14:59', 120),
               (735, 'Kalbarri', 'Perth', '12:59', '15:19', 30),
               (738, 'Kalbarri', 'Perth', '18:41', '21:10', 60),
               (739, 'Kalbarri', 'Perth', '19:42', '22:18', 60),
               (740, 'Laverton', 'Leonora', '07:39', '08:53', 180),
               (745, 'Laverton', 'Leonora', '12:20', '13:32', 20),
               (748, 'Laverton', 'Leonora', '13:44', '15:08', 30),
               (751, 'Laverton', 'Leonora', '18:00', '19:11', 200),
               (240, 'Laverton', 'Leonora', '20:34', '21:40', 110),
               (754, 'Laverton', 'Perth', '07:21', '08:21', 180),
               (247, 'Laverton', 'Perth', '20:11', '21:22', 160),
               (248, 'Leinster', 'Perth', '08:37', '11:16', 180),
               (249, 'Leinster', 'Perth', '13:44', '16:12', 110),
               (763, 'Leinster', 'Perth', '16:29', '19:06', 160),
               (765, 'Leinster', 'Perth', '19:17', '21:47', 20),
               (981, 'Leinster', 'Wiluna', '10:51', '13:03', 200),
               (770, 'Leinster', 'Wiluna', '16:02', '18:17', 50),
               (259, 'Leinster', 'Wiluna', '19:44', '22:09', 60),
               (772, 'Leonora', 'Laverton', '10:39', '11:59', 110),
               (987, 'Leonora', 'Laverton', '15:56', '17:13', 110),
               (264, 'Leonora', 'Laverton', '21:39', '22:48', 200),
               (779, 'Leonora', 'Perth', '10:29', '11:59', 50),
               (780, 'Leonora', 'Perth', '11:26', '12:58', 50),
               (783, 'Leonora', 'Perth', '19:48', '21:25', 30),
               (278, 'Meekatharra', 'Mt Magnet', '07:40', '08:42', 60),
               (792, 'Meekatharra', 'Mt Magnet', '08:35', '09:35', 60),
               (793, 'Meekatharra', 'Mt Magnet', '11:50', '12:44', 110),
               (796, 'Meekatharra', 'Mt Magnet', '14:32', '15:26', 30),
               (798, 'Meekatharra', 'Mt Magnet', '16:56', '17:52', 160),
               (288, 'Meekatharra', 'Mt Magnet', '19:38', '20:27', 60),
               (289, 'Meekatharra', 'Perth', '08:12', '09:28', 50),
               (803, 'Meekatharra', 'Perth', '09:12', '10:25', 30),
               (805, 'Meekatharra', 'Perth', '12:10', '13:16', 50),
               (298, 'Meekatharra', 'Perth', '13:33', '14:40', 50),
               (391, 'Meekatharra', 'Perth', '16:45', '17:50', 30),
               (815, 'Meekatharra', 'Perth', '20:17', '21:29', 110),
               (817, 'Monkey Mia', 'Perth', '08:26', '10:51', 20),
               (393, 'Monkey Mia', 'Perth', '13:12', '15:51', 30),
               (825, 'Monkey Mia', 'Perth', '21:01', '23:37', 180),
               (314, 'Mt Magnet', 'Meekatharra', '06:29', '07:30', 30),
               (827, 'Mt Magnet', 'Meekatharra', '08:56', '10:00', 50),
               (829, 'Mt Magnet', 'Meekatharra', '13:09', '14:14', 30),
               (832, 'Mt Magnet', 'Meekatharra', '14:10', '15:09', 30),
               (833, 'Mt Magnet', 'Meekatharra', '17:39', '18:41', 180),
               (322, 'Mt Magnet', 'Meekatharra', '19:51', '20:55', 160),
               (333, 'Mt Magnet', 'Perth', '07:53', '08:38', 120),
               (846, 'Mt Magnet', 'Perth', '15:45', '16:29', 20),
               (967, 'Mt Magnet', 'Perth', '18:04', '18:49', 20),
               (336, 'Mt Magnet', 'Wiluna', '07:34', '09:08', 200),
               (338, 'Mt Magnet', 'Wiluna', '13:35', '15:17', 30),
               (856, 'Mt Magnet', 'Wiluna', '14:54', '16:27', 50),
               (345, 'Mt Magnet', 'Wiluna', '18:03', '19:35', 50),
               (859, 'Perth', 'Broome', '07:21', '09:14', 50),
               (348, 'Perth', 'Broome', '10:37', '12:46', 60),
               (349, 'Perth', 'Broome', '12:56', '14:57', 20),
               (350, 'Perth', 'Broome', '15:01', '17:11', 110),
               (356, 'Perth', 'Broome', '18:03', '20:03', 60),
               (364, 'Perth', 'Broome', '18:45', '20:54', 150),
               (880, 'Perth', 'Carnarvon', '07:39', '10:09', 50),
               (884, 'Perth', 'Carnarvon', '10:33', '13:11', 30),
               (374, 'Perth', 'Carnarvon', '12:04', '14:31', 50),
               (375, 'Perth', 'Carnarvon', '13:59', '16:32', 30),
               (378, 'Perth', 'Carnarvon', '17:04', '19:38', 50),
               (299, 'Perth', 'Carnarvon', '19:27', '22:09', 50),
               (383, 'Perth', 'Kalbarri', '06:41', '09:12', 120),
               (384, 'Perth', 'Kalbarri', '12:42', '15:03', 20),
               (898, 'Perth', 'Kalbarri', '19:13', '21:38', 30),
               (390, 'Perth', 'Laverton', '10:20', '11:23', 60),
               (321, 'Perth', 'Laverton', '14:08', '15:03', 60),
               (905, 'Perth', 'Laverton', '19:58', '20:53', 100),
               (395, 'Perth', 'Leinster', '06:59', '09:28', 200),
               (396, 'Perth', 'Leinster', '10:17', '12:48', 100),
               (401, 'Perth', 'Leinster', '14:24', '16:50', 50),
               (914, 'Perth', 'Leinster', '18:54', '21:34', 160),
               (404, 'Perth', 'Leonora', '11:03', '12:40', 30),
               (918, 'Perth', 'Leonora', '12:37', '14:17', 150),
               (408, 'Perth', 'Leonora', '20:42', '22:10', 100),
               (923, 'Perth', 'Meekatharra', '06:21', '07:35', 110),
               (927, 'Perth', 'Meekatharra', '10:25', '11:26', 20),
               (933, 'Perth', 'Meekatharra', '14:27', '15:24', 50),
               (934, 'Perth', 'Meekatharra', '17:49', '18:50', 200),
               (941, 'Perth', 'Meekatharra', '21:56', '23:08', 30),
               (430, 'Perth', 'Monkey Mia', '06:18', '08:48', 30),
               (943, 'Perth', 'Monkey Mia', '12:11', '14:48', 180),
               (432, 'Perth', 'Monkey Mia', '17:32', '20:13', 50),
               (433, 'Perth', 'Monkey Mia', '19:48', '22:23', 100),
               (947, 'Perth', 'Mt Magnet', '06:43', '07:23', 100),
               (948, 'Perth', 'Mt Magnet', '13:59', '14:54', 20),
               (954, 'Perth', 'Mt Magnet', '15:44', '16:26', 120),
               (955, 'Perth', 'Mt Magnet', '19:34', '20:26', 200),
               (475, 'Perth', 'Wiluna', '07:34', '09:57', 60),
               (959, 'Perth', 'Wiluna', '09:44', '12:22', 50),
               (455, 'Perth', 'Wiluna', '12:22', '14:45', 60),
               (969, 'Perth', 'Wiluna', '14:26', '16:59', 50),
               (458, 'Perth', 'Wiluna', '17:19', '19:38', 60),
               (459, 'Perth', 'Wiluna', '19:09', '21:35', 30),
               (461, 'Wiluna', 'Leinster', '07:54', '10:16', 20),
               (462, 'Wiluna', 'Leinster', '08:35', '10:50', 200),
               (463, 'Wiluna', 'Leinster', '11:50', '14:01', 200),
               (976, 'Wiluna', 'Leinster', '13:54', '16:15', 50),
               (469, 'Wiluna', 'Leinster', '17:24', '19:43', 30),
               (984, 'Wiluna', 'Leinster', '19:58', '22:13', 200),
               (847, 'Wiluna', 'Mt Magnet', '07:13', '08:42', 30),
               (478, 'Wiluna', 'Mt Magnet', '11:48', '13:14', 50),
               (993, 'Wiluna', 'Mt Magnet', '13:00', '14:27', 20),
               (483, 'Wiluna', 'Mt Magnet', '17:20', '18:57', 60),
               (422, 'Wiluna', 'Mt Magnet', '21:40', '23:21', 60),
               (494, 'Wiluna', 'Perth', '08:28', '11:07', 160),
               (253, 'Wiluna', 'Perth', '11:17', '13:41', 150),
               (498, 'Wiluna', 'Perth', '13:53', '16:13', 60),
               (501, 'Wiluna', 'Perth', '17:59', '20:27', 20),
               (505, 'Wiluna', 'Perth', '20:21', '22:41', 180)]


def test_best_flights():
    flights = find_best_flights(all_flights, 'Mt Magnet', 'Fitzroy Crossing')
    assert flights == [314, 803, 348, 530, 112]

    flights = find_best_flights(all_flights, 'Leonora', 'Fitzroy Crossing')
    assert flights is None

    flights = find_best_flights(all_flights, 'Meekatharra', 'Wiluna')
    assert flights == [391, 459]


# test is a simple example that shows taking best cost at each country can lead to a worse overall cost
flights2 = [(1, 'a', 'b', '12:00', '13:00', 2), (2, 'a', 'b', '13:00', "14:00", 1), (3, 'a', 'b', "08:00", "09:00", 2),
            (4, "b", "c", "16:00", "18:00", 2), (5, "b", "c", "15:00", "16:30", 2), (6, "b", "c", "10:00", "11:00", 1),
            (7, "b", "d", "15:00", "15:30", 1), (8, "d", "c", "16:00", "17:00", 2)]
# test showed that marking off countries early for all paths causes errors.
flights3 = [(1,'a','b','08:00','08:30',1), (2,'b','c','09:00','10:00',5), (3,'b','f','09:00','09;30',1),
            (4,'c','d','10:30','11:00',1), (5,'f','c','09:40','10;20',1), (6,'d','e','11:30','12:00',1),
            (7,'f','e','10:00','11:00',10)]


# test_best_flights()
# find_best_flights(flights2, 'a', 'c')
# find_best_flights(flights3, 'a', 'e')


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

