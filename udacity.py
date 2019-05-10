""" udacity tutorials and stuff """


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


def make_link(G, node1, node2):
    if node1 not in G:
        G[node1] = {}
    (G[node1])[node2] = 1
    if node2 not in G:
        G[node2] = {}
    (G[node2])[node1] = 1
    return G


def create_rooted_spanning_tree(G, root):
    # initialize stuff
    S = {}
    node_stack = [root]
    S[root] = {}
    # TODO: can replace not in with dict.get() which returns None if no value for that key, is faster
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
    # TODO: this return type, (order, ordered_list), is not allowed for the quiz...try other shitty hack,
    # TODO: by manually counting through dict values to find right order or something else
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
        del ordered_po[po[node] - 1]  # index = post order value - 1
        ordered_po.insert(po[node] - 1, node)
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
        del ordered_po[po[node] - 1]  # index = post order value - 1
        ordered_po.insert(po[node] - 1, node)

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
            print(node)
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


test_bridge_edges()

