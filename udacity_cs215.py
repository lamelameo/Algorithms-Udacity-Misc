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


# Given this function
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
    # return mode

    # initialise the mode key arbitrarily to first item in list (it will be updated if it isnt the mode)
    mode_key = L[0]
    for value in frequencies:
        # determine if the current highest is lower than the value we are currently checking
        if frequencies[value] > frequencies[mode_key]:
            mode_key = value
    return mode_key


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


def test():
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



