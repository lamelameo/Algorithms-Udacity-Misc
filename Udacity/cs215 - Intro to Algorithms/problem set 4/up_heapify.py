#
# write up_heapify, an algorithm that checks if
# node i and its parent satisfy the heap
# property, swapping and recursing if they don't
#
# L should be a heap when up_heapify is done
#


# given
def parent(i):
    return (i - 1) / 2


def up_heapify(L, i):
    # if parent of node i has a greater value than i, swap the values and recurse, else return the List as it is a heap
    # if node i reaches top of heap, the parent will be itself, and fail the comparison and return the valid heap
    # TODO: fails in udacity as it runs python 2, added an if statement to return L once top of heap is reached...
    if i == 0:
        return L
    node = L[i]
    parent_index = int(parent(i))
    parent_node = L[parent_index]
    if node < parent_node:
        L[parent_index] = node
        L[i] = parent_node
        return up_heapify(L, parent_index)
    else:
        return L


# Non recursive version
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
