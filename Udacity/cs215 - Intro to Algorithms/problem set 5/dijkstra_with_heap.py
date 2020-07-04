# Implement Dijkstra's algorithm using a heap instead of linear scan to find smallest distance from source node


# given these functions
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


def dijk_up_heapify(L, values, indexes, i):
    # takes a heap as a list of node keys, a dict of the corresponding values for the keys, a dict containing the
    # indexes of those nodes in the heap, and a reference node index and makes appropriate swaps for node i and parents
    # recursively till list is a heap, returning the list and dict of indexes which may have been altered.
    # The indexes must be updated as the nodes are moved in the heap, since we must know
    # their position to up heapify correctly if a new shorter path is found and its value is updated
    if i == 0:
        return L, indexes
    node = L[i]
    parent_index = int(parent(i))
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
        # find shortest dist neighbour (will be top of heap), and use that as the next current node
        curr_node = dist_so_far_heap[0]
        # set the final distance for this current node and delete from dist_so_far dict
        final_dist[curr_node] = unfinished_nodes[curr_node]
        # replace top of heap with the value at bottom of heap (removing the bottom value)
        # then down heapify to ensure a new valid heap is formed so we can find next shortest path
        dist_so_far_heap[0] = dist_so_far_heap[len(dist_so_far_heap)-1]
        heap_indexes[dist_so_far_heap[0]] = 0
        del dist_so_far_heap[len(dist_so_far_heap)-1]
        del unfinished_nodes[curr_node]
        del heap_indexes[curr_node]
        dist_so_far_heap, heap_indexes = down_heapify(dist_so_far_heap, unfinished_nodes, heap_indexes, 0)
        # check neighbours of current node to see if the distance to them from curr node is shortest path
        for x in G[curr_node]:
            if x not in final_dist:  # neighbour is a child not a parent ie hasnt got a final distance yet
                if x not in unfinished_nodes:  # havent given distance so far yet, calculate and set it
                    unfinished_nodes[x] = final_dist[curr_node] + G[curr_node][x]
                    # add to end of heap for sorting using up heapify
                    dist_so_far_heap.append(x)
                    heap_indexes[x] = len(dist_so_far_heap) - 1
                    dist_so_far_heap, heap_indexes = dijk_up_heapify(dist_so_far_heap, unfinished_nodes, heap_indexes,
                                                                     len(dist_so_far_heap) - 1)
                # has a distance already, check if this path is shorter, if so, update its value
                elif final_dist[curr_node] + G[curr_node][x] < unfinished_nodes[x]:
                    unfinished_nodes[x] = final_dist[curr_node] + G[curr_node][x]
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
