# Rewrite `mark_component` to not use recursion
# and instead use the `open_list` data structure
# discussed in lecture
#


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
