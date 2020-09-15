# Generate a combination lock graph given a list of nodes
#


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
