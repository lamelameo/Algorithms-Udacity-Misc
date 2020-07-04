#
# Write centrality_max to return the maximum distance
# from a node to all the other nodes it can reach
#


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
