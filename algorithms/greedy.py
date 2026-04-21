import time
import heapq
from .base import (SearchResult, reconstruct_path, path_cost,
                   make_heuristic, make_heuristic_old, make_heuristic_dijk)


def greedy_search(G, source, destination, profile, heuristic_version="h_B",
                   landmark_data=None, **kwargs):
    start = time.time()
    if heuristic_version == "h_ALT":
        h = make_heuristic_dijk(G, destination, profile, landmark_data)
    elif heuristic_version == "h_B":
        h = make_heuristic(G, destination, profile)
    else:
        h = make_heuristic_old(G, destination, profile)

    frontier = [(h(source), source)]
    came_from = {}
    visited = set()
    exploration_order = []
    max_frontier = 1

    while frontier:
        max_frontier = max(max_frontier, len(frontier))
        _, node = heapq.heappop(frontier)

        if node in visited:
            continue
        visited.add(node)
        exploration_order.append(node)

        if node == destination:
            p = reconstruct_path(came_from, destination)
            return SearchResult(
                path=p, path_cost=path_cost(G, p, profile),
                nodes_expanded=len(exploration_order),
                max_frontier_size=max_frontier,
                execution_time=time.time() - start,
                found=True, exploration_order=exploration_order)

        for neighbor in G.successors(node):
            if neighbor not in visited:
                came_from[neighbor] = node
                heapq.heappush(frontier, (h(neighbor), neighbor))

    return SearchResult(
        path=[], path_cost=float("inf"),
        nodes_expanded=len(exploration_order),
        max_frontier_size=max_frontier,
        execution_time=time.time() - start,
        found=False, exploration_order=exploration_order)
