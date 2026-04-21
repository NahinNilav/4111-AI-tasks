import time
import heapq
from .base import SearchResult, reconstruct_path, get_min_edge_cost


def ucs_search(G, source, destination, profile, **kwargs):
    start = time.time()
    frontier = [(0.0, source)]
    g_cost = {source: 0.0}
    came_from = {}
    visited = set()
    exploration_order = []
    max_frontier = 1

    while frontier:
        max_frontier = max(max_frontier, len(frontier))
        cost, node = heapq.heappop(frontier)

        if node in visited:
            continue
        visited.add(node)
        exploration_order.append(node)

        if node == destination:
            p = reconstruct_path(came_from, destination)
            return SearchResult(
                path=p, path_cost=g_cost[destination],
                nodes_expanded=len(exploration_order),
                max_frontier_size=max_frontier,
                execution_time=time.time() - start,
                found=True, exploration_order=exploration_order)

        for neighbor in G.successors(node):
            if neighbor not in visited:
                edge_cost, _ = get_min_edge_cost(G, node, neighbor, profile)
                new_cost = g_cost[node] + edge_cost
                if new_cost < g_cost.get(neighbor, float("inf")):
                    g_cost[neighbor] = new_cost
                    came_from[neighbor] = node
                    heapq.heappush(frontier, (new_cost, neighbor))

    return SearchResult(
        path=[], path_cost=float("inf"),
        nodes_expanded=len(exploration_order),
        max_frontier_size=max_frontier,
        execution_time=time.time() - start,
        found=False, exploration_order=exploration_order)
