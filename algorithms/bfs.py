import time
from collections import deque
from .base import SearchResult, reconstruct_path, path_cost


def bfs_search(G, source, destination, profile, **kwargs):
    start = time.time()
    frontier = deque([source])
    came_from = {}
    visited = {source}
    exploration_order = []
    max_frontier = 1

    while frontier:
        max_frontier = max(max_frontier, len(frontier))
        node = frontier.popleft()
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
                visited.add(neighbor)
                came_from[neighbor] = node
                frontier.append(neighbor)

    return SearchResult(
        path=[], path_cost=float("inf"),
        nodes_expanded=len(exploration_order),
        max_frontier_size=max_frontier,
        execution_time=time.time() - start,
        found=False, exploration_order=exploration_order)
