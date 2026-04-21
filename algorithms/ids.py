import time
from .base import SearchResult, reconstruct_path, path_cost


def _dls(G, source, destination, depth_limit):
    
    frontier = [(source, 0)]
    came_from = {}
    visited = set()
    exploration_order = []
    max_frontier = 1

    while frontier:
        max_frontier = max(max_frontier, len(frontier))
        node, depth = frontier.pop()

        if node in visited:
            continue
        visited.add(node)
        exploration_order.append(node)

        if node == destination:
            p = reconstruct_path(came_from, destination)
            return True, p, exploration_order, max_frontier

        if depth < depth_limit:
            for neighbor in G.successors(node):
                if neighbor not in visited:
                    came_from[neighbor] = node
                    frontier.append((neighbor, depth + 1))

    return False, [], exploration_order, max_frontier


def ids_search(G, source, destination, profile, max_depth=200, **kwargs):
    start = time.time()
    all_exploration = []
    total_expanded = 0
    overall_max_frontier = 0

    for depth in range(max_depth):
        found, p, exploration, max_f = _dls(G, source, destination, depth)
        all_exploration = exploration  # keep last iteration's trace
        total_expanded += len(exploration)
        overall_max_frontier = max(overall_max_frontier, max_f)

        if found:
            return SearchResult(
                path=p, path_cost=path_cost(G, p, profile),
                nodes_expanded=total_expanded,
                max_frontier_size=overall_max_frontier,
                execution_time=time.time() - start,
                found=True, exploration_order=all_exploration)

    return SearchResult(
        path=[], path_cost=float("inf"),
        nodes_expanded=total_expanded,
        max_frontier_size=overall_max_frontier,
        execution_time=time.time() - start,
        found=False, exploration_order=all_exploration)
