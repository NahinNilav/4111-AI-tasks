from dataclasses import dataclass, field
from typing import List
from math import radians, sin, cos, sqrt, atan2


@dataclass
class SearchResult:
    path: List[int]
    path_cost: float
    nodes_expanded: int
    max_frontier_size: int
    execution_time: float
    found: bool
    exploration_order: List[int] = field(default_factory=list)


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def make_heuristic_old(G, destination, profile):
    """h_old: h(n) = w_road_length * haversine / max_edge_length.
    Admissible but weak — uses only the road_length component."""
    dest_lat = G.nodes[destination]["y"]
    dest_lon = G.nodes[destination]["x"]
    max_len = G.graph.get("max_edge_length", 1.0)
    w_rl = profile.get("road_length", 0.1)

    def h(node):
        lat = G.nodes[node]["y"]
        lon = G.nodes[node]["x"]
        return w_rl * haversine(lat, lon, dest_lat, dest_lon) / max_len

    return h


def make_heuristic(G, destination, profile):
    """h_B: h(n) = (haversine / max_edge_length) * (w_road_length + c_min)
    """
    
    dest_lat = G.nodes[destination]["y"]
    dest_lon = G.nodes[destination]["x"]
    max_len = G.graph.get("max_edge_length", 1.0)
    w_rl = profile.get("road_length", 0.1)

    # c_min: cheapest non-road-length composite cost across all edges
    c_min = min(
        sum(profile[m] * data.get(m, 0) for m in profile if m != "road_length")
        for _, _, data in G.edges(data=True)
    )

    coeff = w_rl + c_min

    def h(node):
        lat = G.nodes[node]["y"]
        lon = G.nodes[node]["x"]
        return (haversine(lat, lon, dest_lat, dest_lon) / max_len) * coeff

    return h


def make_heuristic_dijk(G, destination, profile, landmark_data):
    
    prof_name = landmark_data["profile_name"]
    lm_table = landmark_data["per_profile"][prof_name]
    dest = destination

    # Precompute landmark→goal and goal→landmark distances once
    cached = []
    for L, dists in lm_table.items():
        fwd_L_goal = dists["forward"].get(dest, float("inf"))
        bwd_goal_L = dists["backward"].get(dest, float("inf"))
        cached.append((L, dists["forward"], dists["backward"], fwd_L_goal, bwd_goal_L))

    def h(node):
        best = 0.0
        for L, fwd, bwd, fwd_L_goal, bwd_goal_L in cached:
            fwd_L_n = fwd.get(node, float("inf"))
            bwd_n_L = bwd.get(node, float("inf"))

            # Bound (a): d(L, goal) - d(L, n)
            if fwd_L_goal != float("inf") and fwd_L_n != float("inf"):
                lb_a = fwd_L_goal - fwd_L_n
                if lb_a > best:
                    best = lb_a

            # Bound (b): d(n, L) - d(goal, L)
            if bwd_n_L != float("inf") and bwd_goal_L != float("inf"):
                lb_b = bwd_n_L - bwd_goal_L
                if lb_b > best:
                    best = lb_b

        return max(0.0, best)

    return h


# Global cache for landmark data (loaded on first use)
_LANDMARK_CACHE = {"data": None, "path": None}


def load_landmarks(path):

    if _LANDMARK_CACHE["data"] is not None and _LANDMARK_CACHE["path"] == path:
        return _LANDMARK_CACHE["data"]
    import pickle
    with open(path, "rb") as f:
        data = pickle.load(f)
    _LANDMARK_CACHE["data"] = data
    _LANDMARK_CACHE["path"] = path
    return data


def composite_cost(edge_data, profile):
    return sum(profile[m] * edge_data.get(m, 0) for m in profile)


def get_min_edge_cost(G, u, v, profile):

    best = float("inf")
    best_key = None
    for key, data in G[u][v].items():
        c = composite_cost(data, profile)
        if c < best:
            best = c
            best_key = key
    return best, best_key


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def path_cost(G, path, profile):
    total = 0
    for i in range(len(path) - 1):
        cost, _ = get_min_edge_cost(G, path[i], path[i + 1], profile)
        total += cost
    return total
