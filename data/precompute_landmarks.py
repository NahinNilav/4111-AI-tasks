
import os, sys, pickle, heapq, random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PROFILES, RANDOM_SEED
from algorithms.base import get_min_edge_cost, composite_cost

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
K_LANDMARKS = 12  # number of landmarks per graph


def dijkstra_from(G, source, profile, reverse=False):

    dist = {source: 0.0}
    heap = [(0.0, source)]
    visited = set()

    neighbors_fn = G.predecessors if reverse else G.successors

    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        for v in neighbors_fn(u):
            # For reverse, edge is v→u in original graph; cost is on that edge
            if reverse:
                edge_cost, _ = get_min_edge_cost(G, v, u, profile)
            else:
                edge_cost, _ = get_min_edge_cost(G, u, v, profile)
            nd = d + edge_cost
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))

    return dist


def select_landmarks_farthest(G, k, seed_node, profile):

    landmarks = [seed_node]
    # Distance from any current landmark (min over landmarks)
    print(f"    Selecting {k} landmarks via farthest-point heuristic...")

    # Initialize min_dist from seed
    dist_from_seed = dijkstra_from(G, seed_node, profile, reverse=False)
    min_dist = dict(dist_from_seed)

    for i in range(1, k):
        # Pick the node with max min_dist
        next_lm = None
        best = -1
        for n, d in min_dist.items():
            if d > best and n not in landmarks:
                best = d
                next_lm = n
        if next_lm is None:
            break
        landmarks.append(next_lm)
        print(f"      Landmark {i+1}: node {next_lm} (farthest = {best:.3f})")

        # Update min_dist with distances from new landmark
        new_dist = dijkstra_from(G, next_lm, profile, reverse=False)
        for n, d in new_dist.items():
            if n not in min_dist or d < min_dist[n]:
                min_dist[n] = d

    return landmarks


def precompute_for_graph(G, graph_name):

    rng = random.Random(RANDOM_SEED)

    # Pick a seed landmark (random) — same seed gives reproducible results
    all_nodes = list(G.nodes)
    seed = rng.choice(all_nodes)

    # We select landmarks ONCE using the 'balanced' profile (neutral spatial coverage)
    # Then precompute distances separately for each profile.
    print(f"\n[{graph_name}] Selecting landmarks (balanced profile for spatial spread)...")
    landmarks = select_landmarks_farthest(G, K_LANDMARKS, seed, PROFILES["balanced"])
    print(f"  {len(landmarks)} landmarks selected.")

    # For each profile, run forward + backward Dijkstra from each landmark
    per_profile = {}
    for prof_name, profile in PROFILES.items():
        print(f"\n[{graph_name}] Computing distances for profile '{prof_name}'...")
        per_profile[prof_name] = {}
        for i, L in enumerate(landmarks):
            fwd = dijkstra_from(G, L, profile, reverse=False)
            bwd = dijkstra_from(G, L, profile, reverse=True)
            per_profile[prof_name][L] = {"forward": fwd, "backward": bwd}
            print(f"    Landmark {i+1}/{len(landmarks)} ({L}): "
                  f"fwd covers {len(fwd)} nodes, bwd covers {len(bwd)} nodes")

    return {"landmarks": landmarks, "per_profile": per_profile}


def main():
    
    output = {}
    for name in ["dhaka_drive"]:  
        path = os.path.join(DATA_DIR, f"{name}.gpickle")
        print(f"\nLoading {path}...")
        with open(path, "rb") as f:
            G = pickle.load(f)
        print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        output[name] = precompute_for_graph(G, name)

    out_path = os.path.join(DATA_DIR, "landmarks.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\n✓ Saved landmark tables to {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
