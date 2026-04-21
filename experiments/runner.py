"""
Experiment Runner

4 routes × 3 profiles = 12 scenarios, run on 13 algorithm configs each:
  4 uninformed:           BFS, DFS, UCS, IDS
  3 informed (h_old):     Greedy, A*, Weighted A*
  3 informed (h_B):       Greedy, A*, Weighted A*
  3 informed (h_ALT):     Greedy, A*, Weighted A*

Each timed over NUM_TIMING_RUNS repetitions. Output: results.csv
"""

import os, sys, time, pickle
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import PROFILES, SD_PAIRS, WEIGHTED_ASTAR_W
from algorithms.base import haversine, make_heuristic, make_heuristic_old
from algorithms import ALL_ALGORITHMS

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
LANDMARKS_PATH = os.path.join(DATA_DIR, "landmarks.pkl")

NUM_TIMING_RUNS = 3  # timing average (kept low — h_ALT is ~17 Dijkstras/run in precompute scope)

# 4 routes: short, medium diagonal, west-east cross, medium diagonal
ROUTE_INDICES = [3, 0, 2, 4]
PROFILE_NAMES = ["fastest", "safest", "balanced"]


def nearest_node(G, lat, lon):
    best, best_d = None, float("inf")
    for n in G.nodes:
        d = haversine(lat, lon, G.nodes[n]["y"], G.nodes[n]["x"])
        if d < best_d:
            best, best_d = n, d
    return best


def run_one(algo_fn, G, src, dst, profile, heuristic_version="h_B", w=1.5, landmark_data=None):
    kwargs = {"heuristic_version": heuristic_version, "w": w, "landmark_data": landmark_data}
    return algo_fn(G, src, dst, profile, **kwargs)


def run_timed(algo_fn, G, src, dst, profile, heuristic_version="h_B", w=1.5,
              landmark_data=None, n_runs=NUM_TIMING_RUNS):
    results = []
    for _ in range(n_runs):
        r = run_one(algo_fn, G, src, dst, profile, heuristic_version, w, landmark_data)
        results.append(r)
    avg_time = sum(r.execution_time for r in results) / n_runs * 1000
    return results[0], avg_time


def main():
    print("Loading drive graph...")
    with open(os.path.join(DATA_DIR, "dhaka_drive.gpickle"), "rb") as f:
        G = pickle.load(f)
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Load landmark tables
    if not os.path.exists(LANDMARKS_PATH):
        print(f"\nERROR: {LANDMARKS_PATH} not found.")
        print("Run `python3 data/precompute_landmarks.py` first.")
        return
    print(f"\nLoading landmark tables from {LANDMARKS_PATH}...")
    with open(LANDMARKS_PATH, "rb") as f:
        all_landmarks = pickle.load(f)
    landmark_data_raw = all_landmarks["dhaka_drive"]
    print(f"  {len(landmark_data_raw['landmarks'])} landmarks, 3 profiles")

    # Resolve S-D pairs
    resolved_pairs = []
    for idx in ROUTE_INDICES:
        pair = SD_PAIRS[idx]
        src = nearest_node(G, pair["source"][0], pair["source"][1])
        dst = nearest_node(G, pair["destination"][0], pair["destination"][1])
        dist_m = haversine(G.nodes[src]["y"], G.nodes[src]["x"],
                           G.nodes[dst]["y"], G.nodes[dst]["x"])
        resolved_pairs.append({
            "name": pair["name"], "character": pair["character"],
            "src": src, "dst": dst, "straight_line_m": dist_m,
        })
        print(f"  {pair['name']}: src={src} dst={dst} dist={dist_m:.0f}m")

    uninformed = ["BFS", "DFS", "UCS", "IDS"]
    informed = ["Greedy", "A*", "Weighted A*"]
    heuristics = ["h_old", "h_B", "h_ALT"]

    configs = []
    for name in uninformed:
        configs.append((name, ALL_ALGORITHMS[name], None, name))
    for h_ver in heuristics:
        for name in informed:
            configs.append((name, ALL_ALGORITHMS[name], h_ver, f"{name} ({h_ver})"))

    rows = []
    total = len(resolved_pairs) * len(PROFILE_NAMES) * len(configs)
    done = 0

    ucs_fn = ALL_ALGORITHMS["UCS"]

    for rp in resolved_pairs:
        for prof_name in PROFILE_NAMES:
            profile = PROFILES[prof_name]

            # UCS optimal for this scenario
            ucs_result = run_one(ucs_fn, G, rp["src"], rp["dst"], profile)
            ucs_optimal = ucs_result.path_cost

            # Precompute h(source) for all three heuristics
            h_old_fn = make_heuristic_old(G, rp["dst"], profile)
            h_B_fn = make_heuristic(G, rp["dst"], profile)

            # Build wrapped landmark_data with the active profile name
            ld_for_profile = {
                "profile_name": prof_name,
                "per_profile": landmark_data_raw["per_profile"],
            }
            from algorithms.base import make_heuristic_dijk
            h_ALT_fn = make_heuristic_dijk(G, rp["dst"], profile, ld_for_profile)

            h_src = {
                "h_old": h_old_fn(rp["src"]),
                "h_B": h_B_fn(rp["src"]),
                "h_ALT": h_ALT_fn(rp["src"]),
            }

            for algo_name, algo_fn, h_ver, label in configs:
                done += 1
                print(f"  [{done}/{total}] {label:25s} / {prof_name:8s} / {rp['name']}")

                result, avg_ms = run_timed(
                    algo_fn, G, rp["src"], rp["dst"], profile,
                    heuristic_version=h_ver or "h_B",
                    w=WEIGHTED_ASTAR_W,
                    landmark_data=ld_for_profile if h_ver == "h_ALT" else None,
                )

                h_val = h_src.get(h_ver, 0) if h_ver else 0

                rows.append({
                    "route": rp["name"],
                    "character": rp["character"],
                    "straight_line_m": rp["straight_line_m"],
                    "profile": prof_name,
                    "algorithm": algo_name,
                    "heuristic": h_ver or "none",
                    "label": label,
                    "found": result.found,
                    "path_cost": result.path_cost,
                    "ucs_optimal": ucs_optimal,
                    "optimality_ratio": result.path_cost / ucs_optimal if ucs_optimal > 0 else float("inf"),
                    "nodes_expanded": result.nodes_expanded,
                    "max_frontier": result.max_frontier_size,
                    "runtime_ms": round(avg_ms, 2),
                    "h_source": round(h_val, 4),
                    "h_over_optimal": round(h_val / ucs_optimal if ucs_optimal > 0 else 0, 4),
                    "path_length": len(result.path),
                })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved {len(df)} rows to {csv_path}")
    return df


if __name__ == "__main__":
    main()
