"""
Analysis — 4 tables + 4 plots comparing h_old, h_B, h_ALT.

Tables:
  1. Full comparison: 7 algorithms × best heuristic (h_ALT for informed)
  2. Heuristic strength: h_old vs h_B vs h_ALT across 3 profiles
  3. Node reduction: h_old → h_B → h_ALT for A* and Weighted A*
  4. Scenario-level: runtime and nodes per route

Plots:
  1. Bar chart — nodes expanded per algorithm (using h_ALT for informed)
  2. Grouped bar — h_old vs h_B vs h_ALT for informed algorithms
  3. Scatter — cost vs nodes expanded
  4. Heuristic quality — h(src) / optimal as % of optimal
"""

import os, sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUT_DIR, "plots")


def load_data():
    return pd.read_csv(os.path.join(OUT_DIR, "results.csv"))


# ─────────── TABLE 1: Full comparison with h_ALT ───────────
def table1_full_comparison(df):
    mask = (df["heuristic"] == "h_ALT") | (df["heuristic"] == "none")
    sub = df[mask].copy()

    agg = sub.groupby("algorithm").agg(
        avg_cost=("path_cost", "mean"),
        avg_optimality=("optimality_ratio", "mean"),
        avg_expanded=("nodes_expanded", "mean"),
        avg_frontier=("max_frontier", "mean"),
        avg_runtime_ms=("runtime_ms", "mean"),
        completeness=("found", "mean"),
    ).round(2)

    algo_order = ["BFS", "DFS", "UCS", "IDS", "Greedy", "A*", "Weighted A*"]
    agg = agg.reindex(algo_order)
    agg.index.name = "Algorithm"

    print("\n" + "=" * 90)
    print("TABLE 1: Full Algorithm Comparison (informed use h_ALT)")
    print("=" * 90)
    print(agg.to_string())
    agg.to_csv(os.path.join(OUT_DIR, "table1_full_comparison.csv"))
    return agg


# ─────────── TABLE 2: Heuristic strength (all 3) ───────────
def table2_heuristic_strength(df):
    informed = df[df["heuristic"].isin(["h_old", "h_B", "h_ALT"])].copy()
    # Dedupe: each (profile, heuristic) has multiple rows (one per algo) but same h_source
    agg = informed.groupby(["profile", "heuristic"]).agg(
        avg_h_source=("h_source", "mean"),
        avg_h_over_optimal=("h_over_optimal", "mean"),
    ).round(4)

    # Reorder
    agg = agg.reset_index()
    agg["heuristic"] = pd.Categorical(agg["heuristic"], ["h_old", "h_B", "h_ALT"])
    agg["profile"] = pd.Categorical(agg["profile"], ["fastest", "safest", "balanced"])
    agg = agg.sort_values(["profile", "heuristic"]).set_index(["profile", "heuristic"])

    print("\n" + "=" * 90)
    print("TABLE 2: Heuristic Strength — h_old vs h_B vs h_ALT")
    print("=" * 90)
    print(agg.to_string())
    agg.to_csv(os.path.join(OUT_DIR, "table2_heuristic_strength.csv"))
    return agg


# ─────────── TABLE 3: Node reduction ───────────
def table3_node_reduction(df):
    rows = []
    for algo in ["Greedy", "A*", "Weighted A*"]:
        for prof in ["fastest", "safest", "balanced"]:
            ho = df[(df["algorithm"] == algo) & (df["heuristic"] == "h_old") & (df["profile"] == prof)]
            hB = df[(df["algorithm"] == algo) & (df["heuristic"] == "h_B") & (df["profile"] == prof)]
            hA = df[(df["algorithm"] == algo) & (df["heuristic"] == "h_ALT") & (df["profile"] == prof)]
            if ho.empty or hA.empty:
                continue
            avg_ho = ho["nodes_expanded"].mean()
            avg_hB = hB["nodes_expanded"].mean()
            avg_hA = hA["nodes_expanded"].mean()
            rows.append({
                "algorithm": algo,
                "profile": prof,
                "h_old": round(avg_ho, 0),
                "h_B": round(avg_hB, 0),
                "h_ALT": round(avg_hA, 0),
                "h_B_reduction_%": round((1 - avg_hB / avg_ho) * 100, 1),
                "h_ALT_reduction_%": round((1 - avg_hA / avg_ho) * 100, 1),
                "h_ALT_speedup_x": round(avg_ho / avg_hA, 1) if avg_hA > 0 else 0,
            })

    t3 = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print("TABLE 3: Node Reduction — h_old baseline, h_B and h_ALT improvements")
    print("=" * 90)
    print(t3.to_string(index=False))
    t3.to_csv(os.path.join(OUT_DIR, "table3_node_reduction.csv"), index=False)
    return t3


# ─────────── TABLE 4: Per-scenario detail ───────────
def table4_per_scenario(df):
    mask = (df["heuristic"] == "h_ALT") | (df["heuristic"] == "none")
    sub = df[mask][["route", "profile", "algorithm", "path_cost",
                    "nodes_expanded", "runtime_ms", "optimality_ratio"]].copy()
    sub = sub.round(3)
    print("\n" + "=" * 90)
    print("TABLE 4: Per-Scenario Detail (informed = h_ALT)")
    print("=" * 90)
    for route in sub["route"].unique():
        print(f"\n── {route} ──")
        r = sub[sub["route"] == route].drop(columns=["route"])
        print(r.to_string(index=False))
    sub.to_csv(os.path.join(OUT_DIR, "table4_per_scenario.csv"), index=False)


# ─────────── PLOT 1: Nodes expanded (h_ALT) ───────────
def plot1_nodes_expanded(df):
    mask = (df["heuristic"] == "h_ALT") | (df["heuristic"] == "none")
    sub = df[mask].copy()

    algo_order = ["BFS", "DFS", "UCS", "IDS", "Greedy", "A*", "Weighted A*"]
    colors = ["#3498db", "#e74c3c", "#27ae60", "#9b59b6", "#e67e22", "#1abc9c", "#f39c12"]

    agg = sub.groupby("algorithm")["nodes_expanded"].mean().reindex(algo_order)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(agg.index, agg.values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Avg Nodes Expanded", fontsize=12)
    ax.set_title("Nodes Expanded per Algorithm (informed use h_ALT)", fontsize=14)
    ax.set_yscale("log")

    for bar, val in zip(bars, agg.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15,
                f"{val:,.0f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "plot1_nodes_expanded.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nSaved {path}")


# ─────────── PLOT 2: 3-way heuristic comparison ───────────
def plot2_heuristic_comparison(df):
    algos = ["Greedy", "A*", "Weighted A*"]
    profiles = ["fastest", "safest", "balanced"]
    heuristics = ["h_old", "h_B", "h_ALT"]
    h_colors = {"h_old": "#e74c3c", "h_B": "#f39c12", "h_ALT": "#1abc9c"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, prof in zip(axes, profiles):
        x = np.arange(len(algos))
        w = 0.27
        for i, h in enumerate(heuristics):
            vals = []
            for algo in algos:
                d = df[(df["algorithm"] == algo) & (df["heuristic"] == h) & (df["profile"] == prof)]
                vals.append(d["nodes_expanded"].mean() if not d.empty else 0)
            offset = (i - 1) * w
            ax.bar(x + offset, vals, w, label=h, color=h_colors[h], alpha=0.85)
            # Value labels
            for xi, v in zip(x, vals):
                if v > 0:
                    ax.text(xi + offset, v * 1.1, f"{int(v):,}",
                            ha="center", fontsize=8, rotation=0)

        ax.set_xticks(x)
        ax.set_xticklabels(algos, fontsize=10)
        ax.set_title(prof, fontsize=13)
        ax.set_ylabel("Avg Nodes Expanded" if prof == "fastest" else "")
        ax.legend(fontsize=9)
        ax.set_yscale("log")

    fig.suptitle("Nodes Expanded: h_old vs h_B vs h_ALT by Profile", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "plot2_heuristic_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ─────────── PLOT 3: Cost vs nodes scatter ───────────
def plot3_cost_vs_nodes(df):
    mask = (df["heuristic"] == "h_ALT") | (df["heuristic"] == "none")
    sub = df[mask].copy()

    algo_order = ["BFS", "DFS", "UCS", "IDS", "Greedy", "A*", "Weighted A*"]
    colors = {"BFS": "#3498db", "DFS": "#e74c3c", "UCS": "#27ae60", "IDS": "#9b59b6",
              "Greedy": "#e67e22", "A*": "#1abc9c", "Weighted A*": "#f39c12"}
    markers = {"BFS": "o", "DFS": "s", "UCS": "D", "IDS": "^",
               "Greedy": "v", "A*": "*", "Weighted A*": "P"}

    fig, ax = plt.subplots(figsize=(10, 6))
    for algo in algo_order:
        d = sub[sub["algorithm"] == algo]
        ax.scatter(d["nodes_expanded"], d["path_cost"],
                   c=colors[algo], marker=markers[algo], s=80, alpha=0.75,
                   label=algo, edgecolors="white", linewidths=0.5)

    ax.set_xlabel("Nodes Expanded (log scale)", fontsize=12)
    ax.set_ylabel("Path Cost", fontsize=12)
    ax.set_title("Cost vs Search Effort — informed algos use h_ALT", fontsize=14)
    ax.set_xscale("log")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "plot3_cost_vs_nodes.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ─────────── PLOT 4: Heuristic quality bar ───────────
def plot4_heuristic_quality(df):
    profiles = ["fastest", "safest", "balanced"]
    heuristics = ["h_old", "h_B", "h_ALT"]
    h_colors = {"h_old": "#e74c3c", "h_B": "#f39c12", "h_ALT": "#1abc9c"}

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(profiles))
    w = 0.27

    for i, h in enumerate(heuristics):
        vals = []
        for prof in profiles:
            d = df[(df["heuristic"] == h) & (df["profile"] == prof)]
            vals.append(d["h_over_optimal"].mean() * 100)
        offset = (i - 1) * w
        bars = ax.bar(x + offset, vals, w, label=h, color=h_colors[h], alpha=0.85)
        for xi, v in zip(x, vals):
            ax.text(xi + offset, v + 1, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(profiles, fontsize=11)
    ax.set_ylabel("h(source) / optimal cost  (%)", fontsize=12)
    ax.set_title("Heuristic Informedness — higher is better (closer to true cost)", fontsize=14)
    ax.legend(fontsize=10)
    ax.axhline(100, color="gray", linestyle="--", alpha=0.5, label="Perfect heuristic")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(105, ax.get_ylim()[1]))

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "plot4_heuristic_quality.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    df = load_data()
    print(f"Loaded {len(df)} rows from results.csv")

    table1_full_comparison(df)
    table2_heuristic_strength(df)
    table3_node_reduction(df)
    table4_per_scenario(df)

    plot1_nodes_expanded(df)
    plot2_heuristic_comparison(df)
    plot3_cost_vs_nodes(df)
    plot4_heuristic_quality(df)

    print("\n✓ Done. Tables: experiments/table*.csv, plots: experiments/plots/")


if __name__ == "__main__":
    main()
