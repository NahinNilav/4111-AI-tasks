import os
import sys
import pickle

import networkx as nx
import osmnx as ox

# Add project root to path so we can import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import BBOX

# Bounding box
NORTH = BBOX["north"]
SOUTH = BBOX["south"]
EAST = BBOX["east"]
WEST = BBOX["west"]

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def extract_graph(network_type: str) -> nx.MultiDiGraph:
    """
    Download and process a road network from OSM.

    Args:
        network_type: "drive" or "walk"

    Returns:
        Simplified NetworkX MultiDiGraph with largest connected component.
    """
    print(f"\n{'='*60}")
    print(f"Downloading {network_type} network for central Dhaka...")
    print(f"  Bounding box: N={NORTH}, S={SOUTH}, E={EAST}, W={WEST}")
    print(f"{'='*60}")

    # Download from OSM (osmnx v2: bbox is (left, bottom, right, top) = (west, south, east, north))
    # simplify=True is the default in v2, so graph is already simplified
    G = ox.graph_from_bbox(
        bbox=(WEST, SOUTH, EAST, NORTH),
        network_type=network_type,
        simplify=True,
    )
    print(f"  Simplified graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Extract largest weakly connected component
    largest_wcc = max(nx.weakly_connected_components(G), key=len)
    G = G.subgraph(largest_wcc).copy()
    print(f"  Largest connected component: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return G


def save_graph(G: nx.MultiDiGraph, filename: str):
    """Save graph as a pickle file."""
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "wb") as f:
        pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  Saved to {filepath} ({size_mb:.1f} MB)")


def main():
    for network_type, filename in [("drive", "dhaka_drive.gpickle"),
                                    ("walk", "dhaka_walk.gpickle")]:
        G = extract_graph(network_type)
        save_graph(G, filename)

        # Print summary stats
        print(f"\n  Summary for {network_type} network:")
        print(f"    Nodes: {G.number_of_nodes()}")
        print(f"    Edges: {G.number_of_edges()}")

        # Sample a node to verify attributes
        sample_node = list(G.nodes)[0]
        node_data = G.nodes[sample_node]
        print(f"    Sample node {sample_node}: lat={node_data.get('y')}, lon={node_data.get('x')}")

        # Sample an edge to verify attributes
        sample_edge = list(G.edges(data=True))[0]
        u, v, data = sample_edge
        print(f"    Sample edge ({u} → {v}):")
        print(f"      length: {data.get('length', 'N/A')} m")
        print(f"      highway: {data.get('highway', 'N/A')}")

    print("\nPhase 1 complete! Both graphs saved.")


if __name__ == "__main__":
    main()
