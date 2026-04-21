import os, sys, pickle, random
from math import radians, sin, cos, sqrt, atan2

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import RANDOM_SEED

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def in_circle(lat, lon, clat, clon, radius_m):
    return haversine(lat, lon, clat, clon) <= radius_m


def in_bbox(lat, lon, lat_min, lat_max, lon_min, lon_max):
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max



CONGESTION_ZONES = [
    # (lat, lng, radius_m, severity)
    (23.7573, 90.3870, 400, 0.95),   # Farmgate
    (23.7780, 90.4050, 500, 0.85),   # Mohakhali
    (23.7260, 90.4120, 400, 0.90),   # Gulistan
    (23.8070, 90.3685, 450, 0.85),   # Mirpur-10
    (23.8740, 90.3990, 350, 0.75),   # Uttara Jasimuddin
    (23.7100, 90.4310, 400, 0.80),   # Jatrabari
    (23.7375, 90.3958, 300, 0.80),   # Shahbagh
    (23.7080, 90.4070, 350, 0.85),   # Sadarghat
]

THANA_LOCATIONS = [
    (23.7465, 90.3730), (23.7815, 90.4155), (23.7935, 90.4020),
    (23.7660, 90.3560), (23.7630, 90.3930), (23.7380, 90.4020),
    (23.7390, 90.3960), (23.7340, 90.4120), (23.7280, 90.4170),
    (23.7190, 90.3890), (23.7310, 90.3720), (23.8060, 90.3680),
    (23.8680, 90.3910), (23.7870, 90.3830), (23.7800, 90.4260),
]

SNATCHING_HOTSPOTS = [
    (23.765, 90.355, 400),   # Mohammadpur inner
    (23.758, 90.425, 300),   # Rampura link road
    (23.800, 90.365, 400),   # Mirpur-1/2
    (23.712, 90.428, 300),   # Jatrabari back lanes
]

FLOOD_ZONES = [
    # (lat, lng, radius_m, severity)
    (23.8080, 90.3650, 600, 0.90),   # Mirpur-10/12
    (23.7490, 90.4100, 400, 0.85),   # Mogbazar/Wireless
    (23.7550, 90.3650, 500, 0.80),   # Dhanmondi-15/Kalyanpur
    (23.7590, 90.4300, 500, 0.85),   # Rampura/Banasree
    (23.7100, 90.4310, 450, 0.85),   # Jatrabari
    (23.7800, 90.4250, 400, 0.80),   # Badda/Merul Badda
    (23.7360, 90.4130, 350, 0.75),   # Shantinagar/Rajarbagh
    (23.7620, 90.3500, 400, 0.80),   # Mohammadpur/Adabor
]

NOISE_BUS_TERMINALS = [(23.778, 90.406), (23.779, 90.345), (23.716, 90.423)]
NOISE_RAIL = [(23.732, 90.418), (23.764, 90.393)]

# Well-lit corridors (as bounding boxes: lat_min, lat_max, lon_min, lon_max)
WELL_LIT_CORRIDORS = [
    (23.745, 23.755, 90.365, 90.380),   # Mirpur Road
    (23.775, 23.795, 90.405, 90.420),   # Gulshan Avenue
    (23.740, 23.760, 90.370, 90.390),   # Dhanmondi main roads
    (23.752, 90.762, 90.383, 90.393),   # Manik Mia Avenue
]

# Dark zones
OLD_DHAKA_BBOX = (23.70, 23.7350, 90.3900, 90.4250)

# Footpath area overrides (bbox: lat_min, lat_max, lon_min, lon_max, value)
FOOTPATH_OVERRIDES = [
    (23.775, 23.800, 90.400, 90.430, 0.6),   # Gulshan/Banani/Baridhara
    (23.740, 23.760, 90.370, 90.390, 0.7),   # Dhanmondi numbered roads
    (23.70, 23.7350, 90.3900, 90.4250, 0.05), # Old Dhaka
    (23.795, 23.815, 90.355, 90.380, 0.2),   # Mirpur interior
]

QUIET_ZONES = [
    (23.775, 23.800, 90.400, 90.430),   # Gulshan/Baridhara
    (23.740, 23.760, 90.370, 90.390),   # Dhanmondi lakeside
]



def classify_highway(hw):
    """Map OSM highway tag to base category."""
    if isinstance(hw, list):
        hw = hw[0]
    hw = str(hw).lower()
    if hw in ("trunk", "trunk_link", "primary", "primary_link"):
        return "trunk_primary"
    elif hw in ("secondary", "secondary_link"):
        return "secondary"
    elif hw in ("tertiary", "tertiary_link"):
        return "tertiary"
    elif hw in ("residential", "living_street", "unclassified"):
        return "residential"
    else:
        return "residential"  # footway, path, pedestrian, etc. → treat as residential



def gen_traffic_jam(hw_class, lat, lon, rng):
    bases = {"trunk_primary": 0.8, "secondary": 0.6, "tertiary": 0.4, "residential": 0.2}
    val = bases[hw_class]
    for clat, clon, rad, severity in CONGESTION_ZONES:
        if in_circle(lat, lon, clat, clon, rad):
            val += severity * 0.15
            break
    val += rng.uniform(-0.1, 0.1)
    return max(0.0, min(1.0, val))


def gen_road_condition(hw_class, lat, lon, rng):
    bases = {"trunk_primary": 0.2, "secondary": 0.4, "tertiary": 0.6, "residential": 0.8}
    val = bases[hw_class]
    # Old Dhaka
    if in_bbox(lat, lon, *OLD_DHAKA_BBOX):
        # Sub-zones with extra degradation
        if in_circle(lat, lon, 23.7230, 90.4080, 200) or in_circle(lat, lon, 23.7190, 90.4030, 200):
            val += 0.20
        else:
            val += 0.15
    val += rng.uniform(-0.1, 0.1)
    return max(0.0, min(1.0, val))


def gen_police_availability(hw_class, lat, lon, rng):
    bases = {"trunk_primary": 0.8, "secondary": 0.5, "tertiary": 0.3, "residential": 0.1}
    avail = bases[hw_class]
    for tlat, tlon in THANA_LOCATIONS:
        dist = haversine(lat, lon, tlat, tlon)
        if dist <= 200 and hw_class == "trunk_primary":
            avail += 0.3
            break
        elif dist <= 500:
            avail += 0.2
            break
    avail += rng.uniform(-0.1, 0.1)
    avail = max(0.0, min(1.0, avail))
    return 1.0 - avail  # cost: less police = higher


def gen_lighting(hw_class, lat, lon, rng, flood_zones_cache):
    bases = {"trunk_primary": 0.9, "secondary": 0.6, "tertiary": 0.4, "residential": 0.2}
    lit = bases[hw_class]
    # Well-lit corridors
    for lat_min, lat_max, lon_min, lon_max in WELL_LIT_CORRIDORS:
        if in_bbox(lat, lon, lat_min, lat_max, lon_min, lon_max):
            lit += 0.2
            break
    # Dark zones
    if in_bbox(lat, lon, *OLD_DHAKA_BBOX):
        lit -= 0.2
    elif in_bbox(lat, lon, 23.760, 23.775, 90.350, 90.365):  # Mohammadpur inner
        lit -= 0.2
    elif in_bbox(lat, lon, 23.775, 23.790, 90.420, 90.435):  # Badda interior
        lit -= 0.2
    # Residential in flood zones = dark
    if hw_class == "residential":
        for clat, clon, rad, _ in FLOOD_ZONES:
            if in_circle(lat, lon, clat, clon, rad):
                lit -= 0.2
                break
    lit += rng.uniform(-0.1, 0.1)
    lit = max(0.0, min(1.0, lit))
    return 1.0 - lit  # cost: darker = higher


def gen_snatching_risk(police_cost, lighting_cost, hw_class, length, lat, lon, rng):
    police_avail = 1.0 - police_cost
    lighting_val = 1.0 - lighting_cost
    # isolation factor
    if hw_class == "residential":
        isolation = 0.8 if length < 100 else 0.5
    elif hw_class == "secondary":
        isolation = 0.3
    else:
        isolation = 0.1
    risk = 0.5 * (1 - police_avail) + 0.3 * (1 - lighting_val) + 0.2 * isolation
    # Hotspot zones
    for hlat, hlon, hrad in SNATCHING_HOTSPOTS:
        if in_circle(lat, lon, hlat, hlon, hrad):
            risk += 0.15
            break
    risk += rng.uniform(-0.1, 0.1)
    return max(0.0, min(1.0, risk))


def gen_women_safety(snatching_risk, police_cost, lighting_cost):
    police_avail = 1.0 - police_cost
    lighting_val = 1.0 - lighting_cost
    score = 0.4 * (1 - snatching_risk) + 0.3 * police_avail + 0.3 * lighting_val
    return max(0.0, min(1.0, 1.0 - score))  # cost


def gen_flood_risk(hw_class, lat, lon, rng):
    risk = 0.1  # base
    for clat, clon, rad, severity in FLOOD_ZONES:
        if in_circle(lat, lon, clat, clon, rad):
            risk = severity
            break
    if hw_class == "residential":
        risk += 0.1
    risk += rng.uniform(-0.1, 0.1)
    return max(0.0, min(1.0, risk))


def gen_footpath(hw_class, lat, lon, rng):
    bases = {"trunk_primary": 0.3, "secondary": 0.6, "tertiary": 0.5, "residential": 0.2}
    score = bases[hw_class]
    for lat_min, lat_max, lon_min, lon_max, override in FOOTPATH_OVERRIDES:
        if in_bbox(lat, lon, lat_min, lat_max, lon_min, lon_max):
            if override <= 0.05:  # Old Dhaka overrides all classes
                score = override
            elif hw_class == "residential":
                score = override
            break
    score += rng.uniform(-0.05, 0.05)
    score = max(0.0, min(1.0, score))
    return 1.0 - score  # cost


def gen_noise(hw_class, lat, lon, rng):
    bases = {"trunk_primary": 0.9, "secondary": 0.6, "tertiary": 0.4, "residential": 0.2}
    val = bases[hw_class]
    for blat, blon in NOISE_BUS_TERMINALS:
        if in_circle(lat, lon, blat, blon, 300):
            val += 0.15
            break
    for rlat, rlon in NOISE_RAIL:
        if in_circle(lat, lon, rlat, rlon, 200):
            val += 0.1
            break
    for lat_min, lat_max, lon_min, lon_max in QUIET_ZONES:
        if in_bbox(lat, lon, lat_min, lat_max, lon_min, lon_max) and hw_class == "residential":
            val -= 0.1
            break
    val += rng.uniform(-0.1, 0.1)
    return max(0.0, min(1.0, val))






# Main processing

def edge_centroid(G, u, v):
    lat = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2
    lon = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2
    return lat, lon


def generate_all_weights(G, label="graph"):
    rng = random.Random(RANDOM_SEED)

    # Road length: normalize by max
    max_len = max(d.get("length", 1) for _, _, d in G.edges(data=True))
    G.graph["max_edge_length"] = max_len

    metrics = ["road_length", "traffic_jam", "road_condition", "women_safety",
               "police_availability", "snatching_risk", "flood_risk",
               "lighting", "footpath", "noise"]

    count = 0
    total = G.number_of_edges()
    for u, v, key, data in G.edges(keys=True, data=True):
        lat, lon = edge_centroid(G, u, v)
        hw_class = classify_highway(data.get("highway", "residential"))
        length = data.get("length", 50)

        # 1. Road length (normalized)
        data["road_length"] = length / max_len

        # 2. Traffic jam
        data["traffic_jam"] = gen_traffic_jam(hw_class, lat, lon, rng)

        # 3. Road condition
        data["road_condition"] = gen_road_condition(hw_class, lat, lon, rng)

        # 5. Police availability (cost) — generate before snatching/women_safety
        data["police_availability"] = gen_police_availability(hw_class, lat, lon, rng)

        # 8. Lighting (cost) — generate before snatching/women_safety
        data["lighting"] = gen_lighting(hw_class, lat, lon, rng, FLOOD_ZONES)

        # 6. Snatching risk
        data["snatching_risk"] = gen_snatching_risk(
            data["police_availability"], data["lighting"],
            hw_class, length, lat, lon, rng)

        # 4. Women safety (cost)
        data["women_safety"] = gen_women_safety(
            data["snatching_risk"], data["police_availability"], data["lighting"])

        # 7. Flood risk
        data["flood_risk"] = gen_flood_risk(hw_class, lat, lon, rng)

        # 9. Footpath (cost)
        data["footpath"] = gen_footpath(hw_class, lat, lon, rng)

        # 10. Noise
        data["noise"] = gen_noise(hw_class, lat, lon, rng)

        count += 1
        if count % 5000 == 0:
            print(f"  [{label}] Processed {count}/{total} edges...")

    # Min-max normalization across all edges for each metric
    for metric in metrics:
        vals = [d[metric] for _, _, d in G.edges(data=True)]
        mn, mx = min(vals), max(vals)
        if mx - mn > 1e-9:
            for _, _, d in G.edges(data=True):
                d[metric] = (d[metric] - mn) / (mx - mn)
        print(f"  [{label}] {metric}: min={mn:.3f} max={mx:.3f}")

    return G


def main():
    for name in ["dhaka_drive", "dhaka_walk"]:
        path = os.path.join(DATA_DIR, f"{name}.gpickle")
        print(f"\nLoading {path}...")
        with open(path, "rb") as f:
            G = pickle.load(f)
        print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        G = generate_all_weights(G, label=name)

        # Save back
        with open(path, "wb") as f:
            pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
        print(f"  Saved enriched graph to {path}")

        # Verify sample edge
        u, v, data = list(G.edges(data=True))[0]
        print(f"  Sample edge ({u} → {v}):")
        for m in ["road_length", "traffic_jam", "road_condition", "women_safety",
                   "police_availability", "snatching_risk", "flood_risk",
                   "lighting", "footpath", "noise"]:
            print(f"    {m}: {data[m]:.4f}")

    print("\nPhase 2 complete!")

if __name__ == "__main__":
    main()
