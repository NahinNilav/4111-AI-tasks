
# Central Dhaka
BBOX = {
    "north": 23.81,
    "south": 23.72,
    "east": 90.42,
    "west": 90.36,
}

RANDOM_SEED = 42

SD_PAIRS = [
    {
        "name": "Dhanmondi-27 → Gulshan-1",
        "source": (23.7461, 90.3742),
        "destination": (23.7808, 90.4133),
        "character": "Medium, cross-city diagonal",
    },
    {
        "name": "Banani → Farmgate",
        "source": (23.7937, 90.4030),
        "destination": (23.7573, 90.3870),
        "character": "North-south, through Tejgaon",
    },
    {
        "name": "Mohammadpur → Ramna",
        "source": (23.7662, 90.3588),
        "destination": (23.7400, 90.4050),
        "character": "West-east, through core",
    },
    {
        "name": "Shahbagh → Karwan Bazar",
        "source": (23.7375, 90.3958),
        "destination": (23.7510, 90.3930),
        "character": "Short, nearby neighborhoods",
    },
    {
        "name": "Mohakhali → Dhanmondi-15",
        "source": (23.7780, 90.4050),
        "destination": (23.7530, 90.3760),
        "character": "Medium, diagonal",
    },
]

# Weights
PROFILES = {
    "fastest": {
        "road_length": 0.35,
        "traffic_jam": 0.35,
        "road_condition": 0.10,
        "women_safety": 0.02,
        "police_availability": 0.02,
        "snatching_risk": 0.02,
        "flood_risk": 0.05,
        "lighting": 0.02,
        "footpath": 0.02,
        "noise": 0.05,
    },
    "safest": {
        "road_length": 0.05,
        "traffic_jam": 0.05,
        "road_condition": 0.05,
        "women_safety": 0.20,
        "police_availability": 0.15,
        "snatching_risk": 0.20,
        "flood_risk": 0.10,
        "lighting": 0.15,
        "footpath": 0.02,
        "noise": 0.03,
    },
    "balanced": {
        "road_length": 0.10,
        "traffic_jam": 0.10,
        "road_condition": 0.10,
        "women_safety": 0.10,
        "police_availability": 0.10,
        "snatching_risk": 0.10,
        "flood_risk": 0.10,
        "lighting": 0.10,
        "footpath": 0.10,
        "noise": 0.10,
    },
}

# Weighted A* epsilon value
WEIGHTED_ASTAR_W = 1.5

DRIVE_GRAPH_PATH = "data/dhaka_drive.gpickle"
WALK_GRAPH_PATH = "data/dhaka_walk.gpickle"
