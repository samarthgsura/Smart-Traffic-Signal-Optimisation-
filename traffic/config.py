"""Shared constants for detection, optimisation and simulation."""

# Passenger Car Units (PCU), approximately following Indian practice (IRC:106).
# A bus takes about as much road and green time as three cars; a two-wheeler about half a car.
PCU = {
    "bicycle": 0.5,
    "motorcycle": 0.5,
    "car": 1.0,
    "bus": 3.0,
    "truck": 3.0,
    "ambulance": 1.0,
}

# COCO class ids (what YOLO detects) -> our vehicle kinds.
COCO_VEHICLES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Typical Indian urban traffic mix used by the simulator (shares sum to 1).
DEFAULT_MIX = {
    "motorcycle": 0.50,
    "car": 0.30,
    "bicycle": 0.10,
    "bus": 0.05,
    "truck": 0.05,
}

APPROACHES = ("North", "East", "South", "West")

# Signal timing safety limits (seconds).
MIN_GREEN = 10      # also long enough for pedestrians to start crossing
MAX_GREEN = 60
INTERGREEN = 5      # 3 s amber + 2 s all-red between phases
STARTUP_LOST = 2    # time for the first vehicle to react when the light turns green
MAX_RED = 120       # no approach waits longer than this for a green

# Saturation flow per approach: PCU that can cross per second of green
# (about 2 lanes x 1,600 PCU/hour).
SATURATION_PCU_PER_S = 0.9

# Rough idling figures for the fuel / CO2 estimate.
IDLE_FUEL_L_PER_H = {"motorcycle": 0.15, "bicycle": 0.0, "car": 0.6,
                     "bus": 2.0, "truck": 2.0, "ambulance": 0.6}
CO2_KG_PER_L = 2.4
