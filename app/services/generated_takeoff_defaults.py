"""Auto-generated from the backend canonical registry. Do not edit manually."""

TAKEOFF_HEURISTICS_VERSION = "2026-04-27-audit-revert"
TAKEOFF_HEURISTICS_SOURCE = "canonical_registry_generated_ai_takeoff_defaults"
CANONICAL_SOURCE = "backend_registry"

AI_TAKEOFF_DEFAULTS = {
    "wall_height_standard_m": float("2.40"),
    "wall_height_loft_m": float("2.30"),
    "opening_deduction_factor": float("0.85"),
    "skirting_deduction_factor": float("0.82"),
    "wet_room_wall_tiling_factor": float("0.55"),
    "architrave_lm_per_door_set": float("5.40"),
}

STANDARD_WALL_HEIGHT_M = AI_TAKEOFF_DEFAULTS["wall_height_standard_m"]
LOFT_WALL_HEIGHT_M = AI_TAKEOFF_DEFAULTS["wall_height_loft_m"]
OPENING_DEDUCTION_FACTOR = AI_TAKEOFF_DEFAULTS["opening_deduction_factor"]
SKIRTING_DEDUCTION_FACTOR = AI_TAKEOFF_DEFAULTS["skirting_deduction_factor"]
WET_ROOM_TILING_FACTOR = AI_TAKEOFF_DEFAULTS["wet_room_wall_tiling_factor"]
ARCHITRAVE_LM_PER_DOOR_SET = AI_TAKEOFF_DEFAULTS["architrave_lm_per_door_set"]

LEVEL_WALL_HEIGHTS_M = {
    "basement": STANDARD_WALL_HEIGHT_M,
    "lower ground floor": STANDARD_WALL_HEIGHT_M,
    "ground floor": STANDARD_WALL_HEIGHT_M,
    "first floor": STANDARD_WALL_HEIGHT_M,
    "second floor": STANDARD_WALL_HEIGHT_M,
    "loft": LOFT_WALL_HEIGHT_M,
}
