from shared.utils import (
    validate_code_ref,
    validate_numeric_range,
    validate_day_month_string
)

# --------------------
# Database Maker Steps
# --------------------

DATABASEMAKER_STEPS = [
    "Preprocessing",
    "Run Clustering", 
    "Review Clustering",
    "Review Database",
    "Download Database"
]

# --------------------
# Database Schema
# --------------------

CONSTRUCTION_TYPE_SCHEMA = {
    "const_type": {
        "type": "string"
    },
    "description": {
        "type": "string"
    },
    "year_start": {
        "type": "int",
        "validation": {
            "min": 1000,
            "max": 2100
        },
        "validator": validate_numeric_range
    },
    "year_end": {
        "type": "int",
        "validation": {
            "min": 1000,
            "max": 2100,
            "validator": validate_numeric_range
        }
    },
    "type_mass": {
        "type": "code_ref",
        "source_table": "ENVELOPE_MASS",
        "validator": validate_code_ref
    },
    "type_leak": {
        "type": "code_ref",
        "source_table": "ENVELOPE_TIGHTNESS",
        "validator": validate_code_ref
    },
    "type_win": {
        "type": "code_ref",
        "source_table": "ENVELOPE_WINDOW",
        "validator": validate_code_ref
    },
    "type_roof": {
        "type": "code_ref",
        "source_table": "ENVELOPE_ROOF",
        "validator": validate_code_ref
    },
    "type_part": {
        "type": "code_ref",
        "source_table": "ENVELOPE_WALL",
        "validator": validate_code_ref
    },
    "type_wall": {
        "type": "code_ref",
        "source_table": "ENVELOPE_WALL",
        "validator": validate_code_ref
    },
    "type_floor": {
        "type": "code_ref",
        "source_table": "ENVELOPE_FLOOR",
        "validator": validate_code_ref
    },
    "type_base": {
        "type": "code_ref",
        "source_table": "ENVELOPE_FLOOR",
        "validator": validate_code_ref
    },
    "type_shade": {
        "type": "code_ref",
        "source_table": "ENVELOPE_SHADING",
        "validator": validate_code_ref
    },
    "Es": {
        "type": "float",
        "validation": {"min": 0, "max": 1},
        "validator": validate_numeric_range
    },
    "Hs": {
        "type": "float",
        "validation": {"min": 0, "max": 1},
        "validator": validate_numeric_range
    },
    "Ns": {
        "type": "float",
        "validation": {"min": 0, "max": 1},
        "validator": validate_numeric_range
    },
    "occupied_bg": {
        "type": "bool"
    },
    "void_deck": {
        "type": "int",
        "validation": {"min": 0},
        "validator": validate_numeric_range
    },
    "wwr_north": {
        "type": "float",
        "validation": {"min": 0.0, "max": 1.0},
        "validator": validate_numeric_range
    },
    "wwr_south": {
        "type": "float",
        "validation": {"min": 0.0, "max": 1.0},
        "validator": validate_numeric_range
    },
    "wwr_east": {
        "type": "float",
        "validation": {"min": 0.0, "max": 1.0},
        "validator": validate_numeric_range
    },
    "wwr_west": {
        "type": "float",
        "validation": {"min": 0.0, "max": 1.0},
        "validator": validate_numeric_range
    },
    "hvac_type_hs": {
        "type": "code_ref",
        "source_table": "HVAC_HEATING",
        "validator": validate_code_ref
    },
    "hvac_type_cs": {
        "type": "code_ref",
        "source_table": "HVAC_COOLING",
        "validator": validate_code_ref
    },
    "hvac_type_dhw": {
        "type": "code_ref",
        "source_table": "HVAC_HOTWATER",
        "validator": validate_code_ref
    },
    "hvac_type_ctrl": {
        "type": "code_ref",
        "source_table": "HVAC_CONTROLLER",
        "validator": validate_code_ref
    },
    "hvac_type_vent": {
        "type": "code_ref",
        "source_table": "HVAC_VENTILATION",
        "validator": validate_code_ref
    },
    "hvac_heat_starts": {
        "type": "string",
        "format": "day_month",  # e.g. "16|09"
        "validator": validate_day_month_string
    },
    "hvac_heat_ends": {
        "type": "string",
        "format": "day_month",
        "validator": validate_day_month_string

    },
    "hvac_cool_starts": {
        "type": "string",
        "format": "day_month",
        "validator": validate_day_month_string
    },
    "hvac_cool_ends": {
        "type": "string",
        "format": "day_month",
        "validator": validate_day_month_string
    },
    "supply_type_hs": {
        "type": "code_ref",
        "source_table": "SUPPLY_HEATING",
        "validator": validate_code_ref
    },
    "supply_type_dhw": {
        "type": "code_ref",
        "source_table": "SUPPLY_HOTWATER",
        "validator": validate_code_ref
    },
    "supply_type_cs": {
        "type": "code_ref",
        "source_table": "SUPPLY_COOLING",
        "validator": validate_code_ref
    },
    "supply_type_el": {
        "type": "code_ref",
        "source_table": "SUPPLY_ELECTRICITY",
        "validator": validate_code_ref
    }
}
