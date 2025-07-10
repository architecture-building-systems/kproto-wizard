import pandas as pd
import numpy as np
import streamlit as st
import re

from pathlib import Path
from archetyper.session import ArchetyperSession


# ------ LOAD DATA ------

def load_clustered_data():
    pass

def load_archetype_database(session: ArchetyperSession, region: str, base_path: Path) -> list[str]:
    session.region = region
    root = base_path
    missing = []

    def safe_read(path):
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            missing.append(str(path))
            return pd.DataFrame()

    session.construction_types = safe_read(root / "ARCHETYPES" / "CONSTRUCTION" / "CONSTRUCTION_TYPES.csv")
    session.use_types = safe_read(root / "ARCHETYPES" / "USE" / "USE_TYPES.csv")

    for key in session.envelope:
        session.envelope[key] = safe_read(root / "ASSEMBLIES" / "ENVELOPE" / f"{key}.csv")
    for key in session.hvac:
        session.hvac[key] = safe_read(root / "ASSEMBLIES" / "HVAC" / f"{key}.csv")
    for key in session.supply:
        session.supply[key] = safe_read(root / "ASSEMBLIES" / "SUPPLY" / f"{key}.csv")
    for key in session.conversion:
        session.conversion[key] = safe_read(root / "COMPONENTS" / "CONVERSION" / f"{key}.csv")
    for key in session.distribution:
        session.distribution[key] = safe_read(root / "COMPONENTS" / "DISTRIBUTION" / f"{key}.csv")
    for key in session.feedstocks:
        session.feedstocks[key] = safe_read(root / "COMPONENTS" / "FEEDSTOCKS" / f"{key}.csv")

    return missing

# ------ VALIDATE INPUT FIELDS ------

def is_valid_session_name(name: str) -> bool:
    # Allow letters, numbers, spaces, underscores, hyphens
    return bool(re.match(r"^[a-zA-Z0-9 _-]+$", name.strip()))

def normalize_session_key(name: str) -> str:
    return name.strip().lower().replace(" ", "_")

# ------ EXPORT DATA ------

def generate_zone_dataframe(session: ArchetyperSession) -> pd.DataFrame:
    base = session.clustered_df.copy()
    zone_data = pd.DataFrame()

    # --- Required ID and cluster
    zone_data["name"] = base["name"]
    zone_data["cluster"] = base["cluster"]  # used only for mapping, dropped later

    # --- Numeric fields
    numeric_fields = ["floors_ag", "floors_bg", "height_ag", "height_bg", "year"]
    for field in numeric_fields:
        source_col = session.field_mapping.get(field)
        if source_col and source_col in base.columns:
            zone_data[field] = base[source_col]
        else:
            zone_data[field] = None  # leave blank (can be filled in later)

    # --- Construction type
    if hasattr(session, "archetype_map"):
        zone_data["const_type"] = zone_data["cluster"].map(session.archetype_map)
    else:
        zone_data["const_type"] = None

    # --- Use types and ratios
    for i in range(1, 4):
        zone_data[f"use_type{i}"] = None
        zone_data[f"use_type{i}r"] = 0.0

    if hasattr(session, "use_type_map"):
        for idx, row in zone_data.iterrows():
            cluster = row["cluster"]
            use_def = session.use_type_map.get(cluster, [])
            for i, (utype, ratio) in enumerate(use_def[:3]):
                zone_data.at[idx, f"use_type{i+1}"] = utype
                zone_data.at[idx, f"use_type{i+1}r"] = float(ratio)

    # --- Final cleanup
    zone_data.drop(columns=["cluster"], inplace=True)

    return zone_data