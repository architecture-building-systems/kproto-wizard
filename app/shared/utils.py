import pandas as pd
from pathlib import Path
from typing import List
from archetyper.session import ArchetyperSession

# -------------------------
# Shared Utilities
# -------------------------

def normalize_session_key(name: str) -> str:
    return name.strip().lower().replace(" ", "_")

def is_valid_session_name(name: str) -> bool:
    import re
    return bool(re.match(r"^[a-zA-Z0-9 _-]+$", name.strip()))

def get_available_regions(db_base_path: Path = Path("app/databases")) -> List[str]:
    """Returns a sorted list of available region folders for database selection."""
    if not db_base_path.exists():
        return []
    return sorted([p.name for p in db_base_path.iterdir() if p.is_dir()])

def load_database(session: ArchetyperSession, region: str, base_path: Path):
    session.region = region
    root = base_path

    session.construction_types = pd.read_csv(root / "ARCHETYPES" / "CONSTRUCTION" / "CONSTRUCTION_TYPES.csv")
    session.use_types = pd.read_csv(root / "ARCHETYPES" / "USE" / "USE_TYPES.csv")

    for key in session.envelope:
        session.envelope[key] = pd.read_csv(root / "ASSEMBLIES" / "ENVELOPE" / f"{key}.csv")
    for key in session.hvac:
        session.hvac[key] = pd.read_csv(root / "ASSEMBLIES" / "HVAC" / f"{key}.csv")
    for key in session.supply:
        session.supply[key] = pd.read_csv(root / "ASSEMBLIES" / "SUPPLY" / f"{key}.csv")
    for key in session.conversion:
        session.conversion[key] = pd.read_csv(root / "COMPONENTS" / "CONVERSION" / f"{key}.csv")
    for key in session.distribution:
        session.distribution[key] = pd.read_csv(root / "COMPONENTS" / "DISTRIBUTION" / f"{key}.csv")
    for key in session.feedstocks:
        session.feedstocks[key] = pd.read_csv(root / "COMPONENTS" / "FEEDSTOCKS" / f"{key}.csv")
