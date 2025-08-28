import pandas as pd
from pathlib import Path
from typing import List, Dict
from pathlib import Path
from datetime import datetime
import io
import zipfile
import re
import tempfile
import shutil
import os

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

def infer_cea_column_type(df: pd.DataFrame, column: str) -> str:
    """Guess whether a CEA column is numerical or categorical."""
    if column not in df.columns:
        return "unknown"
    dtype = df[column].dtype
    if pd.api.types.is_numeric_dtype(dtype):
        return "numerical"
    return "categorical"

# -------------------------
# Database Load / Export
# -------------------------

def load_default_database_for_region(region: str, db_base_path: Path = Path("app/databases")) -> Dict[str, pd.DataFrame]:
    """Returns a dictionary of dataframes representing the full CEA database for the selected region."""
    db: Dict[str, pd.DataFrame] = {}

    root = db_base_path / region
    if not root.exists():
        raise FileNotFoundError(f"No database folder found for region '{region}'")

    # Load top-level archetype tables
    db["construction_types"] = pd.read_csv(root / "ARCHETYPES" / "CONSTRUCTION" / "CONSTRUCTION_TYPES.csv")
    db["use_types"] = pd.read_csv(root / "ARCHETYPES" / "USE" / "USE_TYPES.csv")

    # Load envelope tables
    for key in [
        "ENVELOPE_FLOOR", "ENVELOPE_MASS", "ENVELOPE_ROOF", "ENVELOPE_SHADING",
        "ENVELOPE_TIGHTNESS", "ENVELOPE_WALL", "ENVELOPE_WINDOW"
    ]:
        db[key] = pd.read_csv(root / "ASSEMBLIES" / "ENVELOPE" / f"{key}.csv")

    # Load HVAC tables
    for key in [
        "HVAC_CONTROLLER", "HVAC_COOLING", "HVAC_HEATING",
        "HVAC_HOTWATER", "HVAC_VENTILATION"
    ]:
        db[key] = pd.read_csv(root / "ASSEMBLIES" / "HVAC" / f"{key}.csv")

    # Load supply tables
    for key in [
        "SUPPLY_COOLING", "SUPPLY_ELECTRICITY", "SUPPLY_HEATING", "SUPPLY_HOTWATER"
    ]:
        db[key] = pd.read_csv(root / "ASSEMBLIES" / "SUPPLY" / f"{key}.csv")

    # Load conversion components
    for key in [
        "ABSORPTION_CHILLERS", "BOILERS", "BORE_HOLES", "COGENERATION_PLANTS",
        "COOLING_TOWERS", "FUEL_CELLS", "HEAT_EXCHANGERS", "HEAT_PUMPS",
        "HYDRAULIC_PUMPS", "PHOTOVOLTAIC_PANELS", "PHOTOVOLTAIC_THERMAL_PANELS",
        "POWER_TRANSFORMERS", "SOLAR_COLLECTORS", "THERMAL_ENERGY_STORAGES",
        "UNITARY_AIR_CONDITIONERS", "VAPOR_COMPRESSION_CHILLERS"
    ]:
        db[key] = pd.read_csv(root / "COMPONENTS" / "CONVERSION" / f"{key}.csv")

    # Load distribution systems
    db["THERMAL_GRID"] = pd.read_csv(root / "COMPONENTS" / "DISTRIBUTION" / "THERMAL_GRID.csv")

    # Load feedstocks
    db["ENERGY_CARRIERS"] = pd.read_csv(root / "COMPONENTS" / "FEEDSTOCKS" / "ENERGY_CARRIERS.csv")

    return db

def export_database_to_directory(db: dict[str, pd.DataFrame], output_path: Path | str) -> None:
    """
    Exports a database dictionary to the correct CEA directory layout as CSV files.
    
    Parameters:
    - db: dict mapping table names to dataframes
    - output_path: base folder to save all CSVs in CEA-style subfolders
    """
    # If output path is string
    output_path = Path(output_path)

    # Define folder mapping for each table group
    folder_map = {
        # Archetype core tables
        "construction_types": output_path / "ARCHETYPES" / "CONSTRUCTION",
        "use_types": output_path / "ARCHETYPES" / "USE",

        # Envelope
        "ENVELOPE_FLOOR": output_path / "ASSEMBLIES" / "ENVELOPE",
        "ENVELOPE_MASS": output_path / "ASSEMBLIES" / "ENVELOPE",
        "ENVELOPE_ROOF": output_path / "ASSEMBLIES" / "ENVELOPE",
        "ENVELOPE_SHADING": output_path / "ASSEMBLIES" / "ENVELOPE",
        "ENVELOPE_TIGHTNESS": output_path / "ASSEMBLIES" / "ENVELOPE",
        "ENVELOPE_WALL": output_path / "ASSEMBLIES" / "ENVELOPE",
        "ENVELOPE_WINDOW": output_path / "ASSEMBLIES" / "ENVELOPE",

        # HVAC
        "HVAC_CONTROLLER": output_path / "ASSEMBLIES" / "HVAC",
        "HVAC_COOLING": output_path / "ASSEMBLIES" / "HVAC",
        "HVAC_HEATING": output_path / "ASSEMBLIES" / "HVAC",
        "HVAC_HOTWATER": output_path / "ASSEMBLIES" / "HVAC",
        "HVAC_VENTILATION": output_path / "ASSEMBLIES" / "HVAC",

        # Supply
        "SUPPLY_COOLING": output_path / "ASSEMBLIES" / "SUPPLY",
        "SUPPLY_ELECTRICITY": output_path / "ASSEMBLIES" / "SUPPLY",
        "SUPPLY_HEATING": output_path / "ASSEMBLIES" / "SUPPLY",
        "SUPPLY_HOTWATER": output_path / "ASSEMBLIES" / "SUPPLY",

        # Conversion
        "ABSORPTION_CHILLERS": output_path / "COMPONENTS" / "CONVERSION",
        "BOILERS": output_path / "COMPONENTS" / "CONVERSION",
        "BORE_HOLES": output_path / "COMPONENTS" / "CONVERSION",
        "COGENERATION_PLANTS": output_path / "COMPONENTS" / "CONVERSION",
        "COOLING_TOWERS": output_path / "COMPONENTS" / "CONVERSION",
        "FUEL_CELLS": output_path / "COMPONENTS" / "CONVERSION",
        "HEAT_EXCHANGERS": output_path / "COMPONENTS" / "CONVERSION",
        "HEAT_PUMPS": output_path / "COMPONENTS" / "CONVERSION",
        "HYDRAULIC_PUMPS": output_path / "COMPONENTS" / "CONVERSION",
        "PHOTOVOLTAIC_PANELS": output_path / "COMPONENTS" / "CONVERSION",
        "PHOTOVOLTAIC_THERMAL_PANELS": output_path / "COMPONENTS" / "CONVERSION",
        "POWER_TRANSFORMERS": output_path / "COMPONENTS" / "CONVERSION",
        "SOLAR_COLLECTORS": output_path / "COMPONENTS" / "CONVERSION",
        "THERMAL_ENERGY_STORAGES": output_path / "COMPONENTS" / "CONVERSION",
        "UNITARY_AIR_CONDITIONERS": output_path / "COMPONENTS" / "CONVERSION",
        "VAPOR_COMPRESSION_CHILLERS": output_path / "COMPONENTS" / "CONVERSION",

        # Distribution
        "THERMAL_GRID": output_path / "COMPONENTS" / "DISTRIBUTION",

        # Feedstocks
        "ENERGY_CARRIERS": output_path / "COMPONENTS" / "FEEDSTOCKS",
    }

    for table_name, df in db.items():
        folder = folder_map.get(table_name)
        if folder:
            folder.mkdir(parents=True, exist_ok=True)
            file_path = folder / f"{table_name}.csv"
            df.to_csv(file_path, index=False)
        else:
            print(f"[export_database_to_directory] Warning: No export path defined for table '{table_name}'")

def export_database_to_zip(db1: dict) -> io.BytesIO:
    """
    Export a structured CEA-style database to a zip archive.
    All folder structure is preserved according to export_database_to_directory.

    Parameters:
    - db1: dict[str, pd.DataFrame] — final merged database (DB1)

    Returns:
    - BytesIO zip stream ready for download
    """
    zip_buffer = io.BytesIO()

    with tempfile.TemporaryDirectory() as tmpdir:
        export_database_to_directory(db1, tmpdir)  # builds correct folder structure

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(tmpdir):
                for file in files:
                    filepath = os.path.join(root, file)
                    arcname = os.path.relpath(filepath, tmpdir)  # preserve folder path
                    zipf.write(filepath, arcname)

    zip_buffer.seek(0)
    return zip_buffer

# -------------------------
# Database Value Retrieval
# -------------------------

def cast_dataframe_to_schema_types(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    df = df.copy()
    for col, meta in schema.items():
        if col not in df.columns:
            continue

        expected_type = meta.get("type")

        try:
            if expected_type == "int":
                df[col] = df[col].astype("Int64")
            elif expected_type == "float":
                df[col] = df[col].astype("float")
            elif expected_type == "bool":
                df[col] = df[col].astype("bool")
            # don't cast strings or code_ref — keep as object
        except Exception as e:
            print(f"Warning: Could not cast column {col} to {expected_type}: {e}")

    return df

def is_dropdown_field(field_name: str, schema: dict[str, dict]) -> bool:
    """
    Returns True if the field is defined as a dropdown (i.e. 'code_ref') in the schema.
    """
    meta = schema.get(field_name, {})
    return meta.get("type") == "code_ref"

def get_dropdown_options_for_field(
    session,
    field_name: str,
    schema: dict[str, dict]
) -> list[str]:
    """
    Returns dropdown options for a given field name, based on a provided schema.

    Parameters:
    - session: the current DatabaseMakerSession (or similar)
    - field_name: the column name being edited
    - schema: a field metadata dictionary (e.g., CONSTRUCTION_TYPE_SCHEMA)

    Returns:
    - List of string values to populate a dropdown (from DB0[source_table] if code_ref)
    """
    meta = schema.get(field_name)
    if not meta or meta.get("type") != "code_ref":
        return []

    source_table = meta.get("source_table")
    if not source_table:
        return []

    df = session.DB0.get(source_table)
    if df is None or "code" not in df.columns:
        return []

    return sorted(df["code"].dropna().unique())

# -------------------------
# Database Value Validation
# -------------------------

def validate_code_ref(value, meta, session) -> bool:
    """
    Validates whether a given value exists in the code list of its source table.
    """
    if not session or "source_table" not in meta:
        return False
    table_name = meta["source_table"]
    df = session.DB0.get(table_name)
    if df is None or "code" not in df.columns:
        return False
    return value in df["code"].values


def validate_numeric_range(value, meta, session=None) -> bool:
    """
    Validates a numeric value against its min/max bounds (inclusive).
    """
    v = meta.get("validation", {})
    min_val = v.get("min", float("-inf"))
    max_val = v.get("max", float("inf"))

    try:
        num = float(value)
    except Exception as e:
        print(f"[validate_numeric_range] Invalid numeric conversion for value: {value} — {e}")
        return False

    if num < min_val:
        print(f"[validate_numeric_range] {num} < min {min_val}")
        return False
    if num > max_val:
        print(f"[validate_numeric_range] {num} > max {max_val}")
        return False

    print(f"[validate_numeric_range] {num} is valid within [{min_val}, {max_val}]")
    return True


def validate_day_month_string(value, meta, session=None) -> bool:
    """
    Validates strings like '16|09' in DD|MM format, with DD ∈ [1,31] and MM ∈ [1,12].
    """
    if not isinstance(value, str):
        return False
    match = re.match(r"^(\d{1,2})\|(\d{1,2})$", value)
    if not match:
        return False
    day, month = int(match.group(1)), int(match.group(2))
    return 1 <= day <= 31 and 1 <= month <= 12


def validate_field_value(value, col, schema, session=None):
    meta = schema.get(col)
    if not meta:
        return True  # no rules, always valid

    expected_type = meta.get("type")
    if expected_type == "int":
        try:
            int_val = int(value)
            min_val, max_val = meta.get("min", -1e10), meta.get("max", 1e10)
            return min_val <= int_val <= max_val
        except:
            return False

    elif expected_type == "float":
        try:
            float_val = float(value)
            min_val, max_val = meta.get("min", -1e10), meta.get("max", 1e10)
            return min_val <= float_val <= max_val
        except:
            return False

    elif expected_type == "bool":
        return str(value).lower() in ["true", "false", "0", "1"]

    elif expected_type == "schedule":
        return isinstance(value, str) and bool(re.match(r"^\d{1,2}\|\d{1,2}$", value.strip()))

    elif expected_type == "code_ref":
        table = meta.get("source_table")
        if session and table:
            df = session.DB0.get(table)
            return df is not None and value in df["code"].values

    return True  # default fallback


# -------------------------
# Database Schema Validation
# -------------------------

def validate_dataframe_against_schema(
    df: pd.DataFrame,
    schema: dict[str, dict],
    session=None
) -> list[dict]:
    """
    Validates all rows in a table against a schema.
    Returns a list of error dicts: {"row": idx, "field": str, "message": str}
    """
    errors = []

    for i, row in df.iterrows():
        row_dict = row.to_dict()

        for field, meta in schema.items():
            value = row_dict.get(field)
            validator = meta.get("validator")

            if validator and not validator(value, meta, session):
                errors.append({
                    "row": i,
                    "field": field,
                    "message": f"Invalid value `{value}`"
                })

    return errors

def validate_full_database(
    database: dict[str, pd.DataFrame],
    schemas: dict[str, dict],
    session=None
) -> dict[str, list[dict]]:
    """
    Validates all tables with available schemas.
    Returns a dict: table_name -> list of error dicts.
    """
    all_errors = {}

    for table_name, schema in schemas.items():
        df = database.get(table_name)
        if df is not None:
            errors = validate_dataframe_against_schema(df, schema, session)
            if errors:
                all_errors[table_name] = errors

    return all_errors

