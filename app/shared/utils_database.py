import pandas as pd
import json
from typing import Dict

from shared.constants import CONSTRUCTION_TYPE_SCHEMA

# -------------------------
# Reference Column
# -------------------------

def merge_reference_columns(cluster_ref, override_ref) -> str:
    """
    Merges two JSON-formatted Reference strings.
    Cluster source is merged with override source, with override taking precedence.

    Returns:
        A single JSON string with combined provenance info.
    """
    def parse(ref):
        if isinstance(ref, str):
            try:
                return json.loads(ref)
            except json.JSONDecodeError:
                return {}
        elif isinstance(ref, dict):
            return ref
        return {}

    cluster_dict = parse(cluster_ref)
    override_dict = parse(override_ref)

    merged = {**cluster_dict, **override_dict}
    return json.dumps(merged)


def update_reference_column(df: pd.DataFrame, row_id, col_name: str, source: str) -> pd.DataFrame:
    """
    Adds or updates a 'Reference' JSON column entry for a given row and column.
    
    Parameters:
        df (pd.DataFrame): Target table.
        row_id (any): Index label of the row to modify.
        col_name (str): Name of the column being updated.
        source (str): One of 'clustering', 'user', or 'database'.
    
    Returns:
        pd.DataFrame: Updated DataFrame with a valid 'Reference' JSON string.
    """
    if "Reference" not in df.columns:
        df["Reference"] = pd.Series([json.dumps({}) for _ in range(len(df))], index=df.index)

    try:
        existing = json.loads(df.at[row_id, "Reference"])
    except (json.JSONDecodeError, TypeError, KeyError):
        existing = {}

    existing[col_name] = source
    df.at[row_id, "Reference"] = json.dumps(existing)
    return df


def compute_final_df(cluster_df, override_df):
    # Merge cluster and override
    final_df = override_df.copy()
    if cluster_df is not None:
        for col in cluster_df.columns:
            if col != "Reference":
                final_df[col] = cluster_df[col].combine_first(override_df[col])

        # Merge references
        merged_refs = []
        for idx in final_df.index:
            c_ref = cluster_df.at[idx, "Reference"] if idx in cluster_df.index else "{}"
            o_ref = override_df.at[idx, "Reference"] if idx in override_df.index else "{}"
            merged_refs.append(merge_reference_columns(c_ref, o_ref))
        final_df["Reference"] = merged_refs

    # Ensure all schema columns are present
    for col in CONSTRUCTION_TYPE_SCHEMA.keys():
        if col not in final_df.columns:
            final_df[col] = None

    return final_df


def highlight_cells_by_reference(df: pd.DataFrame):
    def highlight(row):
        style = {}
        try:
            ref = json.loads(row.get("Reference", "{}"))
        except:
            ref = {}
        for col in df.columns:
            if col == "Reference":
                continue
            source = ref.get(col)
            if source == "clustering":
                style[col] = "background-color: lightgray"
            elif source == "database":
                style[col] = "background-color: lightyellow"
            elif source == "user":
                style[col] = "background-color: lightblue"
            else:
                style[col] = ""
        return pd.Series(style)
    return df.style.apply(highlight, axis=1)


def get_baseline_labels(df: pd.DataFrame, label_field="description") -> dict:
    if df is None:
        return {}
    if label_field in df.columns:
        fallback = pd.Series(df.index.astype(str), index=df.index)
        return df[label_field].fillna(fallback).astype(str).to_dict()
    else:
        return df.index.astype(str).to_series(index=df.index).to_dict()


def get_validation_map_from_schema(schema_dict_by_table: dict[str, dict]) -> dict:
    """
    Constructs a validation map for use in validate_DB_override,
    extracting only fields that define a validator.

    Parameters:
    - schema_dict_by_table: dict of {table_name: {field_name: field_schema}}

    Returns:
    - dict of {table_name: {field_name: validator_function}}
    """
    validation_map = {}
    for table_name, table_schema in schema_dict_by_table.items():       
        validators = {        
            field: (props["validator"], props)  # validator and its metadata            
            for field, props in table_schema.items()
                if "validator" in props
        }        
        if validators:           
            validation_map[table_name] = validators
    return validation_map
