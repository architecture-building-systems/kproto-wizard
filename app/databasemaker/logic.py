import pandas as pd
from typing import Dict

def compute_final_table(self, table_name: str):
    cluster_df = self.DB_cluster.get(table_name)
    override_df = self.DB_override.get(table_name)

    if cluster_df is None:
        raise ValueError(f"DB_cluster does not contain {table_name}")

    merged_df = cluster_df.copy()
    if override_df is not None:
        for col in merged_df.columns:
            if col in override_df.columns:
                merged_df[col] = merged_df[col].combine_first(override_df[col])

    if self.DB1 is None:
        self.DB1 = {}

    self.DB1[table_name] = merged_df

def apply_override_value(self, table_name: str, row_id, col_name: str, value):
    if table_name not in self.DB_override:
        self.DB_override[table_name] = pd.DataFrame(columns=self.DB_cluster[table_name].columns)

    self.DB_override[table_name].at[row_id, col_name] = value
    self.compute_final_table(table_name)

def auto_fill_from_baseline(self, table_name: str, row_id, baseline_row: pd.Series):
    if table_name not in self.DB_override:
        self.DB_override[table_name] = pd.DataFrame(columns=self.DB_cluster[table_name].columns)

    for col in self.DB_cluster[table_name].columns:
        if pd.isna(self.DB_cluster[table_name].at[row_id, col]):
            self.DB_override[table_name].at[row_id, col] = baseline_row.get(col)

    self.compute_final_table(table_name)

def validate_final_table(self, table_name: str, validation_map: Dict[str, callable]):
    df = self.DB1.get(table_name)
    if df is None:
        return {}

    results = {}
    for col in df.columns:
        validator = validation_map.get(col)
        if validator is None:
            continue
        for idx, val in df[col].items():
            try:
                if not validator(val):
                    results[(idx, col)] = False
            except Exception:
                results[(idx, col)] = False
    return results
