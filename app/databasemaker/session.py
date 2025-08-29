from datetime import datetime
from typing import Optional, Dict
import pandas as pd
import json

from shared.constants import (
    DATABASEMAKER_STEPS,
    CONSTRUCTION_TYPE_SCHEMA
)
from shared.utils_database import (
    merge_reference_columns,
    update_reference_column

)

class DatabaseMakerLogger:
    def __init__(self):
        self.log_lines = []

    def log(self, message: str):
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        self.log_lines.append(f"{timestamp} {message}")

    def get_log_text(self):
        return "\n".join(self.log_lines)

    def clear(self):
        self.log_lines = []

class DatabaseMakerSession:
    def __init__(self, name: str, region: str, training_df: pd.DataFrame, database_df: dict, step_list: list[str] = DATABASEMAKER_STEPS):
        # Metadata
        self.name: str = name
        self.region: str = region
        self.created_at: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Inputs
        self.X0: pd.DataFrame = training_df  # Training set
        self.DB0: Dict[str, pd.DataFrame] = database_df  # Full baseline CEA database (raw)

        # Step 2: Feature mapping and column typing
        self.feature_map: Dict[str, str] = {}  # e.g., {"input_col": "cea_col"}
        self.column_types: Dict[str, str] = {}  # {"feature1": "numerical", ...}
        self.input_column_metadata: Optional[Dict[str, tuple]] = None
        self.column_metadata: Optional[Dict[str, tuple]] = None

        # Step 3: Clustering output
        self.clustered_df: Optional[pd.DataFrame] = None  # Training data + cluster labels
        self.selected_k: Optional[int] = None
        self.cluster_overview: Optional[pd.DataFrame] = None
        self.assignments_per_k: Optional[dict] = None
        self.cost_per_k: Optional[dict] = None
        self.silhouette_per_k: Optional[dict] = None
        self.peak_k: Optional[int] = None
        self.shoulder_k: Optional[int] = None
        self.best_k: Optional[int] = None

        # Step 4: Clustered → archetype summaries
        self.Y_count: Optional[pd.DataFrame] = None  # construction_types per cluster
        self.cluster_archetype_map: Dict[int, str] = {}  # {cluster_id: archetype_name}

        # Step 5: User-edited or postprocessed database
        self.DB_cluster: Dict[str, pd.DataFrame] = {}     # partially populated from clustering
        self.DB_override: Dict[str, pd.DataFrame] = {}   # user override
        self.DB1: Optional[Dict[str, pd.DataFrame]] = None  # Final merged DB for export

        # Final components for download
        self.download_ready: bool = False

        # UI state
        self.step_list = step_list
        self.step: str = step_list[0]
        self.step_success: Dict[str, bool] = {step: False for step in step_list}

        self.logger = DatabaseMakerLogger()
        self.clustering_running: bool = False

        # Predefined CEA table structure
        self.subtables = {
            "construction_types": None,
            "use_types": None,
            "envelope": {
                "ENVELOPE_FLOOR": None,
                "ENVELOPE_MASS": None,
                "ENVELOPE_ROOF": None,
                "ENVELOPE_SHADING": None,
                "ENVELOPE_TIGHTNESS": None,
                "ENVELOPE_WALL": None,
                "ENVELOPE_WINDOW": None,
            },
            "hvac": {
                "HVAC_CONTROLLER": None,
                "HVAC_COOLING": None,
                "HVAC_HEATING": None,
                "HVAC_HOTWATER": None,
                "HVAC_VENTILATION": None,
            },
            "supply": {
                "SUPPLY_COOLING": None,
                "SUPPLY_ELECTRICITY": None,
                "SUPPLY_HEATING": None,
                "SUPPLY_HOTWATER": None,
            },
            "conversion": {
                "ABSORPTION_CHILLERS": None,
                "BOILERS": None,
                "BORE_HOLES": None,
                "COGENERATION_PLANTS": None,
                "COOLING_TOWERS": None,
                "FUEL_CELLS": None,
                "HEAT_EXCHANGERS": None,
                "HEAT_PUMPS": None,
                "HYDRAULIC_PUMPS": None,
                "PHOTOVOLTAIC_PANELS": None,
                "PHOTOVOLTAIC_THERMAL_PANELS": None,
                "POWER_TRANSFORMERS": None,
                "SOLAR_COLLECTORS": None,
                "THERMAL_ENERGY_STORAGES": None,
                "UNITARY_AIR_CONDITIONERS": None,
                "VAPOR_COMPRESSION_CHILLERS": None,
            },
            "distribution": {
                "THERMAL_GRID": None,
            },
            "feedstocks": {
                "ENERGY_CARRIERS": None,
            }
        }

    def is_clustering_complete(self):
        return self.clustered_df is not None and self.cluster_overview is not None

    def set_clustering_output(self, clustered_df, overview_df, k, cost_dict, sil_dict, assignments=None):
        self.clustered_df = clustered_df
        self.cluster_overview = overview_df
        self.selected_k = k
        self.cost_per_k = cost_dict
        self.silhouette_per_k = sil_dict
        if assignments:
            self.assignments_per_k = assignments

    def get_assignments_for_k(self, k: int):
        if not self.assignments_per_k or k not in self.assignments_per_k:
            raise ValueError(f"No clustering results for k = {k}")
        return self.assignments_per_k[k]

    def reset_clustering(self):
        self.clustered_df = None
        self.cluster_overview = None
        self.selected_k = None
        self.cost_per_k = None
        self.silhouette_per_k = None
        self.peak_k = None
        self.shoulder_k = None
        self.logger.clear()

    def set_db_output(self, db_modified, db_unmodified, merged_db):
        self.DB_modified = db_modified
        self.DB_unmodified = db_unmodified
        self.DB1 = merged_db
        self.download_ready = True

    # -------------------------
    # Table Edits
    # -------------------------

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
        self.DB_override[table_name] = update_reference_column(self.DB_override[table_name], row_id, col_name, "user")
        self.compute_final_table(table_name)
        

    def auto_fill_from_baseline(self, table_name: str, row_id, baseline_row: pd.Series):
        if table_name not in self.DB_override:
            self.DB_override[table_name] = pd.DataFrame(columns=self.DB_cluster[table_name].columns)

        for col in self.DB_cluster[table_name].columns:
            if pd.isna(self.DB_cluster[table_name].at[row_id, col]):
                self.DB_override[table_name].at[row_id, col] = baseline_row.get(col)
                self.DB_override[table_name] = update_reference_column(self.DB_override[table_name], row_id, col, "database")

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


    # -------------------------
    # Populate self.DB_cluster with cluster results
    # -------------------------

    def populate_DB_cluster(self):
        """
        Aggregates values from clustered_df using feature_map and column_types,
        computing per-cluster means (numerical) or modes (categorical), and populates DB_cluster.
        If no mappings exist or no values are computed, initializes construction_types with correct headers and empty values.
        """
        if self.clustered_df is None:
            raise ValueError("clustered_df is not set")
        if "cluster" not in self.clustered_df.columns:
            raise ValueError("cluster labels not found in clustered_df")

        cluster_col = "cluster"
        df = self.clustered_df.copy()
        df["_cluster"] = df[cluster_col]
        cluster_ids = sorted(df["_cluster"].unique())

        # Prepare fallback empty table
        fallback_rows = []
        for cluster_id in cluster_ids:
            row_data = {
                "code": f"CT_{cluster_id:02d}",
                "Reference": json.dumps({})
            }
            for col in CONSTRUCTION_TYPE_SCHEMA.keys():
                row_data[col] = None
            fallback_rows.append(row_data)
        fallback_table = pd.DataFrame(fallback_rows).set_index("code")

        table_aggregates = {}
        has_valid_mapping = False
        self.logger.log(f"Available CEA columns in metadata: {list(self.column_metadata.keys())}")

        for input_col, cea_col in self.feature_map.items():
            if input_col not in df.columns:
                self.logger.log(f"Skipping missing input column: {input_col}")
                continue
            if cea_col not in self.column_metadata:
                self.logger.log(f"Unknown CEA target column: {cea_col}")
                continue

            table_name, _ = self.column_metadata[cea_col]
            col_type = self.column_types.get(input_col)
            self.logger.log(f"Processing input_col: {input_col}, maps to: {cea_col}, type: {col_type}")
            self.logger.log(f"Non-null values in input_col: {df[input_col].notnull().sum()}")

            if table_name not in table_aggregates:
                table_aggregates[table_name] = {}

            if col_type == "numerical":
                grouped = df.groupby("_cluster")[input_col].mean()
            elif col_type == "categorical":
                grouped = df.groupby("_cluster")[input_col].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
            else:
                self.logger.log(f"Skipping column '{input_col}' due to unknown type: {col_type}")
                continue

            for cluster_id, val in grouped.items():
                row = table_aggregates[table_name].setdefault(cluster_id, {})
                row[cea_col] = val
                row.setdefault("Reference", {})[cea_col] = "clustering"
                has_valid_mapping = True

        self.DB_cluster = {}  # Reset container

        if not has_valid_mapping:
            self.logger.log("No usable mappings or data available. Using fallback empty construction_types table.")
            self.DB_cluster["construction_types"] = fallback_table
            return

        # Build actual tables from aggregation
        for table_name, cluster_dict in table_aggregates.items():
            rows = []
            for cluster_id in sorted(cluster_dict.keys()):
                row_data = cluster_dict[cluster_id]
                reference_dict = row_data.pop("Reference", {}) if "Reference" in row_data else {}
                row_data["code"] = f"CT_{cluster_id:02d}"
                row_data["Reference"] = json.dumps(reference_dict)
                rows.append(row_data)

            table_df = pd.DataFrame(rows).set_index("code")

            # Ensure all expected schema columns are present
            for col in CONSTRUCTION_TYPE_SCHEMA.keys():
                if col not in table_df.columns:
                    table_df[col] = None

            self.DB_cluster[table_name] = table_df

        # Final fallback: ensure construction_types exists (even if no mappings to it)
        if "construction_types" not in self.DB_cluster:
            self.logger.log("No construction_types table was created — injecting fallback.")
            self.DB_cluster["construction_types"] = fallback_table



    # -------------------------
    # Table Identification and UI Setup
    # -------------------------

    def get_populated_cluster_tables(self) -> list[str]:
        """
        Returns all table names in DB_cluster that are non-empty.
        These are the tables that have been modified based on clustering and should be shown in the UI.
        """
        return [
            table_name
            for table_name, df in self.DB_cluster.items()
            if isinstance(df, pd.DataFrame) and not df.empty
        ]


    def initialize_DB_override(self):
        """
        Initializes DB_override:
        - For tables with clustering output, creates an empty DataFrame (same shape + index),
        ready to accept user edits (but not overriding clustering-derived values).
        - For all other tables, copies them from DB0.
        """
        for table in self.DB0:
            # Skip if already initialized (to allow re-entry without overwriting user input)
            if table in self.DB_override:
                continue

            if table in self.DB_cluster:
                # Create empty override table with same shape and index
                cluster_df = self.DB_cluster[table]
                empty_override = pd.DataFrame(
                    index=cluster_df.index,
                    columns=cluster_df.columns
                )

                # Ensure "Reference" column is present as empty JSON
                if "Reference" in empty_override.columns:
                    empty_override["Reference"] = [
                        "{}" for _ in range(len(empty_override))
                    ]

                self.DB_override[table] = empty_override

            else:
                # Copy unmodified table from DB0
                self.DB_override[table] = self.DB0[table].copy()


    # -------------------------
    # Editing and UI Updates
    # -------------------------

    def auto_fill_from_baseline(self, table_name: str, row_id, baseline_row: pd.Series):
        """
        For a given table and row, fills all override values from the baseline_row
        for any column that:
        - is not already populated in DB_cluster (i.e., not locked)
        - is not already manually overridden
        Tracks each filled cell in the Reference column as 'database'.
        """
        # Ensure override table is initialized
        if table_name not in self.DB_override:
            cluster_cols = self.DB_cluster.get(table_name, pd.DataFrame()).columns
            self.DB_override[table_name] = pd.DataFrame(columns=cluster_cols)

        # Ensure override row exists
        if row_id not in self.DB_override[table_name].index:
            self.DB_override[table_name].loc[row_id] = [pd.NA] * len(self.DB_override[table_name].columns)

        cluster_df = self.DB_cluster.get(table_name, pd.DataFrame())
        override_df = self.DB_override[table_name]

        for col in override_df.columns:
            if col == "Reference":
                continue

            cluster_locked = (
                col in cluster_df.columns and
                row_id in cluster_df.index and
                not pd.isna(cluster_df.at[row_id, col])
            )

            already_overridden = (
                col in override_df.columns and
                row_id in override_df.index and
                not pd.isna(override_df.at[row_id, col])
            )

            if not cluster_locked and not already_overridden:
                value = baseline_row.get(col, pd.NA)
                override_df.at[row_id, col] = value
                override_df = update_reference_column(override_df, row_id, col, "database")

        self.DB_override[table_name] = override_df
        self.logger.log(f"Auto-filled {table_name} row {row_id} from baseline template.")


    def apply_override_value(self, table: str, row, col: str, val):
        """
        Applies a user-provided value to DB_override if the cell was not set by clustering.
        Also updates the 'Reference' column to track the origin of this value as 'user'.
        """
        # Ensure override table is initialized
        if table not in self.DB_override:
            cluster_columns = self.DB_cluster.get(table, pd.DataFrame()).columns
            self.DB_override[table] = pd.DataFrame(columns=cluster_columns)

        # Prevent override of cluster-derived values
        cluster_df = self.DB_cluster.get(table)
        if cluster_df is not None:
            if (
                row in cluster_df.index
                and col in cluster_df.columns
                and not pd.isna(cluster_df.at[row, col])
            ):
                return  # Value is locked (clustering source)

        # Apply value and update reference
        self.DB_override[table].at[row, col] = val
        self.DB_override[table] = update_reference_column(
            self.DB_override[table], row, col, "user"
        )
        self.logger.log(f"User override: {table}[{row}, {col}] = {val}")


    # -------------------------
    # Validation and Export
    # -------------------------


    def validate_DB_override(self, validation_map: dict) -> dict:
        """
        Validates all values in DB_override using the provided validation_map:
        {table_name: {column_name: validation_function, ...}, ...}

        Returns:
            errors: dict of {(table, row, col): error_message} for each invalid value.
        """
        errors = {}

        for table, df in self.DB_override.items():
            rules = validation_map.get(table, {})
            for col, validator_info in rules.items():
                if col not in df.columns:
                    continue

                # Unpack (validator, meta)
                if isinstance(validator_info, tuple):
                    validator, meta = validator_info
                else:
                    validator, meta = validator_info, {}

                for idx, val in df[col].items():
                    try:
                        if validator.__code__.co_argcount == 1:
                            valid = validator(val)
                        elif validator.__code__.co_argcount == 2:
                            valid = validator(val, meta)
                        elif validator.__code__.co_argcount == 3:
                            valid = validator(val, meta, self)
                        else:
                            raise ValueError("Unsupported validator signature")

                        if not valid:
                            errors[(table, idx, col)] = f"Invalid value: {val}"
                    except Exception as e:
                        errors[(table, idx, col)] = f"Validation error: {val} ({str(e)})"

        return errors


    def compute_DB1(self):
        """
        For each table in DB_override:
        - Combine DB_cluster + DB_override to produce DB1.
        - DB_cluster values take priority (locked).
        - Merges Reference columns.
        """
        self.DB1 = {}
        for table in self.DB_override:
            override = self.DB_override[table].copy()
            cluster = self.DB_cluster.get(table)

            if cluster is not None:
                for col in cluster.columns:
                    if col == "Reference":
                        continue
                    override[col] = cluster[col].combine_first(override[col])

                # Merge reference columns
                merged_refs = []
                for idx in override.index:
                    c_ref = cluster.at[idx, "Reference"] if idx in cluster.index else "{}"
                    o_ref = override.at[idx, "Reference"] if idx in override.index else "{}"
                    merged_refs.append(merge_reference_columns(c_ref, o_ref))
                override["Reference"] = merged_refs

            self.DB1[table] = override

        self.download_ready = True



