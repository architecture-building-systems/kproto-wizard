from datetime import datetime
from typing import Optional, Dict
import pandas as pd

from shared.constants import DATABASEMAKER_STEPS

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

        # Step 4: Clustered → archetype summaries
        self.Y_count: Optional[pd.DataFrame] = None  # construction_types per cluster
        self.cluster_archetype_map: Dict[int, str] = {}  # {cluster_id: archetype_name}

        # Step 5: User-edited or postprocessed database
        self.DB_modified: Dict[str, pd.DataFrame] = {}     # construction_types + subcomponents (edited)
        self.DB_unmodified: Dict[str, pd.DataFrame] = {}   # all passed-through tables
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
