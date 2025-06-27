import pandas as pd
from datetime import datetime
from typing import Optional

class KPrototyperLogger:
    def __init__(self):
        self.log_lines = []

    def log(self, message: str):
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        self.log_lines.append(f"{timestamp} {message}")

    def get_log_text(self):
        return "\n".join(self.log_lines)

    def clear(self):
        self.log_lines = []

class KPrototyperSession:
    def __init__(self, name: str):
        self.name: str = name
        self.created_at: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Input
        self.input_data: Optional[pd.DataFrame] = None
        self.column_types: Optional[dict] = None  # {"feature1": "numerical", ...}
        self.column_metadata: Optional[dict] = None  # {"feature1": (dtype, n_unique), ...}

        # Output
        self.clustered_data: Optional[pd.DataFrame] = None
        self.cluster_overview: Optional[pd.DataFrame] = None

        # Meta
        self.selected_k: Optional[int] = None
        self.cost_per_k: Optional[dict] = None
        self.silhouette_per_k: Optional[dict] = None
        self.peak_k: Optional[int] = None
        self.shoulder_k: Optional[int] = None
        self.assignments_per_k: Optional[dict] = None

        self.status: str = "initialized"
        self.logger = KPrototyperLogger()

    def is_complete(self) -> bool:
        return self.clustered_data is not None and self.cluster_overview is not None

    def set_input(self, df: pd.DataFrame, column_types: dict, column_metadata: dict = None):
        self.input_data = df
        self.column_types = column_types
        self.column_metadata = column_metadata
        self.status = "ready for clustering"

    def set_output(self, clustered_df: pd.DataFrame, overview_df: pd.DataFrame, k: int,
                   cost_dict: dict = None, sil_dict: dict = None, assignments=None):
        self.clustered_data = clustered_df
        self.cluster_overview = overview_df
        self.selected_k = k
        self.cost_per_k = cost_dict
        self.silhouette_per_k = sil_dict
        self.status = "clustered"
        if assignments:
            self.assignments_per_k = assignments

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "created_at": self.created_at,
            "status": self.status,
            "selected_k": self.selected_k,
            "complete": self.is_complete(),
        }
    
    def get_assignments_for_k(self, k: int):
        """Returns the cluster assignments for a given k, if available."""
        if not self.assignments_per_k:
            raise ValueError("No clustering assignments available.")
        if k not in self.assignments_per_k:
            raise ValueError(f"No clustering results found for k = {k}")
        return self.assignments_per_k[k]
    
    def reset_clustering(self):
        self.clustered_data = None
        self.cluster_overview = None
        self.selected_k = None
        self.cost_per_k = None
        self.silhouette_per_k = None
        self.peak_k = None
        self.shoulder_k = None
        self.status = "ready for clustering"
        self.logger.clear()
