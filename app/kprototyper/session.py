import pandas as pd
from datetime import datetime
from typing import Optional

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

        self.status: str = "initialized"

    def is_complete(self) -> bool:
        return self.clustered_data is not None and self.cluster_overview is not None

    def set_input(self, df: pd.DataFrame, column_types: dict, column_metadata: dict = None):
        self.input_data = df
        self.column_types = column_types
        self.column_metadata = column_metadata
        self.status = "ready for clustering"

    def set_output(self, clustered_df: pd.DataFrame, overview_df: pd.DataFrame, k: int,
                   cost_dict: dict = None, sil_dict: dict = None):
        self.clustered_data = clustered_df
        self.cluster_overview = overview_df
        self.selected_k = k
        self.cost_per_k = cost_dict
        self.silhouette_per_k = sil_dict
        self.status = "clustered"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "created_at": self.created_at,
            "status": self.status,
            "selected_k": self.selected_k,
            "complete": self.is_complete(),
        }
