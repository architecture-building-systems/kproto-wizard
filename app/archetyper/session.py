from typing import Optional, Dict
from datetime import datetime
import pandas as pd

class ArchetyperSession:
    def __init__(self, name: str, region: str, clustered_df: pd.DataFrame):

        # Initialize
        self.name: Optional[str] = name
        self.region: Optional[str] = region
        self.clustered_df: Optional[pd.DataFrame] = clustered_df
        self.archetype_map = {}  # e.g., {0: "MFH_typ1", 1: "Office_midrise"}

        # Meta
        self.created_at: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Clustered data
        self.original_df: Optional[pd.DataFrame] = None
        self.clustered_summary: Optional[Dict[str, pd.DataFrame]] = None
        self.cluster_archetype_map: Dict[int, str] = {}  # e.g., {0: "Office", 1: "Retail"}

        ## Top-level archetypes
        self.construction_types: Optional[pd.DataFrame] = None
        self.use_types: Optional[pd.DataFrame] = None

        ## Envelope assemblies
        self.envelope = {
            "ENVELOPE_FLOOR": None,
            "ENVELOPE_MASS": None,
            "ENVELOPE_ROOF": None,
            "ENVELOPE_SHADING": None,
            "ENVELOPE_TIGHTNESS": None,
            "ENVELOPE_WALL": None,
            "ENVELOPE_WINDOW": None,
        }

        ## HVAC assemblies
        self.hvac = {
            "HVAC_CONTROLLER": None,
            "HVAC_COOLING": None,
            "HVAC_HEATING": None,
            "HVAC_HOTWATER": None,
            "HVAC_VENTILATION": None,
        }

        ## Supply systems
        self.supply = {
            "SUPPLY_COOLING": None,
            "SUPPLY_ELECTRICITY": None,
            "SUPPLY_HEATING": None,
            "SUPPLY_HOTWATER": None,
        }

        ## Conversion components
        self.conversion = {
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
        }

        ## Distribution
        self.distribution = {
            "THERMAL_GRID": None,
        }

        ## Feedstocks (not deeply included for now)
        self.feedstocks = {
            "ENERGY_CARRIERS": None,
        }

    def get_all_instances(cls):
        return cls._instances

    def reset(self):
        self.__init__()
