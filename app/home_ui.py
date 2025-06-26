import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st

def home_page():
    st.title("CEA Database Wizard")
    st.markdown("""
    This tool supports two workflows:

    - **K-Prototyper**: Cluster your dataset using the k-prototypes algorithm.
    - **Archetyper**: Create archetypes from clustered outputs.
    
    Use the sidebar to get started.
    """)


home_page()