import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st

def resources_page():
    st.header("Resources")

resources_page()