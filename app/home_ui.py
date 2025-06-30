import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st

def home_page():
    st.title("CEA Database Wizard")
    st.markdown("This tool uses K-Prototypes mixed data clustering to create custom construction archetypes from urban datasets for City Energy Analyst.")

    kprotoyper_container = st.container(border=True)
    with kprotoyper_container:
        st.markdown("#### :material/smart_toy: K-Prototyper")
        st.markdown("Analyze your urban dataset using the k-prototypes algorithm to identify representative clusters, forming the basis for archetypes.")
        kp1, kp2 = st.columns([3,2])
        with kp2:
            if st.button("K-Prototyper →", type="primary", use_container_width=True):
                st.switch_page("kprototyper/kprototyper_ui.py")
    
    archetyper_container = st.container(border=True)
    with archetyper_container:
        st.markdown("#### :material/villa: Archetyper")
        st.markdown("Define construction archetypes from clusters elicited by the K-Prototyper, creating a custom CEA-compatible database for UBEM.")
        ar1, ar2 = st.columns([3,2])
        with ar2:
            if st.button("Archetyper →", type="primary", use_container_width=True):
                st.switch_page("archetyper/archetyper_ui.py")

    yourdata_container = st.container(border=True)
    with yourdata_container:
        st.markdown("#### :material/download: Your Data")
        st.markdown("View and download your data from current and previous sessions.")
        yd1, yd2 = st.columns([3,2])
        with yd2:
            if st.button("Your Data →", type="secondary", use_container_width=True):
                st.switch_page("userdata_ui.py")

home_page()