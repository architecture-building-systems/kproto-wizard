import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st

def home_page():
    st.title("CEA Database Wizard")
    st.markdown("This tool uses K-Prototypes mixed data clustering to create custom construction archetypes from urban datasets for City Energy Analyst.")
    
    databasemaker_container = st.container(border=True)
    with databasemaker_container:
        st.markdown("#### :material/database: Database Maker")
        
        db1, db2 = st.columns([3,2])

        db1.caption("Define construction archetypes from clustering using K-Prototypes, creating a custom CEA-compatible database for UBEM.")
        if db2.button("Database Maker →", type="primary", use_container_width=True):
            st.switch_page("databasemaker/databasemaker_ui.py")

    yourdata_container = st.container(border=True)
    with yourdata_container:
        st.markdown("#### :material/folder: Your Data")
        yd1, yd2 = st.columns([3,2])
        yd1.caption("View and download your data from current and previous Database Maker sessions.")
        if yd2.button("Your Data →", type="secondary", use_container_width=True):
            st.switch_page("userdata_ui.py")

    info_container = st.container(border=True)
    with info_container:
        st.markdown("#### :material/info: About")
        ic1, ic2 = st.columns([3,2])
        ic1.caption("Learn about the City Energy Analyst and the research behind Database Maker.")
        with ic2:
            if st.button("About →", type="secondary", use_container_width=True):
                st.switch_page("about_ui.py")

home_page()