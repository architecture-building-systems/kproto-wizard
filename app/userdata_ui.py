import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[0]))

import pandas as pd
import streamlit as st

from archetyper.session import ArchetyperSession
from kprototyper.session import KPrototyperSession

def userdata_page():
    st.markdown("## :material/folder: Your Data")
    
    col1, col2 = st.columns([6, 1])
    col1.markdown("View active sessions and stored data.")
    if col2.button("", key="refresh_your_data", help="Refresh", icon=":material/refresh:", use_container_width=True):
        st.rerun()

    # --- KPrototyper Sessions ---
    st.subheader(":material/smart_toy: K-Prototyper Sessions")
    kproto_sessions = st.session_state.get("kprototyper__sessions", {})
    active_kproto_key = st.session_state.get("kprototyper__active_key", None)

    if kproto_sessions:
        for key, session in sorted(kproto_sessions.items()):
            with st.container(border=True):
                st.markdown(f"**Session Name:** `{session.name}`")
                st.markdown(f"Created: {session.created_at}")
                st.markdown(f"Status: `{session.status}`")
                if session.selected_k:
                    st.markdown(f"Selected k: `{session.selected_k}`")
                if key == active_kproto_key:
                    st.success("Active Session")
    else:
        st.info("No K-Prototyper sessions found.")

    st.divider()

    # --- Archetyper Sessions ---
    st.subheader(":material/villa: Archetyper Sessions")
    archetyper_sessions = st.session_state.get("archetyper__sessions", {})
    active_arch_key = st.session_state.get("archetyper__active_key", None)

    if archetyper_sessions:
        for key, session in sorted(archetyper_sessions.items()):
            with st.container(border=True):
                st.markdown(f"**Session Name:** `{session.name}`")
                st.markdown(f"Created: {session.created_at}")
                st.markdown(f"Region: `{session.region}`")
                st.markdown(f"Clusters: `{session.clustered_df['cluster'].nunique()}`" if session.clustered_df is not None else "`No data`")
                if key == active_arch_key:
                    st.success("Active Session")
    else:
        st.info("No Archetyper sessions found.")

    st.divider()
    if st.button(":material/refresh:", key="refresh_your_data", help="Refresh Page"):
        st.rerun()

userdata_page()