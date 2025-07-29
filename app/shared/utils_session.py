import pandas as pd
import streamlit as st

from pathlib import Path
from typing import Optional, Dict

from kprototyper.session import KPrototyperSession
from archetyper.session import ArchetyperSession

from shared.utils import (
    is_valid_session_name,
    normalize_session_key,
    load_database
)

# -------------------------
# Archetyper Session State
# -------------------------

def init_archetyper_state():
    if "archetyper__sessions" not in st.session_state:
        st.session_state["archetyper__sessions"] = {}
    if "archetyper__active_key" not in st.session_state:
        st.session_state["archetyper__active_key"] = None

def create_archetyper_session(
    name: str,
    region: str,
    df: pd.DataFrame,
    original_df: Optional[pd.DataFrame] = None,
    clustered_summary: Optional[Dict[str, pd.DataFrame]] = None
):
    if not name or not is_valid_session_name(name):
        st.error("Invalid session name. Only letters, numbers, spaces, underscores, and hyphens are allowed.")
        return

    key = normalize_session_key(name)

    if key in st.session_state.get("archetyper__sessions", {}):
        st.error(f"A session named `{name}` already exists. Please choose a different name.")
        return

    # Create and populate session
    session = ArchetyperSession(name, region, df)
    session.original_df = original_df
    session.clustered_summary = clustered_summary

    # Load archetype database
    base_path = Path("app/databases") / region
    missing_files = load_database(session, region, base_path)

    # Store session
    if "archetyper__sessions" not in st.session_state:
        st.session_state["archetyper__sessions"] = {}
    st.session_state["archetyper__sessions"][key] = session
    st.session_state["archetyper__active_key"] = key

    st.success(f"Session `{name}` created and activated.")

    if missing_files:
        st.warning("Some database files were missing:")
        for f in missing_files:
            st.code(f)

def get_active_archetyper_session():
    key = st.session_state.get("archetyper__active_key")
    return st.session_state["archetyper__sessions"].get(key) if key else None

def list_archetyper_sessions():
    return list(st.session_state["archetyper__sessions"].keys())

def set_active_archetyper_session(key: str):
    if key in st.session_state["archetyper__sessions"]:
        st.session_state["archetyper__active_key"] = key

def delete_archetyper_session(key: str):
    if key in st.session_state["archetyper__sessions"]:
        del st.session_state["archetyper__sessions"][key]
        if st.session_state["archetyper__active_key"] == key:
            st.session_state["archetyper__active_key"] = None

def rename_archetyper_session(old_key: str, new_name: str):
    sessions = st.session_state["archetyper__sessions"]
    if old_key in sessions:
        new_key = new_name.lower().replace(" ", "_")
        if new_key in sessions:
            st.error("A session with this name already exists.")
            return

        session = sessions.pop(old_key)
        session.name = new_name  # update display name
        sessions[new_key] = session

        if st.session_state["archetyper__active_key"] == old_key:
            st.session_state["archetyper__active_key"] = new_key

        st.success(f"Renamed session to `{new_name}`.")


# -------------------------
# KPrototyper Session State
# -------------------------

def init_kprototyper_state():
    st.session_state.setdefault("kprototyper__sessions", {})
    st.session_state.setdefault("kprototyper__active_key", None)

    # UI-specific session state
    st.session_state.setdefault("kproto_step", "Upload")
    st.session_state.setdefault("kproto_step_success", {
        step: False for step in ["Assign Column Types", "Run Clustering", "Review & Download"]
    })

def get_kprototyper_sessions():
    return st.session_state["kprototyper__sessions"]

def get_active_kprototyper_key():
    return st.session_state.get("kprototyper__active_key")

def get_active_kprototyper_session():
    key = get_active_kprototyper_key()
    return get_kprototyper_sessions().get(key) if key else None

def set_active_kprototyper_session(key):
    if key in get_kprototyper_sessions():
        st.session_state["kprototyper__active_key"] = key

# -------------------------
# K-Prototyper Session Migration
# -------------------------

def export_to_archetyper(session: KPrototyperSession, region: str):
    """Creates a new ArchetyperSession from a finalized KPrototyperSession."""
    if not session.is_complete():
        raise ValueError("Session is not complete and cannot be exported.")
    if session.clustered_df is None or session.cluster_overview is None:
        raise ValueError("Missing required clustering outputs.")

    create_archetyper_session(
        name=session.name,
        region=region,
        df=session.clustered_df,
        original_df=session.input_df,
        clustered_summary={"summary": session.cluster_overview}
    )

    st.toast(f"Exported session '{session.name}' to Archetyper", icon=":material/package_2:")

# -------------------------
# Database Maker Session State
# -------------------------

def init_databasemaker_state(step_list):
    st.session_state.setdefault("databasemaker__sessions", {})
    st.session_state.setdefault("databasemaker__active_key", None)

    # UI-specific session state
    st.session_state.setdefault("databasemaker__step", step_list[0])
    st.session_state.setdefault("databasemaker__step_success", {
        step: False for step in step_list
    })