import pandas as pd
import streamlit as st
from pathlib import Path

from kprototyper.session import KPrototyperSession

from shared.utils import (
    load_database,
    get_available_regions,
    is_valid_session_name,
    normalize_session_key
)

from shared.utils_session import (
    get_kprototyper_sessions,
    get_active_kprototyper_session,
    set_active_kprototyper_session,
    create_archetyper_session,
    set_active_archetyper_session,
    get_active_archetyper_session,
    list_archetyper_sessions,
    delete_archetyper_session,
    rename_archetyper_session,
)

# -------------------------
# KPrototyper UI Helpers
# -------------------------

def show_create_kprototyper_session_ui():
    name = st.text_input("New session name", key="kp_create_name")
    uploaded_file = st.file_uploader("Upload dataset (CSV or XLSX)", type=["csv", "xlsx"], key="kp_create_upload")

    if st.button("Create KPrototyper Session", type="primary", use_container_width=True):
        if not name or not is_valid_session_name(name):
            st.error("Invalid session name.")
            return

        key = normalize_session_key(name)
        sessions = get_kprototyper_sessions()

        if key in sessions:
            st.error("Session already exists.")
            return

        if not uploaded_file:
            st.error("Please upload a dataset.")
            return

        # --- Load and validate file ---
        try:
            df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith("xlsx") else pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return

        # --- Create and configure session ---
        session = KPrototyperSession(name)
        session.input_df = df
        session.status = "data_loaded"

        # Optional: assign default column config here if not already handled in step 1
        # session.column_config = infer_column_types(df)

        # --- Store session and activate ---
        sessions[key] = session
        set_active_kprototyper_session(key)

        # --- Initialize wizard state ---
        st.session_state.kproto_step = "Assign Column Types"
        st.session_state.kproto_step_success = {step: False for step in ["Assign Column Types", "Run Clustering", "Review & Download"]}

        # --- Redirect to wizard ---
        st.rerun()

def show_switch_kprototyper_session_ui():
    sessions = get_kprototyper_sessions()
    if not sessions:
        st.info("No existing sessions.")
        return

    labels = {k: s.name for k, s in sessions.items()}
    selected_key = st.selectbox("Select active session", list(sessions.keys()), format_func=lambda k: labels[k], key="kp_switch_select")

    if st.button("Set Active Session", key="kp_switch_button", use_container_width=True):
        set_active_kprototyper_session(selected_key)
        st.success(f"Switched to session `{labels[selected_key]}`.")

def show_manage_kprototyper_session_ui():
    sessions = get_kprototyper_sessions()
    if not sessions:
        st.info("No sessions to manage.")
        return

    selected = st.selectbox("Rename or delete session:", list(sessions.keys()), key="kp_manage_select")

    st.divider()

    new_name = st.text_input("Rename session to:", key="kp_rename_input")

    if st.button("Rename Session", key="kp_rename_button", use_container_width=True):
        if not new_name.strip():
            st.error("New name cannot be empty.")
        else:
            new_key = normalize_session_key(new_name)
            if new_key in sessions:
                st.error("A session with this name already exists.")
            else:
                session = sessions.pop(selected)
                session.name = new_name
                sessions[new_key] = session
                if st.session_state["kprototyper__active_key"] == selected:
                    st.session_state["kprototyper__active_key"] = new_key
                st.success(f"Renamed session to `{new_name}`.")
    
    st.divider()
    
    kp_confirm_delete = st.checkbox("Confirm deletion", key="kp_confirm_delete")
    if st.button("Delete Session", key="kp_delete_button", type="primary", use_container_width=True, disabled=not kp_confirm_delete):
        del sessions[selected]
        if st.session_state["kprototyper__active_key"] == selected:
            st.session_state["kprototyper__active_key"] = None
        st.success(f"Deleted session `{selected}`.")

def show_debug_kprototyper_session_ui():
    session = get_active_kprototyper_session()
    if not session:
        st.info("No active session.")
        return

    st.json({
        "name": session.name,
        "status": session.status,
        "column_config": session.column_config,
        "selected_k": session.selected_k,
        "clustered_df_preview": session.clustered_df.head().to_dict() if session.clustered_df is not None else None
    })


# -------------------------
# Archetyper UI Helpers
# -------------------------

def show_create_session_ui():
    name = st.text_input("New session name")
    available_regions = get_available_regions()
    region = st.selectbox("Database region", available_regions)

    source = st.radio("Clustered dataset source", ["Use KPrototyper session", "Upload manually"])
    clustered_df = None

    if source == "Use KPrototyper session":
        kp_sessions = st.session_state.get("kprototyper__sessions", {})

        # Filter sessions that have clustered data
        available_sessions = {
            key: s for key, s in kp_sessions.items()
            if s.clustered_df is not None and "cluster" in s.clustered_df.columns and "name" in s.clustered_df.columns
        }

        if available_sessions:
            session_key = st.selectbox("Select KPrototyper session", list(available_sessions.keys()))
            selected_session = available_sessions[session_key]
            clustered_df = selected_session.clustered_df
            st.success(f"Using clustered data from session `{selected_session.name}`.")
        else:
            st.warning("No finalized KPrototyper sessions with clustered data found.")
    else:
        uploaded_file = st.file_uploader("Upload clustered dataset (CSV or XLSX)", type=["csv", "xlsx"])
        if uploaded_file:
            clustered_df = (
                pd.read_excel(uploaded_file) if uploaded_file.name.endswith("xlsx")
                else pd.read_csv(uploaded_file)
            )
            st.success("Clustered dataset uploaded.")

    if st.button("Create Session", type="primary", use_container_width=True):
        if not name or not is_valid_session_name(name):
            st.error("Invalid session name. Only letters, numbers, spaces, underscores, and hyphens are allowed.")
        else:
            key = normalize_session_key(name)
            if key in st.session_state["archetyper__sessions"]:
                st.error(f"A session named `{name}` already exists. Please choose a different name.")
            elif not region or clustered_df is None:
                st.error("Please complete all fields and provide a valid dataset.")
            elif "cluster" not in clustered_df.columns or "name" not in clustered_df.columns:
                st.error("Dataset must contain 'cluster' and 'name' columns.")
            else:
                create_archetyper_session(name, region, clustered_df)
                st.rerun()

def show_switch_session_ui():
    session_keys = list_archetyper_sessions()
    if session_keys:
        labels = {k: st.session_state["archetyper__sessions"][k].name for k in session_keys}
        selected_key = st.selectbox("Select active session", session_keys, format_func=lambda k: labels[k])
        if st.button("Set Active Session", use_container_width=True):
            set_active_archetyper_session(selected_key)
            st.success(f"Switched to session: `{labels[selected_key]}`")
    else:
        st.info("No sessions available.")

def show_manage_session_ui():
    keys = list_archetyper_sessions()
    if not keys:
        st.info("No sessions to manage.")
        return

    selected = st.selectbox("Rename or delete session:", keys, key="manage_select")

    st.divider()

    new_name = st.text_input("Rename session to:", key="rename_input")

    if st.button("Rename Session", use_container_width=True):
        if not new_name.strip():
            st.error("New name cannot be empty.")
        else:
            rename_archetyper_session(selected, new_name)

    st.divider()
    confirm_delete = st.checkbox("Confirm deletion")
    if st.button("Delete Session", type="primary", disabled=not confirm_delete, use_container_width=True):
        delete_archetyper_session(selected)
        st.success(f"Deleted session `{selected}`.")

def show_reload_database_ui():
    active = get_active_archetyper_session()
    if st.button("Refresh Database", use_container_width=True, icon=":material/refresh:"):
        if active:
            base_path = Path("app/databases") / active.region
            missing = load_database(active, active.region, base_path)
            if missing:
                st.warning("Reloaded with missing files:")
                for f in missing:
                    st.code(f)
            else:
                st.success("Archetype database reloaded successfully.")
        else:
            st.info("No active session selected.")

