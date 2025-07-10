import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import streamlit as st
import pandas as pd

from archetyper.session import ArchetyperSession
from archetyper.logic import load_archetype_database, normalize_session_key, is_valid_session_name, export_zone_dataframe_ui


# ---------- Session State Utilities ----------

def init_archetyper_state():
    if "archetyper__sessions" not in st.session_state:
        st.session_state["archetyper__sessions"] = {}
    if "archetyper__active_key" not in st.session_state:
        st.session_state["archetyper__active_key"] = None

def create_archetyper_session(name: str, region: str, df: pd.DataFrame):
    if not name or not is_valid_session_name(name):
        st.error("Invalid session name. Only letters, numbers, spaces, underscores, and hyphens are allowed.")
        return

    key = normalize_session_key(name)

    if key in st.session_state["archetyper__sessions"]:
        st.error(f"A session named `{name}` already exists. Please choose a different name.")
        return

    session = ArchetyperSession(name, region, df)
    base_path = Path("app/databases") / region
    missing_files = load_archetype_database(session, region, base_path)

    st.session_state["archetyper__sessions"][key] = session
    st.session_state["archetyper__active_key"] = key

    st.success(f"Session `{name}` created and activated.")

    if missing_files:
        st.warning("Some database files were missing:")
        for f in missing_files:
            st.code(f)

def get_active_session():
    key = st.session_state.get("archetyper__active_key")
    return st.session_state["archetyper__sessions"].get(key) if key else None

def list_sessions():
    return list(st.session_state["archetyper__sessions"].keys())

def set_active_session(key: str):
    if key in st.session_state["archetyper__sessions"]:
        st.session_state["archetyper__active_key"] = key

def delete_session(key: str):
    if key in st.session_state["archetyper__sessions"]:
        del st.session_state["archetyper__sessions"][key]
        if st.session_state["archetyper__active_key"] == key:
            st.session_state["archetyper__active_key"] = None

def rename_session(old_key: str, new_name: str):
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


# ---------- UI HELPER ----------

def map_numeric_fields_ui(session):
    st.markdown("### 🧾 Map Numeric Building Fields")

    numeric_fields = ["floors_ag", "floors_bg", "height_ag", "height_bg", "year"]
    available_cols = list(session.clustered_df.columns)

    if not hasattr(session, "field_mapping"):
        session.field_mapping = {}

    with st.expander("📊 Assign Source Columns", expanded=True):
        for field in numeric_fields:
            default = session.field_mapping.get(field)
            selected = st.selectbox(
                f"Map `{field}` to column:",
                ["(not mapped)"] + available_cols,
                index=(available_cols.index(default) + 1) if default in available_cols else 0,
                key=f"fieldmap_{field}"
            )
            session.field_mapping[field] = selected if selected != "(not mapped)" else None

def assign_archetypes_ui(session):
    st.markdown("### 🏷️ Assign Archetypes to Clusters")

    # Ensure required data exists
    if not hasattr(session, "construction_types") or session.construction_types is None:
        st.error("Construction types table not found in session.")
        return

    if "cluster" not in session.clustered_df.columns:
        st.error("Clustered dataset does not contain a 'cluster' column.")
        return

    clusters = sorted(session.clustered_df["cluster"].unique())
    construction_df = session.construction_types

    # Build description → const_type lookup
    label_to_key = dict(zip(construction_df["description"], construction_df["const_type"]))
    key_to_label = dict(zip(construction_df["const_type"], construction_df["description"]))

    # Temporary holder for inputs (not yet saved)
    inputs = {}

    for cluster_id in clusters:
        default = None
        if cluster_id in session.archetype_map:
            const_type_key = session.archetype_map[cluster_id]
            default = key_to_label.get(const_type_key)

        selected_label = st.selectbox(
            f"Cluster {cluster_id}",
            options=list(label_to_key.keys()),
            index=(list(label_to_key.keys()).index(default) if default in label_to_key.values() else 0),
            key=f"cluster_{cluster_id}_selector"
        )
        inputs[cluster_id] = label_to_key[selected_label]

    if st.button("Save Archetype Assignments"):
        if all(v is not None for v in inputs.values()):
            session.archetype_map = inputs
            st.success("Archetype assignments saved.")
        else:
            st.error("Please assign an archetype to each cluster.")

def assign_use_types_ui(session):
    st.markdown("### 🧩 Assign Use Types to Clusters")

    if not hasattr(session, "use_types") or session.use_types is None:
        st.error("Use types table not found in session.")
        return

    clusters = sorted(session.clustered_df["cluster"].unique())
    use_type_options = sorted(session.use_types["use_type"].unique())

    if not hasattr(session, "use_type_map"):
        session.use_type_map = {}

    for cluster_id in clusters:
        st.markdown(f"**Cluster {cluster_id}**")

        # Get existing values if previously set
        existing = session.use_type_map.get(cluster_id, [])

        types = []
        for i in range(1, 4):
            col1, col2 = st.columns([3, 1])
            default_value = existing[i-1][0] if len(existing) >= i else None
            default_ratio = existing[i-1][1] if len(existing) >= i else 0.0

            with col1:
                selected_type = st.selectbox(
                    f"Use Type {i}",
                    ["(none)"] + use_type_options,
                    index=(use_type_options.index(default_value) + 1) if default_value in use_type_options else 0,
                    key=f"use_type_{cluster_id}_{i}"
                )

            with col2:
                ratio = st.number_input(
                    f"Ratio {i}",
                    min_value=0.0, max_value=1.0, step=0.05,
                    value=default_ratio,
                    key=f"use_ratio_{cluster_id}_{i}"
                )

            if selected_type != "(none)":
                types.append((selected_type, ratio))

        session.use_type_map[cluster_id] = types



# ---------- UI ----------

def show_archetyper_ui():
    init_archetyper_state()
    st.markdown("## 🧬 Archetyper")

    # ---- SESSION CREATION ----
    with st.expander("➕ Create New Session", expanded=True):
        name = st.text_input("Session name")
        
        # Get available regions from subfolders
        db_base_path = Path("app/databases")
        available_regions = sorted([p.name for p in db_base_path.iterdir() if p.is_dir()])
        region = st.selectbox("Database region", available_regions)

        source = st.radio("Clustered dataset source", ["Use KPrototyper session", "Upload manually"])
        clustered_df = None

        if source == "Use KPrototyper session":
            if "kprototyper" in st.session_state and hasattr(st.session_state.kprototyper, "clustered_df"):
                clustered_df = st.session_state.kprototyper.clustered_df
                st.success("Using clustered data from KPrototyper.")
            else:
                st.error("No KPrototyper session found.")
        else:
            uploaded_file = st.file_uploader("Upload clustered dataset (CSV or XLSX)", type=["csv", "xlsx"])
            if uploaded_file:
                clustered_df = (
                    pd.read_excel(uploaded_file) if uploaded_file.name.endswith("xlsx")
                    else pd.read_csv(uploaded_file)
                )
                st.success("Clustered dataset uploaded.")

        if st.button("Create Session"):
            if not name or not region or clustered_df is None:
                st.error("Please complete all fields and provide a valid dataset.")
            elif "cluster" not in clustered_df.columns or "name" not in clustered_df.columns:
                st.error("Dataset must contain 'cluster' and 'name' columns.")
            else:
                create_archetyper_session(name, region, clustered_df)
                st.success(f"Session `{name}` created and activated.")

    # ---- SESSION SWITCHING ----
    st.markdown("### 🔁 Switch Active Session")
    session_keys = list_sessions()
    if session_keys:
        selected_key = st.selectbox("Available sessions", session_keys)
        if st.button("Set Active Session"):
            set_active_session(selected_key)
            st.success(f"Switched to session: `{selected_key}`")
    else:
        st.info("No sessions available yet.")

    # ---- ACTIVE SESSION DETAILS ----
    active = get_active_session()
    if active:
        st.markdown("### ✅ Active Session")
        st.write(f"**Name**: {active.name}")
        st.write(f"**Region**: {active.region}")
        st.dataframe(active.clustered_df.head())
    else:
        st.warning("No active session selected.")

    # ---- ARCHETYPE ASSIGNMENT ----
    active = get_active_session()
    if active:
        assign_archetypes_ui(active)
        assign_use_types_ui(active)
        map_numeric_fields_ui(active)
        export_zone_dataframe_ui(active)

    if st.button("🔄 Reload Archetype Database"):
        if active:
            base_path = Path("app/databases") / active.region
            missing_files = load_archetype_database(active, active.region, base_path)
            if not missing_files:
                st.success("Reloaded archetype database successfully.")
            else:
                st.warning("Reloaded with missing files:")
                for f in missing_files:
                    st.code(f)

    # ---- SESSION MANAGEMENT ----
    st.markdown("### 🛠️ Manage Existing Sessions")
    with st.expander("🗂️ Delete or Rename Sessions", expanded=False):
        keys = list_sessions()
        if not keys:
            st.info("No sessions to manage.")
        else:
            selected = st.selectbox("Select session to manage", keys, key="manage_select")

            # --- Rename ---
            new_name = st.text_input("Rename session to:", key="rename_input")
            if st.button("Rename Session"):
                if not new_name.strip():
                    st.error("New name cannot be empty.")
                else:
                    rename_session(selected, new_name)

            st.markdown("---")

            # --- Delete ---
            confirm_delete = st.checkbox("Confirm deletion")
            if st.button("Delete Session", type="primary", disabled=not confirm_delete):
                delete_session(selected)
                st.success(f"Deleted session `{selected}`.")
    
    with st.expander("🧪 Debug: Active Session Contents"):
        if active:
            st.subheader("Archetype Map")
            st.json({
            "archetype_map": active.archetype_map,
            "use_type_map": active.use_type_map,
            "field_mapping": active.field_mapping,
        })

            st.subheader("Available Construction Types")
            st.dataframe(active.construction_types)

            st.subheader("Clustered Data Sample")
            st.dataframe(active.clustered_df.head())

            st.caption(f"Region: `{active.region}`")
        else:
            st.info("No active session.")

# ---------- UI CALL ----------
show_archetyper_ui()
