import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import streamlit as st
import pandas as pd

from archetyper.session import ArchetyperSession
from archetyper.logic import load_archetype_database, normalize_session_key, is_valid_session_name, generate_zone_dataframe


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


# ----------  UI HELPER FUNCTIONS ----------

def show_create_session_ui():
    st.markdown("###### Create New Session")

    name = st.text_input("Session name")
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
    st.markdown("###### Switch Active Session")

    session_keys = list_sessions()
    if session_keys:
        labels = {k: st.session_state["archetyper__sessions"][k].name for k in session_keys}
        selected_key = st.selectbox("Choose session", session_keys, format_func=lambda k: labels[k])
        if st.button("Set Active Session", use_container_width=True):
            set_active_session(selected_key)
            st.success(f"Switched to session: `{labels[selected_key]}`")
    else:
        st.info("No sessions available.")

def show_manage_session_ui():
    st.markdown("###### Rename or Delete Sessions")

    keys = list_sessions()
    if not keys:
        st.info("No sessions to manage.")
        return

    selected = st.selectbox("Manage session", keys, key="manage_select")

    st.divider()

    new_name = st.text_input("Rename session to:", key="rename_input")

    if st.button("Rename Session", use_container_width=True):
        if not new_name.strip():
            st.error("New name cannot be empty.")
        else:
            rename_session(selected, new_name)

    st.divider()
    confirm_delete = st.checkbox("Confirm deletion")
    if st.button("Delete Session", type="primary", disabled=not confirm_delete, use_container_width=True):
        delete_session(selected)
        st.success(f"Deleted session `{selected}`.")

def show_reload_database_ui():
    st.markdown("###### Reload Archetype Database")

    active = get_active_session()
    if st.button("Reload", use_container_width=True):
        if active:
            base_path = Path("app/databases") / active.region
            missing = load_archetype_database(active, active.region, base_path)
            if missing:
                st.warning("Reloaded with missing files:")
                for f in missing:
                    st.code(f)
            else:
                st.success("Archetype database reloaded successfully.")
        else:
            st.info("No active session selected.")

def map_numeric_fields_ui(session):
    st.markdown("#### Map Numeric Building Fields")

    numeric_fields = ["floors_ag", "floors_bg", "height_ag", "height_bg", "year"]
    available_cols = list(session.clustered_df.columns)

    if not hasattr(session, "field_mapping"):
        session.field_mapping = {}

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
    st.markdown("#### Assign Archetypes to Clusters")

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
    st.markdown("#### Assign Use Types to Clusters")

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

def export_zone_dataframe_ui(session):
    st.markdown("#### Export Zone Table")

    if st.button("Generate Zone DataFrame"):
        zone_df = generate_zone_dataframe(session)

        st.success("Zone table generated.")
        st.dataframe(zone_df)

        csv = zone_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name=f"zone_{session.name}.csv",
            mime="text/csv"
        )

def show_debug_panel(session):
    st.markdown("#### Debug: Active Session State")

    st.markdown("###### Archetype Map (cluster → const_type)")
    st.json(session.archetype_map if hasattr(session, "archetype_map") else {})

    st.markdown("###### Use Type Map (cluster → [(use_type, ratio), ...])")
    st.json(session.use_type_map if hasattr(session, "use_type_map") else {})

    st.markdown("###### Field Mapping (output field → input column)")
    st.json(session.field_mapping if hasattr(session, "field_mapping") else {})

    st.markdown("###### Sample Clustered Data")
    st.dataframe(session.clustered_df.head())

    st.markdown("###### Construction Types")
    if hasattr(session, "construction_types"):
        st.dataframe(session.construction_types)
    else:
        st.warning("`construction_types` not loaded.")

    st.markdown("###### Use Types")
    if hasattr(session, "use_types"):
        st.dataframe(session.use_types)
    else:
        st.warning("`use_types` not loaded.")

    st.caption(f"Session key: `{normalize_session_key(session.name)}` | Region: `{session.region}`")


# ---------- UI ----------

def show_archetyper_ui():
    init_archetyper_state()
    active = get_active_session()

    st.markdown(f"## :material/villa: Archetyper")    

    t1, t2, t3, t4 = st.columns(4)
    with t1.popover("New", icon=":material/add:", use_container_width=True):
        show_create_session_ui()
    
    with t2.popover("Switch", icon=":material/menu_open:", use_container_width=True):
        show_switch_session_ui()
    
    with t3.popover("Manage", icon=":material/settings:", use_container_width=True):
        show_manage_session_ui()
    
    with t4.popover("Database", icon=":material/database:", use_container_width=True):
        show_reload_database_ui()


    if not active:
        st.warning("No active session selected. Please create or switch to a session.")
        return

    st.success(f"**Active Session**: {active.name} (region: {active.region})")

    st.markdown(f"#### Assign Use and Construction Archetypes")
    st.markdown(f"Populate zone table for City Energy Analyst using from clustering results and training data.")

    with st.expander(":material/foundation: Assign Construction Types to Clusters"):
        assign_archetypes_ui(active)

    with st.expander(":material/location_home: Assign Use Types to Clusters"):
        assign_use_types_ui(active)
    
    with st.expander(":material/123: Map Column Data to Numeric Fields"):
        map_numeric_fields_ui(active)
    
    export_zone_dataframe_ui(active)

    with st.expander("Debug: Session State", expanded=False):
        show_debug_panel(active)

# ---------- UI CALL ----------
show_archetyper_ui()
