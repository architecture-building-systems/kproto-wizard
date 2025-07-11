import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import streamlit as st
import pandas as pd

from archetyper.session import ArchetyperSession
from archetyper.logic import (
    normalize_session_key,
    generate_zone_dataframe
)

from shared.utils_ui import (
    show_create_session_ui,
    show_switch_session_ui,
    show_manage_session_ui,
    show_reload_database_ui,
)

from shared.utils_session import (
    init_archetyper_state,
    get_active_archetyper_session
)

# ----------  UI FUNCTIONS ----------

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
    active = get_active_archetyper_session()

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
