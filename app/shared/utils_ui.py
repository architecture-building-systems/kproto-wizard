import pandas as pd
import streamlit as st
from pathlib import Path
import json

from kprototyper.session import KPrototyperSession

from shared.utils import (
    load_database,
    load_default_database_for_region,
    get_available_regions,
    is_valid_session_name,
    normalize_session_key,
    get_dropdown_options_for_field,
    validate_field_value,
    cast_dataframe_to_schema_types
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
    rename_archetyper_session
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


# -------------------------
# Database Maker UI Helpers
# -------------------------

from databasemaker.session import DatabaseMakerSession
from shared.utils import get_available_regions, normalize_session_key, is_valid_session_name
import pandas as pd
import streamlit as st

def show_create_databasemaker_session_ui(step_list: list[str]):
    name = st.text_input("New session name", key="dbm_create_name")
    available_regions = get_available_regions()
    region = st.selectbox("Select baseline database region", available_regions, key="dbm_create_region")

    uploaded_file = st.file_uploader("Upload training dataset (CSV or XLSX)", type=["csv", "xlsx"], key="dbm_create_upload")

    if st.button("Create DatabaseMaker Session", type="primary", use_container_width=True):
        # --- Validation ---
        if not name or not is_valid_session_name(name):
            st.error("Invalid session name. Only letters, numbers, spaces, underscores, and hyphens are allowed.")
            return

        key = normalize_session_key(name)
        sessions = st.session_state.setdefault("databasemaker__sessions", {})

        if key in sessions:
            st.error(f"A session named `{name}` already exists. Please choose a different name.")
            return

        if not region or not uploaded_file:
            st.error("Please select a region and upload a training dataset.")
            return

        # --- Load training set ---
        try:
            training_df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith("xlsx") else pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading dataset: {e}")
            return

        # --- Load baseline DB from selected region ---
        try:
            base_database = load_default_database_for_region(region)  # returns a dict of dataframes
        except Exception as e:
            st.error(f"Error loading database for region '{region}': {e}")
            return

        # --- Create and store session ---
        session = DatabaseMakerSession(name, region, training_df, base_database)
        sessions[key] = session
        st.session_state["databasemaker__active_key"] = key

        # --- Initialize wizard state ---
        st.session_state["databasemaker__step"] = step_list[0]
        st.session_state["databasemaker__step_success"] = {step: False for step in step_list}

        st.rerun()

def show_switch_databasemaker_session_ui():
    sessions = st.session_state.get("databasemaker__sessions", {})
    if not sessions:
        st.info("No existing sessions.")
        return

    labels = {k: s.name for k, s in sessions.items()}
    selected_key = st.selectbox(
        "Select active session",
        options=list(sessions.keys()),
        format_func=lambda k: labels[k],
        key="dbm_switch_select"
    )

    if st.button("Set Active Session", key="dbm_switch_button", use_container_width=True):
        st.session_state["databasemaker__active_key"] = selected_key
        st.success(f"Switched to session `{labels[selected_key]}`.")
        st.rerun()

def show_manage_databasemaker_session_ui():
    sessions = st.session_state.get("databasemaker__sessions", {})
    if not sessions:
        st.info("No sessions to manage.")
        return

    selected = st.selectbox("Rename or delete session:", list(sessions.keys()), key="dbm_manage_select")

    st.divider()

    new_name = st.text_input("Rename session to:", key="dbm_rename_input")

    if st.button("Rename Session", key="dbm_rename_button", use_container_width=True):
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
                if st.session_state["databasemaker__active_key"] == selected:
                    st.session_state["databasemaker__active_key"] = new_key
                st.success(f"Renamed session to `{new_name}`.")
                st.rerun()

    st.divider()

    confirm_delete = st.checkbox("Confirm deletion", key="dbm_confirm_delete")
    if st.button("Delete Session", key="dbm_delete_button", type="primary", use_container_width=True, disabled=not confirm_delete):
        del sessions[selected]
        if st.session_state["databasemaker__active_key"] == selected:
            st.session_state["databasemaker__active_key"] = None
        st.success(f"Deleted session `{selected}`.")
        st.rerun()

def show_debug_databasemaker_session_ui():
    sessions = st.session_state.get("databasemaker__sessions", {})
    active_key = st.session_state.get("databasemaker__active_key")

    if not active_key or active_key not in sessions:
        st.info("No active session.")
        return

    session = sessions[active_key]

    st.json({
        "name": session.name,
        "region": session.region,
        "step": session.step,
        "step_success": session.step_success,
        "column_types": session.column_types,
        "feature_map": session.feature_map,
        "selected_k": session.selected_k,
        "clustered_df_preview": session.clustered_df.head().to_dict() if session.clustered_df is not None else None,
        "DB0_tables": list(session.DB0.keys()) if session.DB0 else None,
        "DB1_ready": session.download_ready,
    })

def show_table_edit_ui(
    table_key: str,
    schema: dict,
    session,
    default_prefill_table: pd.DataFrame,
    title: str = "",
    description: str = "",
    step_index: int = None,
    step_list: list[str] = None
):
    cluster_count = len(session.cluster_overview)
    should_rebuild = False

    if table_key not in session.DB_modified:
        should_rebuild = True
    else:
        existing_df = session.DB_modified[table_key]
        if len(existing_df) != cluster_count:
            should_rebuild = True

    if should_rebuild:
        mapped_columns = [col for col in session.feature_map.values() if col != "None"]
        summary_df = session.cluster_overview.copy()
        summary_df["const_type"] = [f"CLUSTER_{i}" for i in summary_df.index]
        summary_df["description"] = ["" for _ in summary_df.index]
        summary_df["reference"] = [
            json.dumps({col: "cluster" if col in mapped_columns else "empty" for col in schema})
            for _ in summary_df.index
        ]
        for col in schema:
            if col not in summary_df.columns:
                summary_df[col] = ""
        summary_df = summary_df[[col for col in schema] + ["reference"]]
        session.DB_modified[table_key] = summary_df

    session.DB_modified[table_key] = session.DB_modified[table_key].reset_index(drop=True)
    df_mod = session.DB_modified[table_key]
    updated_rows = []
    validation_errors = {}

    # if title:
    #     st.markdown(f"##### :material/factory: {title}")
    # if description:
    #     st.caption(description)

    for i, row in df_mod.iterrows():
        row_id = row.name
        st.markdown(f"###### {title or table_key} Cluster {i}")
        ref_dict = json.loads(row["reference"])
        editable_row = pd.DataFrame([row.drop("reference")]).reset_index(drop=True)

        # --- Prefill dropdown + apply button ---
        c1, c2 = st.columns([4,2])

        # --- Prefill dropdown
        with c1:
            prefill_key = f"default_select_{table_key}_{row_id}"
            selected_default = st.selectbox(
                label=f"Prefill missing values from:",
                options=["None"] + list(default_prefill_table["const_type"]),
                key=prefill_key,
                label_visibility="collapsed"
            )
        
        # --- Prefill key
        with c2:
            apply_prefill_key = f"apply_prefill_{table_key}_{row_id}"
            if st.button("Apply Prefill", key=apply_prefill_key, use_container_width=True):
                if selected_default != "None" and selected_default in default_prefill_table["const_type"].values:
                    default_row = default_prefill_table[default_prefill_table["const_type"] == selected_default].iloc[0]
                    for col in schema:
                        if col in editable_row.columns and editable_row.at[0, col] in [None, "", "nan", "NaN"]:
                            editable_row.at[0, col] = str(default_row[col])
                            ref_dict[col] = "database"

        # --- Column config (simplified, no icons or types)
        column_config = {}
        for col in editable_row.columns:
            options = get_dropdown_options_for_field(session, col, schema)
            if options:
                column_config[col] = st.column_config.SelectboxColumn(
                    label=col,
                    options=options,
                    required=False
                )
            else:
                column_config[col] = st.column_config.TextColumn(label=col)

        edited = st.data_editor(
            editable_row,
            num_rows="fixed",
            column_config=column_config,
            use_container_width=True,
            key=f"editable_row_{table_key}_{row_id}"
        ).reset_index(drop=True)

        row_dict = edited.iloc[0].to_dict()
        new_ref = {}
        for col, new_val in row_dict.items():
            old_val = row.get(col)
            new_ref[col] = "user" if new_val != old_val else ref_dict.get(col, "cluster")

        edited["reference"] = json.dumps(new_ref)
        updated_rows.append(edited.iloc[0])

    # --- Save button & deferred validation ---
    if st.button(f"Save {title or table_key}", type="primary", use_container_width=True, key=f"save_button_{table_key}"):
        final_df = pd.DataFrame(updated_rows).reset_index(drop=True)
        session.DB_modified[table_key] = final_df

        for i, row in final_df.iterrows():
            errors = []
            for col in schema:
                val = row[col]
                if not validate_field_value(val, col, schema, session):
                    errors.append(f"`{col}` has invalid value: `{val}`")
            if not row["const_type"] or not row["description"]:
                errors.append("Missing required `const_type` or `description`")
            if errors:
                validation_errors[i] = errors

        if not validation_errors:
            if step_index is not None and step_list is not None:
                session.step_success[step_list[step_index]] = True
            st.success(f"✅ All entries in {title or table_key} saved and validated.")
        else:
            st.markdown("### :red[Validation Errors]")
            for i, errs in validation_errors.items():
                st.error(f"Cluster {i}:\n" + "\n".join(errs))