import pandas as pd
import streamlit as st
import json
import io

from databasemaker.session import DatabaseMakerSession

from shared.constants import (
    DATABASEMAKER_STEPS,
    CONSTRUCTION_TYPE_SCHEMA
)

from shared.utils import (
    infer_cea_column_type,
    get_dropdown_options_for_field,
    validate_field_value,
    export_database_to_zip
)

from shared.utils_session import (
    init_databasemaker_state
)

from shared.utils_ui import (
    show_create_databasemaker_session_ui,
    show_switch_databasemaker_session_ui,
    show_manage_databasemaker_session_ui,
    show_debug_databasemaker_session_ui,
    show_table_edit_ui
)

from shared.clustering import (
    preprocess_loaded_file,
    run_kprototypes_clustering
)

from shared.vizualization import (
    plot_kprototypes_results
)

# --------------------------------------------------------------------------------------
# SESSION STATE
# --------------------------------------------------------------------------------------

init_databasemaker_state(DATABASEMAKER_STEPS)

# --------------------------------------------------------------------------------------
# UI ELEMENTS
# --------------------------------------------------------------------------------------

def navigation_bar(session, step_index: int, step_list: list[str]):
    current_step = step_list[step_index]
    progress_state = session.step_success.get(current_step, False)

    cols = st.columns([4, 1, 1])
    with cols[0]:
        st.markdown(f"#### Step {step_index + 1}: {current_step}")
    if step_index > 0 and cols[1].button("← Back", use_container_width=True):
        session.step = step_list[step_index - 1]
        st.rerun()
    if step_index < len(step_list) - 1 and cols[2].button(
        "Next →",
        use_container_width=True,
        disabled=not progress_state,
        type="primary",
    ):
        session.step = step_list[step_index + 1]
        st.rerun()


def render_sidebar_progress(session, step_list: list[str]):
    total_steps = len(step_list)
    completed_steps = sum(session.step_success.get(step, False) for step in step_list)
    current_step = session.step

    with st.sidebar:
        st.markdown(f"### :material/folder_open: `{session.name}`")
        st.progress(
            completed_steps / total_steps,
            text=f"{completed_steps} of {total_steps} steps complete"
        )


# --------------------------------------------------------------------------------------
# SUBPAGE UI LAYOUTS
# --------------------------------------------------------------------------------------

def show_column_mapping_ui(session, step_index: int, step_list: list[str]):
    navigation_bar(session, step_index, step_list)

    df = session.X0
    if df is None or df.empty:
        st.warning("No input data found in the session.")
        return

    # Load available columns in construction_types (excluding 'name')
    cea_columns = list(session.DB0.get("construction_types", pd.DataFrame()).columns)
    cea_columns = [col for col in cea_columns if col != "name"]
    cea_columns.insert(0, "None")

    # Infer metadata if not yet present
    if session.column_types is None or session.column_metadata is None:
        metadata = preprocess_loaded_file(df)
        inferred = {col["name"]: col["inferred_type"] for col in metadata["column_summary"]}
        if "name" in inferred:
            inferred["name"] = "off"
        session.column_types = inferred
        session.column_metadata = {
            col["name"]: (col["dtype"], col["unique"]) for col in metadata["column_summary"]
        }

    if session.feature_map is None:
        session.feature_map = {}

    # --- Save button ---
    if st.button("Save Assignments", icon=":material/save:", use_container_width=True, type="primary"):
        session.reset_clustering()

        valid = True
        for col, mapped in session.feature_map.items():
            if mapped == "None":
                continue
            input_type = session.column_types.get(col)
            cea_type = infer_cea_column_type(session.DB0["construction_types"], mapped)
            if input_type != cea_type:
                st.warning(f"Type mismatch: '{col}' ({input_type}) vs. '{mapped}' ({cea_type})")
                valid = False

        if valid:
            num_num = sum(v == "numerical" for v in session.column_types.values())
            num_cat = sum(v == "categorical" for v in session.column_types.values())
            st.success(f"Saved: {num_num} numerical and {num_cat} categorical columns.")

            current_step = step_list[step_index]
            next_step = step_list[step_index + 1] if step_index + 1 < len(step_list) else None

            session.step_success[current_step] = True
            session.step = next_step
            st.rerun()

    # --- Table Header ---
    h1, h2, h3, h4, h5 = st.columns([2, 1, 1, 2, 3])
    h1.markdown("**Input Column**")
    h2.markdown("**Dtype**")
    h3.markdown("**Unique**")
    h4.markdown("**Map to DB**")
    h5.markdown("**Assign Type**")

    # --- Table Rows ---
    for col in df.columns:
        dtype, n_unique = session.column_metadata.get(col, ("unknown", "?"))
        default_type = session.column_types.get(col, "off")
        default_map = session.feature_map.get(col, "None")

        c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 2, 3])
        with c1:
            st.markdown(f"**{col}**")
        with c2:
            st.markdown(f"`{dtype}`")
        with c3:
            st.markdown(f"`{n_unique}`")
        with c4:
            session.feature_map[col] = st.selectbox(
                "Map to",
                cea_columns,
                index=cea_columns.index(default_map) if default_map in cea_columns else 0,
                key=f"map_{col}",
                label_visibility="collapsed"
            )
        with c5:
            session.column_types[col] = st.segmented_control(
                label="Type",
                label_visibility="collapsed",
                options=["categorical", "numerical", "off"],
                selection_mode="single",
                default=default_type,
                format_func=lambda x: x,
                key=f"type_{col}",
                disabled=(col == "name")
            )


def show_clustering_ui(session, step_index: int, step_list: list[str]):
    navigation_bar(session, step_index, step_list)

    # --- Validate session input ---
    df = session.X0
    column_types = session.column_types

    if df is None or column_types is None:
        st.warning("Missing input data or column type assignments.")
        return

    # --- Containers ---
    button_container = st.container()
    log_display = st.empty()

    # --- Logging utility ---
    def log(msg):
        session.logger.log(msg)
        log_display.code(session.logger.get_log_text(), language="text")

    # --- Clustering logic ---
    def run_clustering_pipeline():
        session.logger.clear()
        log("🔄 Starting clustering pipeline...")

        try:
            with st.spinner("Running clustering..."):
                clustered, overview, best_k, cost_dict, sil_dict, peak_k, shoulder_k, assignments = run_kprototypes_clustering(
                    df,
                    column_types,
                    k_range=(2, 30),
                    log_func=log
                )

            session.set_clustering_output(
                clustered_df=clustered,
                overview_df=overview,
                k=best_k,
                cost_dict=cost_dict,
                sil_dict=sil_dict,
                assignments=assignments
            )
            session.peak_k = peak_k
            session.shoulder_k = shoulder_k

            log(f"🎉 Clustering complete. Best k = {best_k}")

            session.step_success[step_list[step_index]] = True
            session.step = step_list[step_index + 1]  # Move to next step

        except Exception as e:
            log(f"❌ Clustering failed: {e}")
            st.error(f"Clustering failed: {e}")

        finally:
            session.clustering_running = False
            st.rerun()

    # --- Run if flagged ---
    if session.clustering_running:
        run_clustering_pipeline()

    # --- Run Button ---
    if st.button(" Run Clustering", use_container_width=True, type="primary", disabled=session.clustering_running, icon=":material/smart_toy:"):
        session.clustering_running = True
        st.rerun()

    # --- Show logs ---
    if session.logger.get_log_text():
        log_display.code(session.logger.get_log_text(), language="text")


def show_review_clustering_ui(session, step_index: int, step_list: list[str]):
    navigation_bar(session, step_index, step_list)

    if not session.is_clustering_complete():
        st.warning("Clustering not yet completed.")
        return

    current_step = step_list[step_index]
    session.step_success[current_step] = True  # mark this step complete

    with st.expander(":material/analytics: Review Clustering Results", expanded=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Peak k (max silhouette)", session.peak_k)
        col2.metric("Shoulder k (≥ 0.5 silhouette)", session.shoulder_k)
        col3.metric("Least cost k", session.selected_k)

        fig = plot_kprototypes_results(
            k_range=session.cost_per_k.keys(),
            costs=session.cost_per_k,
            silhouettes=session.silhouette_per_k,
            peak_k=session.peak_k,
            shoulder_k=session.shoulder_k
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Customize k")
        st.caption("Allows you to reselect a k value. Defaults to peak k.")
        available_ks = sorted(session.cost_per_k.keys())
        default_k = session.peak_k or session.selected_k
        selected_k = st.slider(
            "Select k",
            min_value=min(available_ks),
            max_value=max(available_ks),
            value=default_k,
            key=f"custom_k_slider_{session.name}"
        )

        try:
            cluster_labels = session.get_assignments_for_k(selected_k)
        except ValueError as e:
            st.error(str(e))
            return

        df_clustered = session.X0.copy()
        df_clustered["cluster"] = cluster_labels
        df_summary = df_clustered.groupby("cluster").agg(
            lambda x: x.mode().iloc[0] if not x.isnull().all() else None
        ).reset_index()

        session.selected_k = selected_k
        session.clustered_df = df_clustered
        session.cluster_overview = df_summary

        st.markdown("#### Cluster Summary Table")
        st.dataframe(df_summary, use_container_width=True)


def show_review_database_ui(session, step_index: int, step_list: list[str]):
    navigation_bar(session, step_index, step_list)

    table_key = "construction_types"
    schema = CONSTRUCTION_TYPE_SCHEMA

    # --- Initialize modified construction_types table if missing ---
    if table_key not in session.DB_modified:
        base_df = session.DB0["construction_types"]
        mapped_columns = [col for col in session.feature_map.values() if col != "None"]

        summary_df = session.cluster_overview.copy()
        summary_df["const_type"] = [f"CLUSTER_{i}" for i in summary_df.index]
        summary_df["description"] = ["" for _ in summary_df.index]
        summary_df["reference"] = [
            json.dumps({col: "cluster" if col in mapped_columns else "empty" for col in schema})
            for _ in summary_df.index
        ]

        # Ensure all expected schema columns exist
        for col in schema:
            if col not in summary_df.columns:
                summary_df[col] = None

        summary_df = summary_df[[col for col in schema] + ["reference"]]
        session.DB_modified[table_key] = summary_df

    df_mod = session.DB_modified[table_key]
    default_df = session.DB0["construction_types"]
    updated_rows = []
    validation_errors = {}


    with st.expander("Construction Types"):
        show_table_edit_ui(
            table_key="construction_types",
            schema=CONSTRUCTION_TYPE_SCHEMA,
            session=session,
            default_prefill_table=session.DB0["construction_types"],
            title="Construction Types",
            description="Each row corresponds to one cluster. You may prefill values from a default construction type or edit manually.",
            step_index=step_index,
            step_list=step_list
        )

def show_download_database_ui(session, step_index: int, step_list: list[str]):
    navigation_bar(session, step_index, step_list)

    # --- Preview modified construction types ---
    st.markdown("###### :material/table: Construction Types")
    st.caption("Final construction types table with typologies defined by clusters")
    if "construction_types" in session.DB_modified:
        st.dataframe(session.DB_modified["construction_types"], use_container_width=True)
    else:
        st.warning("No construction types available.")
        return

    # --- Download database as ZIP, training data---
    st.markdown("###### :material/download: Downloads")
    st.caption("Download complete CEA database and training data with clusters appended")
    zip_buffer = export_database_to_zip(session.DB_modified)

    st.download_button(
        label="Download ZIP (Modified CEA Database)",
        data=zip_buffer.getvalue(),
        file_name="cea_modified_database.zip",
        mime="application/zip",
        use_container_width=True,
        type="primary"
    )

    # --- Download clustered training data ---
    if session.clustered_df is not None and "cluster" in session.clustered_df.columns:
        csv_buf = io.StringIO()
        session.clustered_df.to_csv(csv_buf, index=False)

        st.download_button(
            label="Download Clustered Training Set (CSV)",
            data=csv_buf.getvalue(),
            file_name="clustered_input_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("Clustered training data not available.")
                


# --------------------------------------------------------------------------------------
# PAGE LAYOUT
# --------------------------------------------------------------------------------------

def show_databasemaker_page():
    st.markdown("## :material/database: Database Maker")

    # --- Top bar with session actions ---
    t1, t2, t3 = st.columns(3)
    with t1.popover("New", icon=":material/add:", use_container_width=True):
        show_create_databasemaker_session_ui(DATABASEMAKER_STEPS)
    with t2.popover("Switch", icon=":material/menu_open:", use_container_width=True):
        show_switch_databasemaker_session_ui()
    with t3.popover("Manage", icon=":material/settings:", use_container_width=True):
        show_manage_databasemaker_session_ui()

    active_key = st.session_state.get("databasemaker__active_key")
    sessions = st.session_state.get("databasemaker__sessions", {})
    if not active_key or active_key not in sessions:
        st.info("No active session. Use the 'New' button above to create one.")
        st.stop()
    
    # --- Get current session and step
    session = sessions[active_key] 
    step = session.step
    try:
        step_index = DATABASEMAKER_STEPS.index(step)
    except ValueError:
        st.error(f"Unknown step: {step}")
        return

    # --- Sidebar
    render_sidebar_progress(session, DATABASEMAKER_STEPS)

    
    # --- Route to correct step UI
    if step == DATABASEMAKER_STEPS[0]:
        show_column_mapping_ui(session, step_index, DATABASEMAKER_STEPS)
    elif step == DATABASEMAKER_STEPS[1]:
        show_clustering_ui(session, step_index, DATABASEMAKER_STEPS)
    elif step == DATABASEMAKER_STEPS[2]:
        show_review_clustering_ui(session, step_index, DATABASEMAKER_STEPS)
    elif step == DATABASEMAKER_STEPS[3]:
        show_review_database_ui(session, step_index, DATABASEMAKER_STEPS)
    elif step == DATABASEMAKER_STEPS[4]:
        show_download_database_ui(session, step_index, DATABASEMAKER_STEPS)
    else:
        st.error(f"Unkown step: {step}")

    # --- Debug menu

    with st.expander("Debug Info (Developer Only)"):
        show_debug_databasemaker_session_ui()

show_databasemaker_page()