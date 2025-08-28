import pandas as pd
import streamlit as st
import json
import io

from databasemaker.session import DatabaseMakerSession
from copy import deepcopy

from shared.constants import (
    DATABASEMAKER_STEPS,
    CONSTRUCTION_TYPE_SCHEMA
)

from shared.utils import (
    infer_cea_column_type,
    is_dropdown_field,
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
    show_debug_databasemaker_session_ui
)

from shared.clustering import (
    preprocess_loaded_file,
    run_kprototypes_clustering
)

from shared.vizualization import (
    visualize_column_data,
    plot_kprototypes_results
)

from shared.utils_database import (
    compute_final_df,
    highlight_cells_by_reference,
    get_baseline_labels,
    get_validation_map_from_schema
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

    cols = st.columns([5, 1, 1])
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


def set_rerun_flag():
    st.session_state["rerun_after_dropdown_change"] = True


def update_override_cell(override_df, row_id, col, new_val, source: str):
    """
    Updates a cell in the override_df and sets the correct reference source.

    Parameters:
        override_df (pd.DataFrame): The override table to modify
        row_id (int): The row index
        col (str): Column name to update
        new_val: New value to set
        source (str): One of "user", "clustering", or "database"
    """
    override_df.at[row_id, col] = new_val
    try:
        ref_dict = json.loads(override_df.at[row_id, "Reference"] or "{}")
    except Exception:
        ref_dict = {}
    ref_dict[col] = source
    override_df.at[row_id, "Reference"] = json.dumps(ref_dict)

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
        session.input_column_metadata = {
            col["name"]: (col["dtype"], col["unique"]) for col in metadata["column_summary"]
        }
        session.column_metadata = {
            col: (
                schema["source_table"] if "source_table" in schema else "construction_types",
                schema.get("validator")
            )
            for col, schema in CONSTRUCTION_TYPE_SCHEMA.items()
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
    h1, h2 = st.columns([1,1])
    h1.markdown("Input Feature")
    h2.markdown("Manage")

    # --- Table Rows ---
    for col in df.columns:
        with st.container(border=True):
            dtype, n_unique = session.column_metadata.get(col, ("unknown", "?"))
            default_type = session.column_types.get(col, "off")
            default_map = session.feature_map.get(col, "None")

            c1, c2 = st.columns([1,1])
            
            with c1:
                # --- Feature ---
                st.markdown(f"**{col}**")

                # --- Feature chart ---
                col_data = df[col]
                col_dtype = session.column_types.get(col, "off")
                chart = visualize_column_data(col_dtype, col_data, col_name=col)
                st.altair_chart(chart, use_container_width=True)
            
            with c2:
                
                # --- Column Mapper ---
                session.feature_map[col] = st.selectbox(
                    "Map training dataset feature to features CEA database",
                    cea_columns,
                    index=cea_columns.index(default_map) if default_map in cea_columns else 0,
                    key=f"map_{col}",
                    label_visibility="visible",
                    disabled=(col == "name")  
                )
                
                # # --- Column Data Type ---
                # session.column_types[col] = st.segmented_control(
                #     label="Data Type",
                #     label_visibility="visible",
                #     options=["categorical", "numerical", "off"],
                #     selection_mode="single",
                #     default=default_type,
                #     format_func=lambda x: x,
                #     key=f"type_{col}",
                #     disabled=(col == "name")
                # )

                # --- Column Data Type (dropdown) ---
                type_options = ["categorical", "numerical", "off"]
                default_index = type_options.index(default_type.lower()) if default_type.lower() in type_options else 2

                selected_type = st.selectbox(
                    "Feature datatype",
                    options=[opt.capitalize() for opt in type_options],
                    index=default_index,
                    key=f"type_{col}",
                    label_visibility="visible",
                    disabled=(col == "name")
                )

                # Normalize to lowercase for internal logic
                session.column_types[col] = selected_type.lower()


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
            session.best_k = best_k

            log(f"🎉 Clustering complete. Least cost k = {best_k}")

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

    # --- Layout Header ---
    h1, h2 = st.columns([1,1])
    h1.markdown("Clustering Results")
    h2.markdown("Review")

    c1, c2 = st.columns([1,1])

    # Review Column
    with c2:
        with st.container(border=True):
            st.markdown("Customize construction typology count by selecting k value. Defaults to peak k.")
            available_ks = sorted(session.cost_per_k.keys())
            default_k = session.peak_k or session.best_k
            selected_k = st.slider(
                "Select k",
                min_value=min(available_ks),
                max_value=max(available_ks),
                value=default_k,
                key=f"custom_k_slider_{session.name}"
            )
    
    # Clustering results
    with c1:
        with st.expander(":material/analytics: Clustering Metrics", expanded=True):
            col1, col2, col3 = st.columns(3)
            col1.metric("Peak k (max silhouette)", session.peak_k)
            col2.metric("Shoulder k (≥ 0.5 silhouette)", session.shoulder_k)
            col3.metric("Least cost k", session.best_k)

        with st.expander(":material/line_axis: All Clustering Results", expanded=True):
            fig = plot_kprototypes_results(
                k_range=session.cost_per_k.keys(),
                costs=session.cost_per_k,
                silhouettes=session.silhouette_per_k,
                peak_k=session.peak_k,
                shoulder_k=session.shoulder_k
            )
            st.plotly_chart(fig, use_container_width=True)

            try:
                cluster_labels = session.get_assignments_for_k(selected_k)
            except ValueError as e:
                st.error(str(e))
                return


    with st.expander(":material/table: Cluster Summary Table", expanded=True):
        df_clustered = session.X0.copy()
        df_clustered["cluster"] = cluster_labels
        df_summary = df_clustered.groupby("cluster").agg(
            lambda x: x.mode().iloc[0] if not x.isnull().all() else None
        ).reset_index()

        # Fully reset cluster-dependent parts of the session
        session.selected_k = selected_k
        session.clustered_df = df_clustered
        session.cluster_overview = df_summary

        session.DB_cluster = {}
        session.DB_override = {}
        session.DB1 = None
        session.download_ready = False

        session.populate_DB_cluster()
        session.initialize_DB_override()

        st.markdown("#### Cluster Summary Table")
        st.dataframe(df_summary, use_container_width=True)


def show_review_database_ui(session, step_index: int, step_list: list[str]):
    import json
    from copy import deepcopy

    navigation_bar(session, step_index, step_list)

    # Assign tables
    table = "construction_types"
    cluster_df = session.DB_cluster.get(table)
    override_df = session.DB_override.get(table)
    baseline_df = session.DB0.get(table)

    if cluster_df is None:
        st.warning("Missing cluster_df")
        return
    if override_df is None:
        st.warning("Missing override_df")
        return
    if baseline_df is None:
        st.warning("Missing baseline_df")
        return

    # Create validation map from schema
    validation_map = get_validation_map_from_schema({"construction_types": CONSTRUCTION_TYPE_SCHEMA})

    # --- Section 0: Validate + Save Button ---
    st.subheader("Finalize Construction Types Table")
    if st.button("Validate and Save", type="primary"):
        validation_errors = session.validate_DB_override(validation_map)
        if validation_errors:
            st.error("Validation failed. Please correct the highlighted issues.")
            for (table, row, col), msg in validation_errors.items():
                st.markdown(f"- **{table}**, row `{row}`, column `{col}`: {msg}")
        else:
            session.compute_DB1()
            session.download_ready = True

            current_step = step_list[step_index]
            next_step = step_list[step_index + 1] if step_index + 1 < len(step_list) else None
            session.step_success[current_step] = True
            session.step = next_step

            st.success("Validation passed. Database saved and ready for download.")
            st.rerun()

    # --- Section 1: Read-only overview ---
    st.subheader("Current Table Overview")
    final_df = compute_final_df(cluster_df, override_df)
    styled = highlight_cells_by_reference(final_df)
    st.dataframe(styled, use_container_width=True)

    # --- Section 2: Row-level editor with consolidated cluster view ---

    # Database editor ui and
    with st.container():
        st.subheader("Populate construction archetype features")
        
        ui_cluster_select, ui_autofill_select = st.columns(2)
        
        with ui_cluster_select.container(border=True, height=205):
            st.markdown("##### Select active cluster")
            row_id = st.selectbox("Select a cluster to edit:", override_df.index)
        
        with ui_autofill_select.container(border=True, height=205):
            st.markdown("##### Autofill cluster features from CEA database")
            baseline_labels = get_baseline_labels(baseline_df, label_field="description")
            selected_baseline = st.selectbox(
                "Auto-fill empty values from baseline row:",
                options=baseline_df.index,
                format_func=lambda idx: baseline_labels.get(idx, str(idx))
            )

            # Load input state for comparison
            input_df = override_df.copy()
            input_ref_str = input_df.at[row_id, "Reference"] or "{}"
            input_dict = json.loads(input_ref_str)

            # Prepare output containers
            output_df = input_df.copy()
            output_dict = deepcopy(input_dict)
            proposed_changes = {}

            # Button
            if st.button("Apply Auto-Fill", key=f"apply_baseline_{row_id}", use_container_width=True, icon=":material/save:"):
                for col in override_df.columns:
                    if col == "Reference":
                        continue
                    cluster_locked = not pd.isna(cluster_df.at[row_id, col])
                    already_overridden = not pd.isna(input_df.at[row_id, col])
                    if not cluster_locked and not already_overridden:
                        val = baseline_df.at[selected_baseline, col]
                        output_df.at[row_id, col] = val
                        output_dict[col] = "database"
                output_df.at[row_id, "Reference"] = json.dumps(output_dict)
                session.DB_override[table] = output_df
                st.rerun()

    # Consolidated UI container per feature

    with st.container(border=True):
        result_col, save_col = st.columns([2,2])
        with result_col:
            st.markdown("##### Manually edit cluster features")

        for col in override_df.columns:
            if col == "Reference":
                continue

            with st.container(border=True):
                c1, c2 = st.columns([1, 1])

                with c1:
                    st.markdown(f"**{col}**")
                    value = input_df.at[row_id, col]
                    source = input_dict.get(col, "")
                    if source:
                        st.markdown(f":gray-background[`{source}`] {value if pd.notna(value) else ''}")
                    else:
                        st.text(f"{value if pd.notna(value) else ''}")

                with c2:
                    source = input_dict.get(col, "")
                    editable = source != "clustering"

                    if is_dropdown_field(col, CONSTRUCTION_TYPE_SCHEMA):
                        options = get_dropdown_options_for_field(session, col, CONSTRUCTION_TYPE_SCHEMA)
                        current_value = input_df.at[row_id, col]
                        default_index = 0
                        if pd.notna(current_value) and current_value in options:
                            default_index = options.index(current_value) + 1
                        user_input = st.selectbox(
                            label="",
                            options=[""] + options,
                            index=default_index,
                            key=f"edit_{row_id}_{col}",
                            label_visibility="collapsed",
                            disabled=not editable
                        )
                        proposed_changes[col] = user_input if user_input else None
                    else:
                        current_value = input_df.at[row_id, col]
                        user_input = st.text_input(
                            label="",
                            value="" if pd.isna(current_value) else str(current_value),
                            key=f"edit_{row_id}_{col}",
                            label_visibility="collapsed",
                            disabled=not editable
                        )
                        proposed_changes[col] = user_input if user_input else None

    with save_col:
        if st.button("Apply Manual Edits", key=f"save_row_{row_id}", use_container_width=True, icon=":material/save:"):
            for col, new_val in proposed_changes.items():
                if col == "Reference":
                    continue
                old_val = input_df.at[row_id, col]

                if pd.isna(old_val) and new_val:
                    output_df.at[row_id, col] = new_val
                    output_dict[col] = "user"
                elif new_val != old_val:
                    output_df.at[row_id, col] = new_val
                    output_dict[col] = "user"
                else:
                    output_dict[col] = input_dict.get(col, "clustering")

            output_df.at[row_id, "Reference"] = json.dumps(output_dict)
            session.DB_override[table] = output_df
            st.rerun()






def show_download_database_ui(session, step_index: int, step_list: list[str]):
    navigation_bar(session, step_index, step_list)

    if not session.download_ready or session.DB1 is None:
        st.warning("Database has not yet been finalized. Please return to the previous step to validate and save.")
        return

    # --- Preview final construction types ---
    st.markdown("###### :material/table: Construction Types")
    st.caption("Final construction types table with typologies defined by clusters and/or user edits")
    if "construction_types" in session.DB1:
        st.dataframe(session.DB1["construction_types"], use_container_width=True)
    else:
        st.warning("No construction_types table found in the final database.")

    # --- Download database as ZIP ---
    st.markdown("###### :material/download: Downloads")
    st.caption("Download complete CEA-compatible database and clustered training data")

    zip_buffer = export_database_to_zip(session.DB1)
    st.download_button(
        label="Download ZIP (Final CEA Database)",
        data=zip_buffer.getvalue(),
        file_name=f"{session.name}_cea_database.zip",
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
            file_name=f"{session.name}_clustered_training_data.csv",
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