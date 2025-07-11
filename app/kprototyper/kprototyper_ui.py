import io
import uuid
import zipfile
import pandas as pd
import streamlit as st
from datetime import datetime

from kprototyper.session import KPrototyperSession

from kprototyper.logic import (
    preprocess_loaded_file,
    run_kprototypes_clustering,
    plot_kprototypes_results
)

from shared.utils import (
    get_available_regions
)

from shared.utils_session import (
    init_kprototyper_state,
    export_to_archetyper
)

from shared.utils_ui import (
    show_create_kprototyper_session_ui,
    show_switch_kprototyper_session_ui,
    show_manage_kprototyper_session_ui,
    show_debug_kprototyper_session_ui
)

# -------------------------------
# Session State
# -------------------------------

init_kprototyper_state()


# -------------------------------
# UI 
# -------------------------------

def navigation_bar(session, step_index: int):
    KPROTO_STEPS = ["Assign Column Types", "Run Clustering", "Review & Download"]
    current_step = KPROTO_STEPS[step_index]
    progress_state = session.step_success.get(current_step, False)

    cols = st.columns([4, 1, 1])
    with cols[0]:
        st.markdown(f"#### Step {step_index + 1}: {current_step}")
    if step_index > 0 and cols[1].button("← Back", use_container_width=True):
        session.step = KPROTO_STEPS[step_index - 1]
        st.rerun()
    if step_index < len(KPROTO_STEPS) - 1 and cols[2].button(
        "Next →",
        use_container_width=True,
        disabled=not progress_state,
        type="primary",
    ):
        session.step = KPROTO_STEPS[step_index + 1]
        st.rerun()


def show_column_typing_ui(session, step_index: int):
    navigation_bar(session, step_index)
    df = session.input_df

    if df is None or df.empty:
        st.warning("No input data found in the session.")
        return

    # Infer column types and metadata if not already set
    if session.column_config is None or session.column_metadata is None:
        metadata = preprocess_loaded_file(df)
        inferred = {col["name"]: col["inferred_type"] for col in metadata["column_summary"]}
        if "name" in inferred:
            inferred["name"] = "off"
        session.column_config = inferred
        session.column_metadata = {
            col["name"]: (col["dtype"], col["unique"]) for col in metadata["column_summary"]
        }

    if st.button("Save Assignments", icon=":material/save:", use_container_width=True, type="primary"):
        session.reset_clustering()
        session.step_success["Run Clustering"] = False  # ensure downstream is re-done

        num_num = sum(v == "numerical" for v in session.column_config.values())
        num_cat = sum(v == "categorical" for v in session.column_config.values())
        st.success(f"Saved: {num_num} numerical and {num_cat} categorical columns.")

        session.step_success["Assign Column Types"] = True
        session.step = "Run Clustering"
        st.rerun()

    # Simulated header row
    h1, h2, h3, h4 = st.columns([2, 1, 1, 3])
    h1.markdown("**Column**")
    h2.markdown("**Dtype**")
    h3.markdown("**Unique**")
    h4.markdown("**Assign Type**")

    for col in df.columns:
        dtype, n_unique = session.column_metadata.get(col, ("unknown", "?"))
        default_type = session.column_config.get(col, "off")

        c1, c2, c3, c4 = st.columns([2, 1, 1, 3])
        with c1:
            st.markdown(f"**{col}**")
        with c2:
            st.markdown(f"`{dtype}`")
        with c3:
            st.markdown(f"`{n_unique}`")
        with c4:
            session.column_config[col] = st.segmented_control(
                label="Type",
                label_visibility="collapsed",
                options=["categorical", "numerical", "off"],
                selection_mode="single",
                default=default_type,
                format_func=lambda x: x,
                key=f"type_{col}",
                disabled=(col == "name")
            )


def show_clustering_ui(session, step_index: int):
    navigation_bar(session, step_index)

    # --- Validate session input ---
    df = session.input_df
    column_config = session.column_config

    if df is None or column_config is None:
        st.warning("Missing input data or column assignments.")
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
                    column_config,
                    k_range=(2, 30),
                    log_func=log
                )

            session.set_output(
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
            session.step_success["Run Clustering"] = True
            session.step = "Review & Download"

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


def show_postprocessing_ui(session, step_index: int):
    import io, zipfile
    from datetime import datetime

    navigation_bar(session, step_index)

    if not session.is_complete():
        st.warning("Clustering not yet completed.")
        return

    session.step_success["Review & Download"] = True

    # --------------------
    # 1. Review Clustering
    # --------------------
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
        st.caption("Allows specification of a custom cluster count for final download and export. Defaults to peak k.")
        available_ks = sorted(session.cost_per_k.keys())
        default_k = session.peak_k or session.selected_k
        selected_k = st.slider(
            "Select k for export",
            min_value=min(available_ks),
            max_value=max(available_ks),
            value=default_k,
            key=f"custom_k_slider_{session.name}"
        )

        cluster_labels = session.get_assignments_for_k(selected_k)
        df_clustered = session.input_df.copy()
        df_clustered["cluster"] = cluster_labels
        df_summary = df_clustered.groupby("cluster").agg(
            lambda x: x.mode().iloc[0] if not x.isnull().all() else None
        ).reset_index()

        st.markdown("#### Cluster Summary Table")
        st.dataframe(df_summary, use_container_width=True)

    # --------------------
    # 2. Download Section
    # --------------------
    with st.expander(":material/download: Download Data"):
        st.caption("Download clustered results, overview summary, and original input data.")
        col1, col2 = st.columns([3, 2])
        with col1:
            format_choice = st.segmented_control(
                label="Format",
                label_visibility="collapsed",
                options=["CSV", "XLSX"],
                selection_mode="single",
                default="XLSX",
                key=f"download_format_{session.name}"
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        buffer = io.BytesIO()

        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            if format_choice == "CSV":
                zipf.writestr(f"cluster_overview_{timestamp}.csv", df_summary.to_csv(index=False))
                zipf.writestr(f"clustered_data_{timestamp}.csv", df_clustered.to_csv(index=False))
                zipf.writestr(f"original_data_{timestamp}.csv", session.input_df.to_csv(index=False))
            else:
                with io.BytesIO() as xlsx_buf:
                    with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
                        df_summary.to_excel(writer, sheet_name="Cluster Overview", index=False)
                        df_clustered.to_excel(writer, sheet_name="Clustered Data", index=False)
                        session.input_df.to_excel(writer, sheet_name="Original Data", index=False)
                    zipf.writestr(f"clustering_output_{timestamp}.xlsx", xlsx_buf.getvalue())

        with col2:
            st.download_button(
                label="Download ZIP",
                data=buffer.getvalue(),
                file_name=f"clustering_results_{timestamp}.zip",
                mime="application/zip",
                use_container_width=True,
                icon=":material/download:"
            )

    # --------------------
    # 3. Finalize & Export
    # --------------------
    with st.expander(":material/check_circle: Finalize and Continue", expanded=True):
        st.caption("Once finalized, this session becomes available to use in the Archetyper.")

        available_regions = get_available_regions()
        region = st.selectbox("Database region", available_regions)

        if st.button("Finalize and Open in Archetyper", type="primary", use_container_width=True):
            session.selected_k = selected_k
            session.clustered_df = df_clustered
            session.cluster_overview = df_summary
            session.status = "finalized"

            export_to_archetyper(session, region=region)
            st.switch_page("archetyper/archetyper_ui.py")


def render_sidebar_progress(session):
    KPROTO_STEPS = ["Assign Column Types", "Run Clustering", "Review & Download"]
    
    with st.sidebar:
        st.markdown(f":material/folder_open: Session :blue-background[{session.name}]")

        total_steps = len(KPROTO_STEPS)
        completed_steps = sum(session.step_success.get(step, False) for step in KPROTO_STEPS)
        current_step = session.step

        # Progress bar with dynamic label
        st.progress(
            completed_steps / total_steps,
            text=f"{current_step}"
        )
    

# -------------------------------
# Page Layout
# -------------------------------

KPROTO_STEPS = ["Assign Column Types", "Run Clustering", "Review & Download"]

def show_kprototyper_page():
    st.markdown("## :material/smart_toy: K-Prototyper")

    # --- Top bar with session actions ---
    t1, t2, t3 = st.columns(3)
    with t1.popover("New", icon=":material/add:", use_container_width=True):
        show_create_kprototyper_session_ui()
    with t2.popover("Switch", icon=":material/menu_open:", use_container_width=True):
        show_switch_kprototyper_session_ui()
    with t3.popover("Manage", icon=":material/settings:", use_container_width=True):
        show_manage_kprototyper_session_ui()

    # --- Guard clause: No session selected ---
    active_key = st.session_state.get("kprototyper__active_key")
    sessions = st.session_state.get("kprototyper__sessions", {})
    if not active_key or active_key not in sessions:
        st.info("No active session. Use the 'New' button above to create one.")
        st.stop()

    # --- Get current session and step ---
    session = sessions[active_key]
    render_sidebar_progress(session) 
    step = session.step

    try:
        step_index = KPROTO_STEPS.index(step)
    except ValueError:
        st.error(f"Unknown step: {step}")
        return

    # --- Route to correct step UI ---
    if step == "Assign Column Types":
        show_column_typing_ui(session, step_index)
    elif step == "Run Clustering":
        show_clustering_ui(session, step_index)
    elif step == "Review & Download":
        show_postprocessing_ui(session, step_index)
    else:
        st.error(f"Unknown step: {step}")

    # --- Debug expander for session state ---
    with st.expander("Debug", expanded=False):
        show_debug_kprototyper_session_ui()

show_kprototyper_page()