import sys
from pathlib import Path

import streamlit as st
from wizard_pages.upload_page import upload_page
from wizard_pages.preprocessing_page import preprocessing_page
from wizard_pages.clustering_page import clustering_page
from wizard_pages.postprocessing_page import postprocessing_page
from wizard_pages.download_page import download_page

# Ensure project root is in Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))


# Pages
PAGES = {
    "Upload Input": upload_page,
    "Assign Column Types": preprocessing_page,
    "Run Clustering": clustering_page,
    "Review Results": postprocessing_page,
    "Download Output": download_page,
}


# Set default page if not already set
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Upload Input"


# Radio menu with pre-selected index based on current page
page_names = list(PAGES.keys())
current_index = page_names.index(st.session_state["current_page"])

st.sidebar.title("K-Prototypes Wizard")
selection = st.sidebar.radio("Navigation", page_names, index=current_index)

# Update session and load page
st.session_state["current_page"] = selection
PAGES[selection]()