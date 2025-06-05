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

PAGES = {
    "Upload Input": upload_page,
    "Assign Column Types": preprocessing_page,
    "Run Clustering": clustering_page,
    "Review Results": postprocessing_page,
    "Download Output": download_page,
}

st.sidebar.title("K-Prototypes Wizard")
selection = st.sidebar.radio("Navigation", list(PAGES.keys()))
PAGES[selection]()