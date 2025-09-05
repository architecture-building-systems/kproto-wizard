import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st

def about_page():
    st.header(":material/info: About")

    with st.container():
        st.markdown("#### :material/laptop_windows: Software")
        st.markdown("**City Energy Analyst (CEA)** is an urban building energy modeling (UBEM) platform and one of the first open-source initiatives of computation tools for the design of low-carbon and highly efficient cities. The CEA combines knowledge of urban design and energy systems engineering in an integrated simulation platform. This allows to study of the effects, trade-offs and synergies of urban design scenarios and energy infrastructure plans.")
        st.link_button(
            "City Energy Analyst",
            icon=":material/north_east:",
            url="https://www.cityenergyanalyst.com/"
        )

    with st.container():
        st.markdown("#### :material/article: Published Research ")
        st.markdown("**A generalizable framework for urban building energy model archetype generation using k-prototypes mixed-data clustering**")
        st.caption("Published in a CISBAT 2025 special issue of IOP's JOURNAL OF PHYSICS Conference Series")
        st.markdown("Developing building archetypes from urban data is essential for urban building energy modeling (UBEM), yet current approaches lack generalizability, particularly regarding categorical data. This paper compares k-means, k-medoids, and k-prototypes clustering using Zurich’s building registry to evaluate the impact of preprocessing and algorithm choice. Results show preprocessing had limited effect on clustering, while k-prototypes—capable of handling numerical and categorical data concurrently—produced practical, interpretable clusters. The method’s simplicity and robust handling of mixed-type data suggest k-prototypes can streamline UBEM archetype generation workflows.")
        st.link_button(
            "ETH Research Collection",
            icon=":material/north_east:",
            url="https://www.research-collection.ethz.ch/items/78c3ddb8-cd60-4e33-87ad-e6091ec4dda9"
        )

about_page()