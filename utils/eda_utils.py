from autoviz.AutoViz_Class import AutoViz_Class
import streamlit as st
import os

def run_autoviz(csv_file_path: str):
    """
    Runs AutoViz on the given CSV file and renders the report in Streamlit.

    Args:
        csv_file_path (str): Path to the CSV file to analyze.

    Returns:
        None
    """
    AV = AutoViz_Class()

    # Generate the AutoViz report (returns a matplotlib Figure)
    st.write("Generating AutoViz report... This may take a moment.")

    try:
        report = AV.AutoViz(
            filename=csv_file_path,
            sep=',',
            depVar='',
            dfte=None,
            header=0,
            verbose=0,
            lowess=False,
            chart_format='svg',
            max_rows_analyzed=150000,
            max_cols_analyzed=30
        )
        # AutoViz generates matplotlib figures and also saves HTML reports internally.
        # The report variable can be a matplotlib figure or None.

        if report is not None:
            # If report is a matplotlib figure, show it
            st.pyplot(report)
        else:
            st.info("AutoViz finished but returned no visual output.")
    except Exception as e:
        st.error(f"Error running AutoViz: {e}")
