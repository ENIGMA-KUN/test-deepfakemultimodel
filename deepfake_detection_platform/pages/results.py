import streamlit as st
import os
import sys

# Add parent directory to path to import from components and utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.result_viewer import display_result
from components.confidence_meter import show_confidence
from components.heatmap_overlay import display_heatmap
from utils.api import get_analysis_result
from utils.report_generator import generate_pdf_report

st.set_page_config(
    page_title="Results - Deepfake Detection Platform",
    page_icon="🔍",
    layout="wide"
)

def main():
    st.title("Analysis Results")
    
    # Check if we have a result to display
    if "last_result_id" in st.session_state:
        result_id = st.session_state["last_result_id"]
        
        # Get analysis results
        result = get_analysis_result(result_id)
        
        if result:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Media Analysis")
                # Display the analyzed media with result overlay
                display_result(result)
                
                # For images and videos, show the heatmap where AI detected issues
                if result["media_type"] in ["image", "video"]:
                    st.subheader("Detection Heatmap")
                    display_heatmap(result)
            
            with col2:
                st.subheader("Analysis Summary")
                
                # Show the confidence meter for the prediction
                show_confidence(
                    score=result["confidence_score"],
                    label=result["prediction"]
                )
                
                st.markdown(f"""
                **Prediction:** {result["prediction"]}
                
                **Analysis Date:** {result["timestamp"]}
                
                **Media Type:** {result["media_type"].capitalize()}
                
                **File Name:** {result["filename"]}
                """)
                
                # Generate PDF report
                if st.button("Generate PDF Report"):
                    report_path = generate_pdf_report(result)
                    st.success(f"Report generated successfully!")
                    with open(report_path, "rb") as file:
                        st.download_button(
                            label="Download Report",
                            data=file,
                            file_name=f"deepfake_analysis_{result_id}.pdf",
                            mime="application/pdf"
                        )
            
            # Detailed analysis information
            st.subheader("Detailed Analysis")
            
            if "detection_details" in result:
                details = result["detection_details"]
                
                # Create tabs for different aspects of the analysis
                tabs = st.tabs(["Technical Analysis", "Detection Features", "Metadata"])
                
                with tabs[0]:
                    st.markdown(details.get("technical_explanation", "No technical analysis available."))
                
                with tabs[1]:
                    if "features" in details:
                        for feature, value in details["features"].items():
                            st.metric(label=feature, value=f"{value:.2f}")
                    else:
                        st.write("No feature details available.")
                
                with tabs[2]:
                    if "metadata" in details:
                        st.json(details["metadata"])
                    else:
                        st.write("No metadata available.")
        else:
            st.error("Result not found. The analysis may still be in progress or an error occurred.")
    else:
        st.info("No analysis results to display. Please upload a file for analysis on the Upload page.")
        
        # Sample results button for demonstration
        if st.button("Show Sample Results"):
            st.session_state["last_result_id"] = "sample_result"
            st.experimental_rerun()

if __name__ == "__main__":
    main()
