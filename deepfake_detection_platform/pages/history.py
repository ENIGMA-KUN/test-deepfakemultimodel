import streamlit as st
import os
import sys
import pandas as pd
from datetime import datetime

# Add parent directory to path to import from components and utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.api import get_analysis_history

st.set_page_config(
    page_title="History - Deepfake Detection Platform",
    page_icon="🔍",
    layout="wide"
)

def main():
    st.title("Analysis History")
    
    st.markdown("""
    ## View your previous analysis results
    
    This page shows all your previous analyses, allowing you to revisit results 
    and track patterns over time.
    """)
    
    # Get analysis history from API
    history = get_analysis_history()
    
    if not history or len(history) == 0:
        st.info("No analysis history found. Start by analyzing some files on the Upload page.")
        
        # Create sample history for demonstration
        if st.button("Load Sample History"):
            history = [
                {
                    "id": f"sample_{i}",
                    "timestamp": (datetime.now()).isoformat(),
                    "filename": f"sample_file_{i}.{media_type}",
                    "media_type": media_type,
                    "prediction": "Real" if i % 2 == 0 else "Fake",
                    "confidence_score": 0.7 + (i / 10)
                }
                for i, media_type in enumerate(["jpg", "mp3", "mp4"])
            ]
    
    if history and len(history) > 0:
        # Create dataframe for easy display
        df = pd.DataFrame(history)
        
        # Add a "View" button column
        df["Action"] = "View Results"
        
        # Convert timestamp to datetime and format
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime('%Y-%m-%d %H:%M')
            df = df.rename(columns={"timestamp": "Date & Time"})
        
        # Rename columns for display
        df = df.rename(columns={
            "id": "ID",
            "filename": "Filename",
            "media_type": "Media Type",
            "prediction": "Result",
            "confidence_score": "Confidence"
        })
        
        # Format the confidence score as percentage
        if "Confidence" in df.columns:
            df["Confidence"] = df["Confidence"].apply(lambda x: f"{x*100:.1f}%")
        
        # Add filter options in the sidebar
        st.sidebar.subheader("Filter Results")
        
        # Media type filter
        if "Media Type" in df.columns:
            media_types = ["All"] + sorted(df["Media Type"].unique().tolist())
            selected_media_type = st.sidebar.selectbox("Media Type", media_types)
            
            if selected_media_type != "All":
                df = df[df["Media Type"] == selected_media_type]
        
        # Prediction filter
        if "Result" in df.columns:
            result_types = ["All"] + sorted(df["Result"].unique().tolist())
            selected_result = st.sidebar.selectbox("Result", result_types)
            
            if selected_result != "All":
                df = df[df["Result"] == selected_result]
        
        # Date range filter
        if "Date & Time" in df.columns:
            df_dates = pd.to_datetime(df["Date & Time"])
            min_date = df_dates.min().date()
            max_date = df_dates.max().date()
            
            date_range = st.sidebar.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            if len(date_range) == 2:
                start_date, end_date = date_range
                mask = (df_dates.dt.date >= start_date) & (df_dates.dt.date <= end_date)
                df = df[mask]
        
        # Display the filtered dataframe
        st.dataframe(df, use_container_width=True)
        
        # Allow clicking on a row to view results
        st.markdown("**Click on any row to view detailed results**")
        
        selected_row = st.data_editor(
            df,
            disabled=True,
            hide_index=True,
            use_container_width=True,
            key="history_table"
        )
        
        if st.session_state.get("history_table_edited_rows"):
            edited_row_index = list(st.session_state["history_table_edited_rows"].keys())[0]
            selected_id = df.iloc[int(edited_row_index)]["ID"]
            st.session_state["last_result_id"] = selected_id
            st.success(f"Loading results for ID: {selected_id}")
            st.switch_page("pages/results.py")
        
        # Add export options
        st.subheader("Export History")
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name="deepfake_analysis_history.csv",
                mime="text/csv"
            )
        
        with col2:
            # Excel export requires the openpyxl package
            try:
                excel_buffer = pd.ExcelWriter("analysis_history.xlsx", engine='openpyxl')
                df.to_excel(excel_buffer, index=False, sheet_name="Analysis History")
                excel_buffer.close()
                
                with open("analysis_history.xlsx", "rb") as f:
                    excel_data = f.read()
                
                st.download_button(
                    label="Download as Excel",
                    data=excel_data,
                    file_name="deepfake_analysis_history.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.warning(f"Excel export not available: {str(e)}")

if __name__ == "__main__":
    main()
