import streamlit as st
import os
from typing import List, Optional

def create_upload_widget(
    label: str, 
    accepted_types: List[str], 
    key: str,
    max_size_mb: int = 200
) -> Optional[object]:
    """
    Creates a custom file upload widget with enhanced UI.
    
    Args:
        label: Display label for the upload widget
        accepted_types: List of accepted file extensions without the dot
        key: Unique key for the Streamlit widget
        max_size_mb: Maximum file size in MB
    
    Returns:
        The uploaded file object or None if no file was uploaded
    """
    # Create a container with a border and styling
    upload_container = st.container()
    
    with upload_container:
        # Style the container to look like a dropzone
        st.markdown("""
        <style>
        .upload-container {
            border: 2px dashed #cccccc;
            border-radius: 5px;
            padding: 20px;
            text-align: center;
            margin-bottom: 20px;
            background-color: #f9f9f9;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Create a column layout for the upload area
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Format accepted types for display
            accepted_str = ", ".join([f".{ext}" for ext in accepted_types])
            
            # The actual file uploader
            uploaded_file = st.file_uploader(
                label=label,
                type=accepted_types,
                key=key,
                help=f"Accepted formats: {accepted_str}. Max size: {max_size_mb}MB"
            )
        
        with col2:
            st.markdown(f"""
            <div class="upload-container">
                <h4>📁 Drag & Drop</h4>
                <p>or click to browse files</p>
                <p><small>Accepted formats: {accepted_str}</small></p>
                <p><small>Max size: {max_size_mb}MB</small></p>
            </div>
            """, unsafe_allow_html=True)
    
    # If a file is uploaded, validate it
    if uploaded_file:
        # Check file size (Streamlit converts to bytes)
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            st.error(f"File is too large! Maximum allowed size is {max_size_mb}MB. Your file is {file_size_mb:.1f}MB.")
            return None
        
        # Validate file extension
        file_ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
        if file_ext not in [ext.lower() for ext in accepted_types]:
            st.error(f"Invalid file type! Accepted formats: {accepted_str}")
            return None
        
        return uploaded_file
    
    return None
