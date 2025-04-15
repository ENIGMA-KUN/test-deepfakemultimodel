import streamlit as st
import os

st.set_page_config(
    page_title="Deepfake Detection Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set up the main page
def main():
    st.title("🔍 Deepfake Detection Platform")
    
    st.markdown("""
    ## Welcome to the Deepfake Detection Platform
    
    This platform uses advanced machine learning models to detect deepfake content in:
    - 🖼️ Images
    - 🎵 Audio
    - 🎬 Videos
    
    Use the sidebar to navigate between different sections of the application.
    """)
    
    st.sidebar.title("Navigation")
    st.sidebar.info("""
    - [Upload & Analyze](/Upload)
    - [View Results](/Results)
    - [Analysis History](/History)
    - [About Deepfakes](/About)
    """)
    
    # Display some statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Files Analyzed", value="0")
    
    with col2:
        st.metric(label="Deepfakes Detected", value="0")
    
    with col3:
        st.metric(label="Detection Accuracy", value="98%")

if __name__ == "__main__":
    main()
