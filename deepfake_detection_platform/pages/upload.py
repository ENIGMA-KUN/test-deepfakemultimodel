import streamlit as st
import os
import sys

# Add parent directory to path to import from components and utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.upload_widget import create_upload_widget
from utils.api import send_file_for_analysis, get_models, get_selected_model
from utils.media_processing import get_media_preview

st.set_page_config(
    page_title="Upload - Deepfake Detection Platform",
    page_icon="🔍",
    layout="wide"
)

def main():
    st.title("Upload Content for Analysis")
    
    st.markdown("""
    ## Upload your media for deepfake analysis
    
    You can upload images, audio files, or videos for analysis. Our AI models will analyze the content
    and generate a detailed report on whether the content is likely genuine or a deepfake.
    """)
    
    # Create tabs for different media types
    tab1, tab2, tab3 = st.tabs(["Image", "Audio", "Video"])
    
    with tab1:
        st.header("Image Analysis")
        uploaded_file = create_upload_widget(
            label="Upload an image file",
            accepted_types=["jpg", "jpeg", "png"],
            key="image_upload"
        )
        
        if uploaded_file:
            st.success("File uploaded successfully!")
            st.subheader("Preview")
            preview = get_media_preview(uploaded_file, "image")
            st.image(preview, caption="Uploaded Image")
            
            # Model selection
            st.subheader("Select Analysis Model")
            models = get_models("image")
            selected_model_info = get_selected_model("image")
            selected_model_id = selected_model_info.get('model_id') if selected_model_info else None
            
            model_options = []
            model_mapping = {}
            
            if "image" in models and models["image"]:
                for model in models["image"]:
                    model_name = f"{model['name']} (v{model['version']})"
                    model_options.append(model_name)
                    model_mapping[model_name] = model["id"]
                    
                # Default to the currently selected model
                default_index = 0
                for i, model in enumerate(models["image"]):
                    if model["id"] == selected_model_id:
                        default_index = i
                        break
                
                selected_model_name = st.selectbox(
                    "Choose image analysis model:",
                    options=model_options,
                    index=default_index,
                    key="image_model_select"
                )
                
                selected_model_id = model_mapping.get(selected_model_name)
                
                # Show model description
                for model in models["image"]:
                    if model["id"] == selected_model_id:
                        st.info(model["description"])
                        break
            else:
                st.warning("No image models available. Default model will be used.")
                selected_model_id = None
            
            if st.button("Analyze Image", key="analyze_image"):
                with st.spinner("Analyzing the image..."):
                    # Send file for analysis with selected model
                    result_id = send_file_for_analysis(uploaded_file, "image", selected_model_id)
                    if result_id:
                        st.session_state["last_result_id"] = result_id
                        st.success("Analysis complete! Redirecting to results...")
                        st.experimental_rerun()
    
    with tab2:
        st.header("Audio Analysis")
        uploaded_file = create_upload_widget(
            label="Upload an audio file",
            accepted_types=["mp3", "wav", "ogg"],
            key="audio_upload"
        )
        
        if uploaded_file:
            st.success("File uploaded successfully!")
            st.subheader("Preview")
            preview = get_media_preview(uploaded_file, "audio")
            st.audio(preview, format="audio/wav")
            
            # Model selection
            st.subheader("Select Analysis Model")
            models = get_models("audio")
            selected_model_info = get_selected_model("audio")
            selected_model_id = selected_model_info.get('model_id') if selected_model_info else None
            
            model_options = []
            model_mapping = {}
            
            if "audio" in models and models["audio"]:
                for model in models["audio"]:
                    model_name = f"{model['name']} (v{model['version']})"
                    model_options.append(model_name)
                    model_mapping[model_name] = model["id"]
                    
                # Default to the currently selected model
                default_index = 0
                for i, model in enumerate(models["audio"]):
                    if model["id"] == selected_model_id:
                        default_index = i
                        break
                
                selected_model_name = st.selectbox(
                    "Choose audio analysis model:",
                    options=model_options,
                    index=default_index,
                    key="audio_model_select"
                )
                
                selected_model_id = model_mapping.get(selected_model_name)
                
                # Show model description
                for model in models["audio"]:
                    if model["id"] == selected_model_id:
                        st.info(model["description"])
                        break
            else:
                st.warning("No audio models available. Default model will be used.")
                selected_model_id = None
            
            if st.button("Analyze Audio", key="analyze_audio"):
                with st.spinner("Analyzing the audio..."):
                    # Send file for analysis with selected model
                    result_id = send_file_for_analysis(uploaded_file, "audio", selected_model_id)
                    if result_id:
                        st.session_state["last_result_id"] = result_id
                        st.success("Analysis complete! Redirecting to results...")
                        st.experimental_rerun()
    
    with tab3:
        st.header("Video Analysis")
        uploaded_file = create_upload_widget(
            label="Upload a video file",
            accepted_types=["mp4", "mov", "avi"],
            key="video_upload"
        )
        
        if uploaded_file:
            st.success("File uploaded successfully!")
            st.subheader("Preview")
            preview = get_media_preview(uploaded_file, "video")
            st.video(preview)
            
            # Model selection
            st.subheader("Select Analysis Model")
            models = get_models("video")
            selected_model_info = get_selected_model("video")
            selected_model_id = selected_model_info.get('model_id') if selected_model_info else None
            
            model_options = []
            model_mapping = {}
            
            if "video" in models and models["video"]:
                for model in models["video"]:
                    model_name = f"{model['name']} (v{model['version']})"
                    model_options.append(model_name)
                    model_mapping[model_name] = model["id"]
                    
                # Default to the currently selected model
                default_index = 0
                for i, model in enumerate(models["video"]):
                    if model["id"] == selected_model_id:
                        default_index = i
                        break
                
                selected_model_name = st.selectbox(
                    "Choose video analysis model:",
                    options=model_options,
                    index=default_index,
                    key="video_model_select"
                )
                
                selected_model_id = model_mapping.get(selected_model_name)
                
                # Show model description
                for model in models["video"]:
                    if model["id"] == selected_model_id:
                        st.info(model["description"])
                        break
            else:
                st.warning("No video models available. Default model will be used.")
                selected_model_id = None
                
            if st.button("Analyze Video", key="analyze_video"):
                with st.spinner("Analyzing the video..."):
                    # Send file for analysis with selected model
                    result_id = send_file_for_analysis(uploaded_file, "video", selected_model_id)
                    if result_id:
                        st.session_state["last_result_id"] = result_id
                        st.success("Analysis complete! Redirecting to results...")
                        st.experimental_rerun()
    
    # Additional information
    st.sidebar.markdown("""## Model Selection
You can select different AI models for analyzing your media. Each model has different strengths and detection capabilities.

Visit the [Models](/Models) page to see details about all available models and their performance metrics.
    """)
    
    st.markdown("""
    ### How it works
    
    Our platform uses state-of-the-art deep learning models to analyze:
    - **Images**: Pixel manipulation detection, facial inconsistencies, and metadata analysis
    - **Audio**: Voice pattern analysis, synthetic voice detection
    - **Video**: Frame-by-frame analysis, temporal consistency, and lip-sync verification
    
    Your files are processed securely and analysis results are available immediately.
    """)

if __name__ == "__main__":
    main()
