import streamlit as st
import os
import sys
import pandas as pd
import requests
import json

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.api import get_models, select_model, get_selected_model

st.set_page_config(
    page_title="Models - Deepfake Detection Platform",
    page_icon="🔍",
    layout="wide"
)

def main():
    st.title("Deepfake Detection Models")
    
    st.markdown("""
    ## Select and manage detection models
    
    Our platform uses state-of-the-art AI models for detecting deepfakes across different media types.
    You can select which models to use for your analysis tasks.
    """)
    
    # Create tabs for different media types
    tab1, tab2, tab3 = st.tabs(["Image Models", "Audio Models", "Video Models"])
    
    # Get all available models
    models_data = get_models()
    
    # Function to create model cards for a media type
    def create_model_cards(media_type):
        if not models_data or media_type not in models_data:
            st.warning(f"No {media_type} models available.")
            return
        
        # Get the currently selected model
        selected_model_info = get_selected_model(media_type)
        selected_model_id = selected_model_info.get('model_id') if selected_model_info else None
        
        # Create model cards in a grid layout
        cols = st.columns(min(len(models_data[media_type]), 3))
        
        for i, model in enumerate(models_data[media_type]):
            col = cols[i % len(cols)]
            
            with col:
                # Create a card with a border
                card_border = "3px solid #00cc00" if model["id"] == selected_model_id else "1px solid #cccccc"
                
                st.markdown(f"""
                <div style="border: {card_border}; border-radius: 5px; padding: 15px; margin-bottom: 20px;">
                    <h3>{model["name"]}</h3>
                    <p><b>Framework:</b> {model["framework"]}</p>
                    <p><b>Version:</b> {model["version"]}</p>
                    <p>{model["description"]}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Select button
                if model["id"] == selected_model_id:
                    st.success("Currently selected")
                else:
                    if st.button(f"Select Model", key=f"select_{model['id']}"):
                        response = select_model(model["id"])
                        if response and response.get("status") == "success":
                            st.success(f"Selected {model['name']} for {media_type} analysis")
                            st.experimental_rerun()
                        else:
                            st.error(f"Error selecting model: {response.get('message', 'Unknown error')}")
                
                # View details button
                with st.expander("View Performance Details"):
                    # Show performance metrics if available
                    performance = models_data.get(media_type, [])[i].get("performance", {})
                    if performance:
                        metrics = pd.DataFrame({
                            "Metric": list(performance.keys()),
                            "Value": list(performance.values())
                        })
                        st.dataframe(metrics, use_container_width=True)
                        
                        # Create a bar chart of performance metrics
                        st.bar_chart(performance)
                    else:
                        st.info("No performance metrics available for this model.")
    
    # Fill each tab with model cards
    with tab1:
        st.header("Image Deepfake Detection Models")
        st.markdown("""
        These models analyze images to detect manipulations, facial swaps, and other image-based deepfakes.
        Select the model that best fits your needs.
        """)
        create_model_cards("image")
    
    with tab2:
        st.header("Audio Deepfake Detection Models")
        st.markdown("""
        These models analyze audio files to detect synthetic voices, audio manipulations, and other audio-based deepfakes.
        Select the model that best fits your needs.
        """)
        create_model_cards("audio")
    
    with tab3:
        st.header("Video Deepfake Detection Models")
        st.markdown("""
        These models analyze videos to detect facial manipulations, lip-sync inconsistencies, and other video-based deepfakes.
        Select the model that best fits your needs.
        """)
        create_model_cards("video")
    
    # Add information about model selection
    st.markdown("""
    ### How Model Selection Works
    
    - Each media type (image, audio, video) can have a different selected model
    - The selected model will be used for all analyses of that media type
    - If no model is explicitly selected, the default model will be used
    - Models labeled as "Not Available" have not been downloaded or are missing required files
    
    ### About Our Models
    
    Our platform offers a variety of pre-trained deepfake detection models, each with different strengths
    and characteristics. Models vary in:
    
    - **Accuracy**: How well the model performs on standard benchmarks
    - **Speed**: How quickly the model can process media
    - **Specialized Detection**: Some models excel at specific types of deepfakes
    
    For the best results, we recommend trying multiple models on the same content to gain confidence
    in the analysis results.
    """)

if __name__ == "__main__":
    main()
