import streamlit as st
import numpy as np
import base64
from PIL import Image
import io

def display_result(result_data):
    """
    Displays the analyzed media with results overlay
    
    Args:
        result_data: Dictionary containing analysis results and media information
    """
    media_type = result_data.get("media_type", "unknown")
    prediction = result_data.get("prediction", "Unknown")
    
    # Apply color coding based on prediction
    if prediction.lower() == "fake":
        border_color = "red"
        badge_color = "#FF4B4B"
    else:
        border_color = "green"
        badge_color = "#00CC96"
    
    # Apply custom CSS for displaying the result
    st.markdown(f"""
    <style>
    .result-container {{
        border: 3px solid {border_color};
        border-radius: 5px;
        padding: 10px;
        position: relative;
    }}
    .result-badge {{
        position: absolute;
        top: 10px;
        right: 10px;
        background-color: {badge_color};
        color: white;
        padding: 5px 10px;
        border-radius: 3px;
        font-weight: bold;
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # Start container div
    st.markdown('<div class="result-container">', unsafe_allow_html=True)
    
    # Display the result badge
    st.markdown(f'<div class="result-badge">{prediction}</div>', unsafe_allow_html=True)
    
    # Display the appropriate media based on type
    if media_type == "image":
        if "media_content" in result_data:
            # Decode base64 image if provided
            try:
                image_data = base64.b64decode(result_data["media_content"])
                image = Image.open(io.BytesIO(image_data))
                st.image(image, use_column_width=True)
            except Exception as e:
                st.error(f"Error displaying image: {str(e)}")
                
                # Fallback to media_path if available
                if "media_path" in result_data:
                    st.image(result_data["media_path"], use_column_width=True)
        elif "media_path" in result_data:
            st.image(result_data["media_path"], use_column_width=True)
        else:
            st.error("No image data available to display")
    
    elif media_type == "audio":
        if "media_path" in result_data:
            st.audio(result_data["media_path"])
        elif "media_content" in result_data:
            # Decode base64 audio if provided
            try:
                audio_data = base64.b64decode(result_data["media_content"])
                st.audio(audio_data)
            except Exception as e:
                st.error(f"Error displaying audio: {str(e)}")
        else:
            st.error("No audio data available to display")
            
        # For audio, also show the waveform visualization
        if "waveform_data" in result_data:
            st.line_chart(result_data["waveform_data"])
        else:
            # Create dummy waveform visualization
            dummy_waveform = np.sin(np.linspace(0, 10*np.pi, 1000)) + np.random.normal(0, 0.1, 1000)
            st.line_chart(dummy_waveform)
    
    elif media_type == "video":
        if "media_path" in result_data:
            st.video(result_data["media_path"])
        elif "media_content" in result_data:
            # Decode base64 video if provided
            try:
                video_data = base64.b64decode(result_data["media_content"])
                with open("temp_video.mp4", "wb") as f:
                    f.write(video_data)
                st.video("temp_video.mp4")
            except Exception as e:
                st.error(f"Error displaying video: {str(e)}")
        else:
            st.error("No video data available to display")
    
    else:
        st.error(f"Unsupported media type: {media_type}")
    
    # End container div
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display analysis timestamp and duration
    if "timestamp" in result_data and "analysis_duration" in result_data:
        st.caption(f"Analyzed on {result_data['timestamp']} • Processing time: {result_data['analysis_duration']:.2f} seconds")
