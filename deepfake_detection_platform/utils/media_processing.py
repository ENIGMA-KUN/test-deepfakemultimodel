import streamlit as st
import numpy as np
import io
import os
import tempfile
from PIL import Image, ImageOps
import cv2
import librosa
import matplotlib.pyplot as plt
from typing import Optional, Union, Any

def get_media_preview(file: Any, media_type: str) -> Any:
    """
    Generates a preview of the uploaded media file for display.
    
    Args:
        file: The uploaded file object
        media_type: Type of media (image, audio, video)
    
    Returns:
        Processed media for preview display
    """
    if media_type == "image":
        return process_image(file)
    elif media_type == "audio":
        return process_audio(file)
    elif media_type == "video":
        return process_video(file)
    else:
        st.error(f"Unsupported media type: {media_type}")
        return None

def process_image(file: Any) -> Image.Image:
    """
    Processes an uploaded image file for preview.
    
    Args:
        file: The uploaded image file
    
    Returns:
        PIL Image object
    """
    try:
        # Read the image
        image = Image.open(io.BytesIO(file.getvalue()))
        
        # Ensure the image isn't too large for display
        max_size = (1200, 1200)
        image.thumbnail(max_size, Image.LANCZOS)
        
        # Handle different color modes and formats
        if image.mode == 'RGBA':
            # Convert transparent background to white
            background = Image.new('RGBA', image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image).convert('RGB')
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        # Return a placeholder image if there's an error
        placeholder = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Error loading image", (100, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return Image.fromarray(placeholder)

def process_audio(file: Any) -> str:
    """
    Processes an uploaded audio file for preview and visualization.
    
    Args:
        file: The uploaded audio file
    
    Returns:
        Path to the processed audio file
    """
    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name
        
        # Generate waveform visualization for display
        generate_audio_waveform(tmp_path)
        
        return tmp_path
    
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None

def generate_audio_waveform(audio_path: str) -> None:
    """
    Generates and displays a waveform visualization for an audio file.
    
    Args:
        audio_path: Path to the audio file
    """
    try:
        # Load the audio file
        y, sr = librosa.load(audio_path, sr=None, duration=30)  # Limit to 30 seconds
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.set_title("Audio Waveform")
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        
        # Display the plot
        st.pyplot(fig)
        
        # Generate and display a mel spectrogram
        fig, ax = plt.subplots(figsize=(10, 4))
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', 
                                       sr=sr, fmax=8000, ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title('Mel-frequency spectrogram')
        
        st.pyplot(fig)
    
    except Exception as e:
        st.error(f"Error generating audio visualization: {str(e)}")

def process_video(file: Any) -> str:
    """
    Processes an uploaded video file for preview.
    
    Args:
        file: The uploaded video file
    
    Returns:
        Path to the processed video file
    """
    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name
        
        # Extract a thumbnail for preview
        generate_video_thumbnail(tmp_path)
        
        return tmp_path
    
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return None

def generate_video_thumbnail(video_path: str) -> None:
    """
    Extracts and displays a thumbnail from a video file.
    
    Args:
        video_path: Path to the video file
    """
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Display video information
        st.markdown(f"""
        **Video Information:**
        - Duration: {duration:.2f} seconds
        - Frame Rate: {fps:.2f} fps
        - Total Frames: {frame_count}
        - Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))} x {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
        """)
        
        # Extract frames for thumbnail grid
        frames = []
        positions = [0.1, 0.3, 0.5, 0.7, 0.9]  # Positions as percentage of duration
        
        for pos in positions:
            frame_position = int(frame_count * pos)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        
        cap.release()
        
        # Display the thumbnail grid if we have frames
        if frames:
            # Create a grid of thumbnails
            fig, axes = plt.subplots(1, len(frames), figsize=(15, 3))
            
            for i, (frame, ax) in enumerate(zip(frames, axes if len(frames) > 1 else [axes])):
                ax.imshow(frame)
                ax.set_title(f"Frame {positions[i]*100:.0f}%")
                ax.axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    except Exception as e:
        st.error(f"Error generating video thumbnail: {str(e)}")

def get_media_type_from_extension(filename: str) -> str:
    """
    Determines the media type based on file extension.
    
    Args:
        filename: Name of the file
    
    Returns:
        Media type (image, audio, video or unknown)
    """
    ext = os.path.splitext(filename)[1].lower()
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
    audio_extensions = ['.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac']
    video_extensions = ['.mp4', '.mov', '.avi', '.webm', '.mkv', '.flv']
    
    if ext in image_extensions:
        return "image"
    elif ext in audio_extensions:
        return "audio"
    elif ext in video_extensions:
        return "video"
    else:
        return "unknown"

def resize_image(image: Image.Image, max_width: int = 1200, quality: int = 85) -> Image.Image:
    """
    Resizes an image while maintaining aspect ratio.
    
    Args:
        image: PIL Image object
        max_width: Maximum width of the resized image
        quality: JPEG quality (1-100)
    
    Returns:
        Resized PIL Image object
    """
    # Calculate new dimensions while maintaining aspect ratio
    width, height = image.size
    
    if width <= max_width:
        return image
    
    # Calculate new height to maintain aspect ratio
    new_height = int(height * (max_width / width))
    
    # Resize the image
    return image.resize((max_width, new_height), Image.LANCZOS)
