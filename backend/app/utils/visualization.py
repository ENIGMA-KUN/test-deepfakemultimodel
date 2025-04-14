import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Any, Optional
import logging
import uuid

from app.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)


def generate_heatmap_visualization(
    image: np.ndarray, 
    heatmap: np.ndarray, 
    output_path: Optional[str] = None
) -> str:
    """
    Generate a visualization of a heatmap overlaid on an image.
    
    Args:
        image (np.ndarray): The original image
        heatmap (np.ndarray): The heatmap
        output_path (str, optional): Path to save the visualization
    
    Returns:
        str: Path to the saved visualization
    """
    try:
        if output_path is None:
            # Create a unique filename
            filename = f"heatmap_{uuid.uuid4().hex}.png"
            output_path = os.path.join("visualizations", filename)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Resize heatmap to match image if needed
        if heatmap.shape[:2] != image.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Normalize heatmap
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # Convert to colormap
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Convert image to BGR if it's RGB
        if image.shape[2] == 3 and image[0, 0, 0] <= image[0, 0, 2]:  # Simple RGB check
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        
        # Blend image and heatmap
        alpha = 0.6
        overlay = cv2.addWeighted(image_bgr, 1 - alpha, heatmap_colored, alpha, 0)
        
        # Save visualization
        cv2.imwrite(output_path, overlay)
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error generating heatmap visualization: {str(e)}")
        return ""


def generate_temporal_visualization(
    timestamps: List[float], 
    values: List[float], 
    threshold: float = 0.5, 
    title: str = "Temporal Analysis", 
    output_path: Optional[str] = None
) -> str:
    """
    Generate a visualization of temporal analysis.
    
    Args:
        timestamps (List[float]): Timestamps
        values (List[float]): Values at each timestamp
        threshold (float): Threshold line
        title (str): Title for the visualization
        output_path (str, optional): Path to save the visualization
    
    Returns:
        str: Path to the saved visualization
    """
    try:
        if output_path is None:
            # Create a unique filename
            filename = f"temporal_{uuid.uuid4().hex}.png"
            output_path = os.path.join("visualizations", filename)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create figure
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, values, 'b-', linewidth=2)
        plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.7)
        
        # Add labels and title
        plt.xlabel("Time (s)")
        plt.ylabel("Score")
        plt.title(title)
        
        # Add grid and legend
        plt.grid(True, alpha=0.3)
        plt.legend(["Detection Score", "Threshold"])
        
        # Customize appearance
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path, dpi=100)
        plt.close()
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error generating temporal visualization: {str(e)}")
        return ""


def generate_frequency_visualization(
    frequency_data: Dict[str, float], 
    title: str = "Frequency Analysis", 
    output_path: Optional[str] = None
) -> str:
    """
    Generate a visualization of frequency analysis.
    
    Args:
        frequency_data (Dict[str, float]): Frequency analysis data
        title (str): Title for the visualization
        output_path (str, optional): Path to save the visualization
    
    Returns:
        str: Path to the saved visualization
    """
    try:
        if output_path is None:
            # Create a unique filename
            filename = f"frequency_{uuid.uuid4().hex}.png"
            output_path = os.path.join("visualizations", filename)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Extract data
        categories = list(frequency_data.keys())
        values = list(frequency_data.values())
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Bar plot
        bars = plt.bar(range(len(categories)), values, color='skyblue')
        
        # Add labels and title
        plt.xlabel("Frequency Band")
        plt.ylabel("Energy")
        plt.title(title)
        plt.xticks(range(len(categories)), categories, rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # Customize appearance
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path, dpi=100)
        plt.close()
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error generating frequency visualization: {str(e)}")
        return ""


def generate_confidence_gauge(
    confidence: float, 
    output_path: Optional[str] = None
) -> str:
    """
    Generate a gauge visualization for confidence score.
    
    Args:
        confidence (float): Confidence score (0-1)
        output_path (str, optional): Path to save the visualization
    
    Returns:
        str: Path to the saved visualization
    """
    try:
        if output_path is None:
            # Create a unique filename
            filename = f"gauge_{uuid.uuid4().hex}.png"
            output_path = os.path.join("visualizations", filename)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Set up the gauge figure
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        
        # Hide regular axes
        ax.set_axis_off()
        
        # Create gauge background
        gauge_background = plt.Rectangle((0, 0), 1, 0.3, facecolor='lightgray', alpha=0.3)
        ax.add_patch(gauge_background)
        
        # Create gauge fill
        gauge_fill = plt.Rectangle((0, 0), confidence, 0.3, facecolor='red' if confidence > 0.5 else 'green')
        ax.add_patch(gauge_fill)
        
        # Add threshold marker
        plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.7, linewidth=2)
        
        # Add labels
        plt.text(0.05, 0.4, "Real", fontsize=12)
        plt.text(0.85, 0.4, "Fake", fontsize=12)
        plt.text(confidence, 0.15, f"{confidence:.2f}", fontsize=14, 
                 horizontalalignment='center', verticalalignment='center',
                 color='white' if 0.3 <= confidence <= 0.7 else 'black',
                 weight='bold')
        
        # Set axis limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.5)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=100)
        plt.close()
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error generating confidence gauge: {str(e)}")
        return ""


def generate_spectral_discontinuity_visualization(
    audio_path: str,
    splice_times: List[float],
    threshold: float = 0.5,
    output_path: Optional[str] = None
) -> str:
    """
    Generate a visualization of spectral discontinuities in audio.
    
    Args:
        audio_path (str): Path to the audio file
        splice_times (List[float]): Time points of potential splices in seconds
        threshold (float): Threshold used for splice detection
        output_path (str, optional): Path to save the visualization
    
    Returns:
        str: Path to the saved visualization
    """
    try:
        if output_path is None:
            # Create a unique filename
            filename = f"spectral_disc_{uuid.uuid4().hex}.png"
            output_path = os.path.join("visualizations", filename)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Load audio and generate spectrogram
        import librosa
        import librosa.display
        
        y, sr = librosa.load(audio_path, sr=16000)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot spectrogram
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
        plt.colorbar(format='%+2.0f dB')
        
        # Mark splice points with vertical lines
        for splice_time in splice_times:
            plt.axvline(x=splice_time, color='red', linestyle='--', alpha=0.7, linewidth=2)
        
        # Add labels and title
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title("Audio Spectrogram with Potential Splice Points")
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='red', linestyle='--', lw=2, label='Potential Splice')]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error generating spectral discontinuity visualization: {str(e)}")
        return ""


def generate_voice_consistency_visualization(
    timestamps: List[int],
    segment_diffs: List[float],
    mean_diff: float,
    output_path: Optional[str] = None
) -> str:
    """
    Generate a visualization of voice consistency over time.
    
    Args:
        timestamps (List[int]): Segment indices
        segment_diffs (List[float]): Difference between consecutive segments
        mean_diff (float): Mean difference value 
        output_path (str, optional): Path to save the visualization
    
    Returns:
        str: Path to the saved visualization
    """
    try:
        if output_path is None:
            # Create a unique filename
            filename = f"voice_consist_{uuid.uuid4().hex}.png"
            output_path = os.path.join("visualizations", filename)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot segment differences
        plt.bar(timestamps, segment_diffs, color='skyblue', alpha=0.7)
        
        # Add mean line
        plt.axhline(y=mean_diff, color='red', linestyle='-', alpha=0.7, linewidth=2)
        
        # Add labels and title
        plt.xlabel("Segment Index")
        plt.ylabel("Voice Characteristic Difference")
        plt.title("Voice Consistency Analysis")
        
        # Add grid and legend
        plt.grid(True, alpha=0.3)
        plt.legend(["Mean Difference", "Segment Difference"])
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=100)
        plt.close()
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error generating voice consistency visualization: {str(e)}")
        return ""


def generate_silence_segments_visualization(
    audio_path: str,
    silence_segments: List[Dict[str, float]],
    output_path: Optional[str] = None
) -> str:
    """
    Generate a visualization of silence segments in audio.
    
    Args:
        audio_path (str): Path to the audio file
        silence_segments (List[Dict]): List of silence segments with start/end times
        output_path (str, optional): Path to save the visualization
    
    Returns:
        str: Path to the saved visualization
    """
    try:
        if output_path is None:
            # Create a unique filename
            filename = f"silence_{uuid.uuid4().hex}.png"
            output_path = os.path.join("visualizations", filename)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Load audio
        import librosa
        import librosa.display
        
        y, sr = librosa.load(audio_path, sr=16000)
        duration = len(y) / sr
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot waveform in first subplot
        librosa.display.waveshow(y, sr=sr, ax=ax1)
        ax1.set_title("Audio Waveform with Silence Segments")
        
        # Mark silence regions on waveform
        for segment in silence_segments:
            start = segment["start"]
            end = segment["end"]
            ax1.axvspan(start, end, alpha=0.3, color='red')
        
        # Add silence visualization in second subplot
        ax2.set_xlim(0, duration)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Silence")
        ax2.set_xlabel("Time (s)")
        
        # Plot silence regions as blocks
        for segment in silence_segments:
            start = segment["start"]
            end = segment["end"]
            width = end - start
            ax2.add_patch(plt.Rectangle((start, 0), width, 1, facecolor='red', alpha=0.7))
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.7, label='Silence')]
        ax2.legend(handles=legend_elements, loc='upper right')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error generating silence segments visualization: {str(e)}")
        return ""


def generate_audio_comparison_visualization(
    original_features: Dict[str, List[float]],
    comparison_features: Dict[str, List[float]],
    feature_names: List[str],
    output_path: Optional[str] = None
) -> str:
    """
    Generate a comparison visualization between two sets of audio features.
    
    Args:
        original_features (Dict): Features from a reference audio
        comparison_features (Dict): Features from the audio being analyzed
        feature_names (List[str]): Names of the features to compare
        output_path (str, optional): Path to save the visualization
    
    Returns:
        str: Path to the saved visualization
    """
    try:
        if output_path is None:
            # Create a unique filename
            filename = f"audio_compare_{uuid.uuid4().hex}.png"
            output_path = os.path.join("visualizations", filename)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Set up the comparison data
        features_to_plot = []
        original_values = []
        comparison_values = []
        
        for feat in feature_names:
            if feat in original_features and feat in comparison_features:
                # Use mean if the feature is a list/array
                if isinstance(original_features[feat], (list, np.ndarray)):
                    original_val = np.mean(original_features[feat])
                else:
                    original_val = original_features[feat]
                    
                if isinstance(comparison_features[feat], (list, np.ndarray)):
                    comparison_val = np.mean(comparison_features[feat])
                else:
                    comparison_val = comparison_features[feat]
                
                features_to_plot.append(feat)
                original_values.append(original_val)
                comparison_values.append(comparison_val)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Set bar positions
        x = np.arange(len(features_to_plot))
        width = 0.35
        
        # Create bars
        plt.bar(x - width/2, original_values, width, label='Original/Reference', color='blue', alpha=0.7)
        plt.bar(x + width/2, comparison_values, width, label='Analyzed Audio', color='red', alpha=0.7)
        
        # Add labels and title
        plt.xlabel('Audio Feature')
        plt.ylabel('Value')
        plt.title('Audio Feature Comparison')
        plt.xticks(x, features_to_plot, rotation=45)
        plt.legend()
        
        # Add grid
        plt.grid(True, alpha=0.3, axis='y')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=100)
        plt.close()
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error generating audio comparison visualization: {str(e)}")
        return ""
