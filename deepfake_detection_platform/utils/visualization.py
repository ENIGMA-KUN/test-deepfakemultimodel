import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
import cv2
import io
import base64
from typing import Dict, List, Any, Tuple, Optional, Union

def create_confidence_chart(scores: Dict[str, float], threshold: float = 0.5) -> plt.Figure:
    """
    Creates a horizontal bar chart for visualization of confidence scores.
    
    Args:
        scores: Dictionary mapping feature names to confidence scores
        threshold: Threshold value for distinguishing real/fake
        
    Returns:
        Matplotlib figure object
    """
    # Sort scores for better visualization
    sorted_scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot horizontal bars
    features = list(sorted_scores.keys())
    values = list(sorted_scores.values())
    
    # Define colors based on scores (red for fake indicators, green for real)
    colors = ['red' if score >= threshold else 'green' for score in values]
    
    # Create bars
    bars = ax.barh(features, values, color=colors, alpha=0.7)
    
    # Add a vertical line at the threshold
    ax.axvline(x=threshold, color='gray', linestyle='--', alpha=0.7, label=f'Threshold ({threshold})')
    
    # Customize the plot
    ax.set_xlim(0, 1)
    ax.set_xlabel('Confidence Score')
    ax.set_title('Feature Analysis Confidence Scores')
    
    # Add value labels on the bars
    for i, v in enumerate(values):
        ax.text(max(v + 0.03, 0.1), i, f'{v:.2f}', va='center')
    
    # Add legend
    ax.legend()
    
    # Tight layout
    fig.tight_layout()
    
    return fig

def create_confusion_matrix(matrix_data: np.ndarray, classes: List[str]) -> plt.Figure:
    """
    Creates a confusion matrix visualization.
    
    Args:
        matrix_data: 2D numpy array containing the confusion matrix values
        classes: List of class names
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create the heatmap
    sns.heatmap(matrix_data, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=classes, yticklabels=classes, ax=ax)
    
    # Set labels
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    
    # Tight layout
    fig.tight_layout()
    
    return fig

def create_roc_curve(fpr: List[float], tpr: List[float], auc: float) -> plt.Figure:
    """
    Creates a ROC curve visualization.
    
    Args:
        fpr: List of false positive rates
        tpr: List of true positive rates
        auc: Area under the curve value
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the ROC curve
    ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    
    # Plot the diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Set plot limits and labels
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    
    # Add legend
    ax.legend(loc='lower right')
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Tight layout
    fig.tight_layout()
    
    return fig

def create_time_series_plot(data: pd.DataFrame, x_col: str, y_col: str, color_col: str = None) -> plt.Figure:
    """
    Creates a time series plot.
    
    Args:
        data: Pandas DataFrame with the data
        x_col: Column name for the x-axis (usually time)
        y_col: Column name for the y-axis
        color_col: Optional column name for color coding
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if color_col:
        # Group by the color column and plot each group
        for name, group in data.groupby(color_col):
            ax.plot(group[x_col], group[y_col], marker='o', linestyle='-', label=name)
        
        # Add legend
        ax.legend()
    else:
        # Simple line plot
        ax.plot(data[x_col], data[y_col], marker='o', linestyle='-')
    
    # Set labels and title
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f'{y_col} over {x_col}')
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels if they're dates
    plt.xticks(rotation=45)
    
    # Tight layout
    fig.tight_layout()
    
    return fig

def create_image_grid(images: List[np.ndarray], titles: List[str], cols: int = 3) -> plt.Figure:
    """
    Creates a grid of images for comparison.
    
    Args:
        images: List of images as numpy arrays
        titles: List of titles for each image
        cols: Number of columns in the grid
        
    Returns:
        Matplotlib figure object
    """
    # Calculate number of rows needed
    n_images = len(images)
    rows = (n_images + cols - 1) // cols
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    
    # Flatten axes array for easier indexing if multiple rows and columns
    if rows > 1 and cols > 1:
        axes = axes.flatten()
    elif rows == 1 and cols > 1:
        axes = axes  # axes is already 1D
    elif rows > 1 and cols == 1:
        axes = axes.flatten()
    else:  # Single image
        axes = [axes]
    
    # Add each image to the grid
    for i, (img, title) in enumerate(zip(images, titles)):
        if i < len(axes):
            if len(img.shape) == 2 or img.shape[2] == 1:  # Grayscale
                axes[i].imshow(img, cmap='gray')
            else:  # Color
                axes[i].imshow(img)
            
            axes[i].set_title(title)
            axes[i].axis('off')
    
    # Hide any unused subplots
    for i in range(n_images, len(axes)):
        fig.delaxes(axes[i])
    
    # Tight layout
    fig.tight_layout()
    
    return fig

def overlay_heatmap_on_image(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """
    Overlays a heatmap on an image.
    
    Args:
        image: Original image as numpy array
        heatmap: Heatmap as numpy array (same size as image)
        alpha: Transparency of the heatmap overlay
        
    Returns:
        Numpy array with heatmap overlaid on image
    """
    # Ensure image is RGB
    if len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Resize heatmap to match image if needed
    if image.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Normalize heatmap to 0-1 range
    heatmap_normalized = cv2.normalize(heatmap, None, 0, 1, cv2.NORM_MINMAX)
    
    # Convert heatmap to RGB colormap
    heatmap_colored = cv2.applyColorMap(
        (heatmap_normalized * 255).astype(np.uint8), 
        cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap on image
    overlay = cv2.addWeighted(image, 1, heatmap_colored, alpha, 0)
    
    return overlay

def create_interactive_3d_plot(data: pd.DataFrame, x_col: str, y_col: str, z_col: str, color_col: str = None) -> go.Figure:
    """
    Creates an interactive 3D scatter plot using Plotly.
    
    Args:
        data: Pandas DataFrame with the data
        x_col, y_col, z_col: Column names for the axes
        color_col: Optional column name for color coding
        
    Returns:
        Plotly figure object
    """
    if color_col:
        fig = px.scatter_3d(data, x=x_col, y=y_col, z=z_col, color=color_col)
    else:
        fig = px.scatter_3d(data, x=x_col, y=y_col, z=z_col)
    
    # Update layout
    fig.update_layout(
        title=f'3D Scatter Plot: {z_col} vs {x_col} and {y_col}',
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col
        )
    )
    
    return fig

def fig_to_base64(fig: plt.Figure) -> str:
    """
    Converts a matplotlib figure to a base64 encoded string.
    
    Args:
        fig: Matplotlib figure object
        
    Returns:
        Base64 encoded string of the figure
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"

def add_text_to_image(image: Image.Image, text: str, position: Tuple[int, int], 
                     font_size: int = 20, color: Tuple[int, int, int] = (255, 0, 0)) -> Image.Image:
    """
    Adds text to an image.
    
    Args:
        image: PIL Image object
        text: Text to add
        position: (x, y) position for the text
        font_size: Font size
        color: RGB color tuple
        
    Returns:
        PIL Image with added text
    """
    # Create a copy of the image
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    
    # Draw the text
    draw.text(position, text, fill=color, font=font)
    
    return img_copy

def create_comparison_slider(before_img: np.ndarray, after_img: np.ndarray) -> go.Figure:
    """
    Creates an interactive slider for comparing two images side by side.
    
    Args:
        before_img: Before image as numpy array
        after_img: After image as numpy array
        
    Returns:
        Plotly figure object
    """
    # Create figure
    fig = go.Figure()

    # Add before image
    fig.add_trace(
        go.Image(z=before_img)
    )

    # Add after image
    fig.add_trace(
        go.Image(z=after_img)
    )

    # Set initial slider position
    fig.data[0].visible = True
    fig.data[1].visible = False

    # Create steps for slider
    steps = []
    for i, label in enumerate(['Before', 'After']):
        step = dict(
            method="update",
            args=[{"visible": [False, False]},
                  {"title": f"{label}"}],
            label=label
        )
        step["args"][0]["visible"][i] = True
        steps.append(step)

    # Add slider
    fig.update_layout(
        sliders=[dict(
            active=0,
            currentvalue={"prefix": "Viewing: "},
            pad={"t": 50},
            steps=steps
        )]
    )

    return fig
