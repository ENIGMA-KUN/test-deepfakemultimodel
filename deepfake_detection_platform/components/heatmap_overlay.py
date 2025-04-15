import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image, ImageOps

def display_heatmap(result_data):
    """
    Displays a heatmap overlay showing which areas of the media were flagged as manipulated.
    
    Args:
        result_data: Dictionary containing analysis results and heatmap information
    """
    media_type = result_data.get("media_type", "unknown")
    
    if media_type not in ["image", "video"]:
        st.warning("Heatmap visualization is only available for image and video content.")
        return
    
    # Check if we have heatmap data
    if "heatmap_data" not in result_data:
        # Generate dummy heatmap for demonstration if no real data
        create_dummy_heatmap(result_data)
        return
    
    # If we have real heatmap data, use it
    try:
        if isinstance(result_data["heatmap_data"], str) and result_data["heatmap_data"].startswith("data:image"):
            # Handle data URI format
            img_data = result_data["heatmap_data"].split(",")[1]
            heatmap_img = Image.open(io.BytesIO(base64.b64decode(img_data)))
            st.image(heatmap_img, use_column_width=True, caption="Manipulation Heatmap (Red areas indicate potential manipulation)")
        elif isinstance(result_data["heatmap_data"], np.ndarray):
            # Handle numpy array format
            display_numpy_heatmap(result_data["heatmap_data"], result_data)
        else:
            # Fallback to dummy heatmap
            create_dummy_heatmap(result_data)
    except Exception as e:
        st.error(f"Error displaying heatmap: {str(e)}")
        create_dummy_heatmap(result_data)

def display_numpy_heatmap(heatmap_array, result_data):
    """
    Displays a heatmap from a numpy array.
    
    Args:
        heatmap_array: Numpy array with heatmap values (higher = more likely manipulation)
        result_data: Dictionary containing analysis results and media information
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get original image if available
    original_img = None
    if "media_content" in result_data:
        try:
            img_data = base64.b64decode(result_data["media_content"])
            original_img = Image.open(io.BytesIO(img_data))
        except:
            pass
    elif "media_path" in result_data:
        try:
            original_img = Image.open(result_data["media_path"])
        except:
            pass
    
    if original_img:
        # Display original image
        ax.imshow(original_img)
        
        # Overlay heatmap with transparency
        heatmap = ax.imshow(heatmap_array, cmap='hot', alpha=0.6)
        plt.colorbar(heatmap, ax=ax, label='Manipulation Probability')
    else:
        # Just show the heatmap if no original image
        heatmap = ax.imshow(heatmap_array, cmap='hot')
        plt.colorbar(heatmap, ax=ax, label='Manipulation Probability')
    
    ax.set_title("Deepfake Detection Heatmap")
    ax.axis('off')
    
    # Display the plot in Streamlit
    st.pyplot(fig)
    
    # Add explanation of the heatmap
    st.markdown("""
    **Heatmap Analysis:**
    - Red/yellow areas indicate regions with high probability of manipulation
    - Blue/green areas are likely authentic
    - The brighter the color, the stronger the model's confidence
    """)

def create_dummy_heatmap(result_data):
    """
    Creates a dummy heatmap for demonstration purposes.
    
    Args:
        result_data: Dictionary containing analysis results and media information
    """
    # Get prediction to determine dummy heatmap appearance
    prediction = result_data.get("prediction", "Unknown").lower()
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Generate a random heatmap
    x, y = np.meshgrid(np.linspace(-3, 3, 256), np.linspace(-3, 3, 256))
    
    if prediction == "fake":
        # For fake predictions, create more intense hotspots
        z = np.exp(-(x**2 + y**2))
        # Add some random hotspots
        for _ in range(3):
            x0, y0 = np.random.uniform(-2, 2, 2)
            z += 0.5 * np.exp(-((x - x0)**2 + (y - y0)**2) / 0.1)
    else:
        # For real predictions, create a more uniform heatmap with low values
        z = 0.1 * np.exp(-(x**2 + y**2))
        # Add very slight random variations
        z += 0.05 * np.random.random(z.shape)
    
    # Normalize
    z = (z - z.min()) / (z.max() - z.min())
    
    # Display the heatmap
    heatmap = ax.imshow(z, cmap='hot')
    plt.colorbar(heatmap, ax=ax, label='Manipulation Probability')
    
    ax.set_title("Deepfake Detection Heatmap (Example)")
    ax.axis('off')
    
    # Display the plot in Streamlit
    st.pyplot(fig)
    
    if prediction == "fake":
        st.markdown("""
        **Heatmap Analysis (Example):**
        - Red/yellow areas show potential manipulated regions
        - Multiple hotspots detected, suggesting facial or image manipulation
        - Highest probabilities are concentrated around facial features
        """)
    else:
        st.markdown("""
        **Heatmap Analysis (Example):**
        - Low intensity throughout the image suggests authentic content
        - No significant manipulation patterns detected
        - Even distribution indicates natural image properties
        """)
    
    st.info("Note: This is an example visualization. For actual analysis, our models generate precise heatmaps highlighting exact manipulation regions.")
