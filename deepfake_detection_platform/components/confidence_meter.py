import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

def show_confidence(score, label="Unknown"):
    """
    Displays a confidence meter visualization for the deepfake detection score.
    
    Args:
        score: Confidence score (0.0 to 1.0) where 1.0 is highest confidence
        label: Prediction label ("Real" or "Fake")
    """
    # Determine color based on label
    if label.lower() == "fake":
        color = "red"
        danger_zone = (0.5, 1.0)  # High confidence for fake is in the right zone
    else:
        color = "green"
        danger_zone = (0.0, 0.5)  # High confidence for real is in the left zone
    
    # Create a gauge chart to display the confidence score
    # Create a half-circle gauge
    
    # Apply custom styling for the confidence meter
    st.markdown(f"""
    <style>
    .confidence-container {{
        text-align: center;
        padding: 10px;
        border-radius: 5px;
        background-color: #f5f5f5;
        margin-bottom: 20px;
    }}
    .confidence-label {{
        font-size: 24px;
        font-weight: bold;
        color: {color};
        margin-bottom: 10px;
    }}
    .confidence-score {{
        font-size: 40px;
        font-weight: bold;
        color: {color};
    }}
    .confidence-description {{
        margin-top: 10px;
        font-style: italic;
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # Start container div
    st.markdown('<div class="confidence-container">', unsafe_allow_html=True)
    
    # Show the prediction label
    st.markdown(f'<div class="confidence-label">{label}</div>', unsafe_allow_html=True)
    
    # Show the confidence score as a percentage
    confidence_percent = int(score * 100)
    st.markdown(f'<div class="confidence-score">{confidence_percent}%</div>', unsafe_allow_html=True)
    
    # Create data for the gauge chart
    theta = np.linspace(0, np.pi, 100)
    radius = 0.5
    
    # Base data for the semi-circle
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    
    # Create a DataFrame for the gauge background
    gauge_bg = pd.DataFrame({
        'x': x,
        'y': y,
        'color': ['background'] * len(x)
    })
    
    # Determine the filled portion based on the score
    fill_idx = int(score * 100)
    x_fill = x[:fill_idx] if fill_idx > 0 else []
    y_fill = y[:fill_idx] if fill_idx > 0 else []
    
    # Create a DataFrame for the filled portion
    gauge_fill = pd.DataFrame({
        'x': x_fill,
        'y': y_fill,
        'color': [color] * len(x_fill)
    })
    
    # Combine both DataFrames
    gauge_data = pd.concat([gauge_bg, gauge_fill], ignore_index=True)
    
    # Create the gauge chart
    gauge_chart = alt.Chart(gauge_data).mark_area().encode(
        x=alt.X('x:Q', scale=alt.Scale(domain=[-0.5, 0.5])),
        y=alt.Y('y:Q', scale=alt.Scale(domain=[0, 0.5])),
        color=alt.Color('color:N', scale=alt.Scale(domain=['background', color], range=['#f0f0f0', color])),
        order='color:N'
    ).properties(
        width=300,
        height=150
    )
    
    # Add score indicator line
    indicator_x = [radius * np.cos(score * np.pi), 0]
    indicator_y = [radius * np.sin(score * np.pi), 0]
    
    indicator_data = pd.DataFrame({
        'x': indicator_x,
        'y': indicator_y
    })
    
    indicator = alt.Chart(indicator_data).mark_line(color='black', strokeWidth=2).encode(
        x='x:Q',
        y='y:Q'
    )
    
    # Combine charts
    chart = gauge_chart + indicator
    
    # Display the chart
    st.altair_chart(chart, use_container_width=True)
    
    # Add confidence description based on score
    confidence_description = get_confidence_description(score, label)
    st.markdown(f'<div class="confidence-description">{confidence_description}</div>', unsafe_allow_html=True)
    
    # End container div
    st.markdown('</div>', unsafe_allow_html=True)

def get_confidence_description(score, label):
    """
    Returns a description of the confidence score.
    
    Args:
        score: Confidence score (0.0 to 1.0)
        label: Prediction label ("Real" or "Fake")
    
    Returns:
        A string description of the confidence level
    """
    if label.lower() == "fake":
        if score >= 0.9:
            return "Very high confidence that this is a deepfake."
        elif score >= 0.75:
            return "High confidence that this is a deepfake."
        elif score >= 0.6:
            return "Moderate confidence that this is a deepfake."
        elif score >= 0.5:
            return "Low confidence that this is a deepfake."
        else:
            return "Very low confidence that this is a deepfake."
    else:
        if score >= 0.9:
            return "Very high confidence that this is authentic."
        elif score >= 0.75:
            return "High confidence that this is authentic."
        elif score >= 0.6:
            return "Moderate confidence that this is authentic."
        elif score >= 0.5:
            return "Low confidence that this is authentic."
        else:
            return "Very low confidence that this is authentic."
