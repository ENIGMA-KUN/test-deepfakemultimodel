import streamlit as st
import os
import io
import base64
import json
import tempfile
from datetime import datetime
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# For PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
    from reportlab.platypus import Table, TableStyle, PageBreak, ListFlowable, ListItem
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
except ImportError:
    st.warning("ReportLab is not installed. PDF reports will not be available.")
    
def generate_pdf_report(result_data: Dict[str, Any]) -> Optional[str]:
    """
    Generates a PDF report based on analysis results.
    
    Args:
        result_data: Dictionary containing analysis results
    
    Returns:
        Path to the generated PDF file
    """
    try:
        # Check if reportlab is available
        if 'reportlab.platypus' not in sys.modules:
            st.warning("ReportLab is not installed. Please install it to generate PDF reports.")
            return None
        
        # Create a temporary file for the PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            pdf_path = tmp_file.name
        
        # Get report elements
        elements = _create_report_elements(result_data)
        
        # Create the PDF document
        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Build the document
        doc.build(elements)
        
        return pdf_path
    
    except Exception as e:
        st.error(f"Error generating PDF report: {str(e)}")
        
        # For demonstration, generate a simple report if PDF generation fails
        return _generate_fallback_report(result_data)
        
def _create_report_elements(result_data: Dict[str, Any]) -> list:
    """
    Creates PDF report elements using ReportLab.
    
    Args:
        result_data: Dictionary containing analysis results
    
    Returns:
        List of reportlab elements for the PDF
    """
    # Get styles
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='Title',
        parent=styles['Heading1'],
        fontSize=20,
        alignment=TA_CENTER,
        spaceAfter=12
    ))
    styles.add(ParagraphStyle(
        name='Subtitle',
        parent=styles['Heading2'],
        fontSize=16,
        alignment=TA_CENTER,
        spaceAfter=12
    ))
    styles.add(ParagraphStyle(
        name='SectionTitle',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        name='Normal',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        spaceAfter=6
    ))
    
    # List to hold the PDF elements
    elements = []
    
    # Title
    elements.append(Paragraph("Deepfake Detection Analysis Report", styles['Title']))
    
    # Current date
    current_date = datetime.now().strftime("%B %d, %Y %H:%M")
    elements.append(Paragraph(f"Generated on: {current_date}", styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Summary section
    elements.append(Paragraph("Analysis Summary", styles['SectionTitle']))
    elements.append(Spacer(1, 6))
    
    # Create a summary table
    data = [
        ["File Name:", result_data.get("filename", "Unknown")],
        ["Media Type:", result_data.get("media_type", "Unknown").capitalize()],
        ["Analysis Date:", result_data.get("timestamp", current_date)],
        ["Analysis Duration:", f"{result_data.get('analysis_duration', 0):.2f} seconds"],
        ["Result:", result_data.get("prediction", "Unknown")],
        ["Confidence:", f"{result_data.get('confidence_score', 0) * 100:.1f}%"]
    ]
    
    table = Table(data, colWidths=[1.5*inch, 4*inch])
    table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (0, 4), (1, 4), 'Helvetica-Bold'),  # Make the result row bold
        ('BACKGROUND', (1, 4), (1, 4), 
            colors.pink if result_data.get("prediction", "").lower() == "fake" else colors.lightgreen),
    ]))
    
    elements.append(table)
    elements.append(Spacer(1, 12))
    
    # Add media preview if available
    if "media_content" in result_data and result_data.get("media_type") == "image":
        elements.append(Paragraph("Media Preview", styles['SectionTitle']))
        try:
            img_data = base64.b64decode(result_data["media_content"])
            
            # Create a temporary file for the image
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(img_data)
                img_path = tmp_file.name
            
            # Add image to report
            img = RLImage(img_path, width=6*inch, height=4*inch)
            elements.append(img)
            
            # Clean up the temporary file
            os.unlink(img_path)
        except:
            elements.append(Paragraph("Image preview not available", styles['Normal']))
        
        elements.append(Spacer(1, 12))
    
    # Technical Details Section
    elements.append(Paragraph("Technical Analysis", styles['SectionTitle']))
    
    if "detection_details" in result_data and "technical_explanation" in result_data["detection_details"]:
        explanation_text = result_data["detection_details"]["technical_explanation"]
        elements.append(Paragraph(explanation_text, styles['Normal']))
    else:
        elements.append(Paragraph("No technical analysis details available.", styles['Normal']))
    
    elements.append(Spacer(1, 12))
    
    # Features Section
    if "detection_details" in result_data and "features" in result_data["detection_details"]:
        elements.append(Paragraph("Detection Features", styles['SectionTitle']))
        elements.append(Spacer(1, 6))
        
        features = result_data["detection_details"]["features"]
        
        # Create a table for the features
        feature_data = [["Feature", "Score"]]
        for feature, score in features.items():
            feature_data.append([feature, f"{score:.2f}"])
        
        feature_table = Table(feature_data, colWidths=[3*inch, 1.5*inch])
        feature_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ]))
        
        elements.append(feature_table)
        elements.append(Spacer(1, 12))
    
    # Recommendations Section
    elements.append(Paragraph("Recommendations", styles['SectionTitle']))
    
    recommendation_text = ""
    if result_data.get("prediction", "").lower() == "fake":
        recommendation_text = """
        Based on our analysis, this content appears to be AI-generated or manipulated. We recommend:
        
        1. Do not share this content as authentic.
        2. If you received this content from someone claiming it is real, be cautious of potential misinformation.
        3. Consider the source of the content and verify with additional sources if possible.
        4. Be aware that deepfake technology continues to improve, making detection increasingly challenging.
        """
    else:
        recommendation_text = """
        Based on our analysis, this content appears to be authentic. However:
        
        1. No detection system is 100% accurate. Consider the confidence score in your decision-making.
        2. If you have additional reasons to believe this content may be manipulated, seek verification from multiple sources.
        3. Remember that context matters — even authentic media can be presented in misleading ways.
        """
    
    elements.append(Paragraph(recommendation_text, styles['Normal']))
    
    # Disclaimer
    elements.append(Spacer(1, 24))
    elements.append(Paragraph("Disclaimer", styles['SectionTitle']))
    
    disclaimer_text = """
    This analysis is provided for informational purposes only. Our deepfake detection technology, while state-of-the-art, 
    cannot guarantee 100% accuracy. The results should be considered as probabilistic rather than definitive. 
    This report should not be used as the sole basis for making important decisions or determinations about the content's authenticity.
    """
    
    elements.append(Paragraph(disclaimer_text, styles['Normal']))
    
    return elements

def _generate_fallback_report(result_data: Dict[str, Any]) -> str:
    """
    Generates a simple text-based report when PDF generation fails.
    
    Args:
        result_data: Dictionary containing analysis results
    
    Returns:
        Path to the text report file
    """
    # Create a temporary file for the text report
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_file:
        report_path = tmp_file.name
        
        # Write report content
        current_date = datetime.now().strftime("%B %d, %Y %H:%M")
        
        report_content = [
            "Deepfake Detection Analysis Report",
            f"Generated on: {current_date}",
            "",
            "Analysis Summary",
            "----------------",
            f"File Name: {result_data.get('filename', 'Unknown')}",
            f"Media Type: {result_data.get('media_type', 'Unknown').capitalize()}",
            f"Analysis Date: {result_data.get('timestamp', current_date)}",
            f"Result: {result_data.get('prediction', 'Unknown')}",
            f"Confidence: {result_data.get('confidence_score', 0) * 100:.1f}%",
            "",
        ]
        
        # Add technical details if available
        if "detection_details" in result_data and "technical_explanation" in result_data["detection_details"]:
            report_content.extend([
                "Technical Analysis",
                "-----------------",
                result_data["detection_details"]["technical_explanation"],
                "",
            ])
        
        # Add features if available
        if "detection_details" in result_data and "features" in result_data["detection_details"]:
            report_content.extend([
                "Detection Features",
                "-----------------",
            ])
            
            features = result_data["detection_details"]["features"]
            for feature, score in features.items():
                report_content.append(f"{feature}: {score:.2f}")
            
            report_content.append("")
        
        # Add recommendations
        report_content.extend([
            "Recommendations",
            "--------------",
        ])
        
        if result_data.get("prediction", "").lower() == "fake":
            report_content.extend([
                "Based on our analysis, this content appears to be AI-generated or manipulated. We recommend:",
                "1. Do not share this content as authentic.",
                "2. If you received this content from someone claiming it is real, be cautious of potential misinformation.",
                "3. Consider the source of the content and verify with additional sources if possible.",
                "4. Be aware that deepfake technology continues to improve, making detection increasingly challenging.",
                "",
            ])
        else:
            report_content.extend([
                "Based on our analysis, this content appears to be authentic. However:",
                "1. No detection system is 100% accurate. Consider the confidence score in your decision-making.",
                "2. If you have additional reasons to believe this content may be manipulated, seek verification from multiple sources.",
                "3. Remember that context matters — even authentic media can be presented in misleading ways.",
                "",
            ])
        
        # Add disclaimer
        report_content.extend([
            "Disclaimer",
            "----------",
            "This analysis is provided for informational purposes only. Our deepfake detection technology, while state-of-the-art,",
            "cannot guarantee 100% accuracy. The results should be considered as probabilistic rather than definitive.",
            "This report should not be used as the sole basis for making important decisions or determinations about the content's authenticity.",
        ])
        
        # Write the report content to the file
        tmp_file.write("\n".join(report_content).encode('utf-8'))
    
    return report_path

def export_result_as_json(result_data: Dict[str, Any]) -> str:
    """
    Exports the analysis result as a JSON file.
    
    Args:
        result_data: Dictionary containing analysis results
    
    Returns:
        JSON string representation of the results
    """
    # Create a copy of the data to modify for export
    export_data = result_data.copy()
    
    # Remove large binary data if present to keep the JSON file manageable
    if "media_content" in export_data:
        del export_data["media_content"]
    
    # Add export timestamp
    export_data["export_timestamp"] = datetime.now().isoformat()
    
    # Convert to JSON string with nice formatting
    json_str = json.dumps(export_data, indent=2)
    
    return json_str

def save_report_to_file(report_content: str, filename: str) -> str:
    """
    Saves report content to a file.
    
    Args:
        report_content: Report content to save
        filename: Base filename (without extension)
    
    Returns:
        Path to the saved file
    """
    # Create a unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_filename = f"{filename}_{timestamp}.txt"
    
    # Save to the results directory
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    
    # Create the directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    file_path = os.path.join(results_dir, full_filename)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return file_path
