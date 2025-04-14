from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import uuid


class ResultQuery(BaseModel):
    task_id: Optional[str] = Field(None, description="Task ID from detection request")
    result_id: Optional[uuid.UUID] = Field(None, description="Result ID from previous query")
    
    class Config:
        schema_extra = {
            "example": {
                "task_id": "abcd1234-ef56-7890-abcd-1234ef567890",
            }
        }


class ResultStatus(BaseModel):
    status: str = Field(..., description="Status of the detection process")
    progress: float = Field(..., description="Progress as a percentage")
    message: Optional[str] = Field(None, description="Additional status message")
    result_id: Optional[uuid.UUID] = Field(None, description="Result ID if complete")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "processing",
                "progress": 65.5,
                "message": "Processing video frames",
                "result_id": None
            }
        }


class HeatmapData(BaseModel):
    url: str = Field(..., description="URL to the heatmap image")
    width: int = Field(..., description="Width of the image")
    height: int = Field(..., description="Height of the image")
    regions: List[Dict[str, Any]] = Field(..., description="Regions of interest with scores")


class TemporalData(BaseModel):
    timestamps: List[float] = Field(..., description="Timestamps for time-series data")
    values: List[float] = Field(..., description="Values for each timestamp")
    threshold: float = Field(..., description="Threshold line value")


class SpectralData(BaseModel):
    url: str = Field(..., description="URL to the spectral visualization image")
    splice_times: List[float] = Field(..., description="Time points of potential splices")
    threshold: float = Field(..., description="Threshold used for detection")


class VoiceConsistencyData(BaseModel):
    url: str = Field(..., description="URL to the voice consistency visualization")
    segment_diffs: List[float] = Field(..., description="Differences between consecutive segments")
    mean_diff: float = Field(..., description="Mean difference value")


class SilenceData(BaseModel):
    url: str = Field(..., description="URL to the silence visualization")
    segments: List[Dict[str, float]] = Field(..., description="Silence segments with timing info")
    total_duration: float = Field(..., description="Total duration of silence")


class ConfidenceGaugeData(BaseModel):
    url: str = Field(..., description="URL to the confidence gauge visualization")
    score: float = Field(..., description="Confidence score visualized")


class VisualizationData(BaseModel):
    heatmap: Optional[HeatmapData] = Field(None, description="Heatmap visualization data")
    temporal: Optional[TemporalData] = Field(None, description="Temporal analysis data")
    frequency: Optional[Dict[str, Any]] = Field(None, description="Frequency analysis data")
    spectral_discontinuity: Optional[SpectralData] = Field(None, description="Spectral discontinuity visualization")
    voice_consistency: Optional[VoiceConsistencyData] = Field(None, description="Voice consistency visualization")
    silence_analysis: Optional[SilenceData] = Field(None, description="Silence segments visualization")
    confidence_gauge: Optional[ConfidenceGaugeData] = Field(None, description="Confidence gauge visualization")


class DetailedResult(BaseModel):
    id: uuid.UUID = Field(..., description="Unique identifier for the result")
    is_fake: bool = Field(..., description="Whether the media is detected as fake")
    confidence_score: float = Field(..., description="Overall confidence score")
    media_type: str = Field(..., description="Type of media that was analyzed")
    
    detection_details: Dict[str, Any] = Field(..., description="Detailed detection information")
    models_used: Dict[str, str] = Field(..., description="Models used for the detection")
    visualizations: Optional[VisualizationData] = Field(None, description="Visualization data")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "abcd1234-ef56-7890-abcd-1234ef567890",
                "is_fake": True,
                "confidence_score": 0.92,
                "media_type": "image",
                "detection_details": {
                    "face_manipulation_score": 0.95,
                    "inconsistency_score": 0.88,
                    "artifact_score": 0.91
                },
                "models_used": {
                    "primary": "XceptionNet",
                    "secondary": "EfficientNet-B4"
                }
            }
        }
