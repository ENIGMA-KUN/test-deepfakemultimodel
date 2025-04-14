import os
import uuid
from celery import Task
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.db.models import DetectionResult
from app.models.ensemble import ensemble_detection
from app.utils.visualization import generate_temporal_visualization
from app.tasks.celery_app import celery_app


class SQLAlchemyTask(Task):
    """Base Task with SQLAlchemy session handling."""
    _session = None
    
    def after_return(self, *args, **kwargs):
        if self._session is not None:
            self._session.close()
    
    @property
    def session(self) -> Session:
        if self._session is None:
            self._session = SessionLocal()
        return self._session


@celery_app.task(base=SQLAlchemyTask, bind=True)
def detect_audio(self, audio_path: str, file_hash: str, detailed: bool = False, confidence_threshold: float = 0.5):
    """
    Process an audio file for deepfake detection.
    
    Args:
        self: Task instance (injected by Celery)
        audio_path (str): Path to the audio file
        file_hash (str): Hash of the file for identification
        detailed (bool): Whether to perform detailed analysis
        confidence_threshold (float): Threshold for detection confidence
    
    Returns:
        str: ID of the result record
    """
    try:
        # Update task status
        self.update_state(state="PROGRESS", meta={"progress": 10, "message": "Starting audio analysis"})
        
        # Perform comprehensive audio analysis if detailed
        if detailed:
            self.update_state(state="PROGRESS", meta={"progress": 15, "message": "Running comprehensive audio analysis"})
            from app.preprocessing.audio_preprocessing import comprehensive_audio_analysis
            comprehensive_results = comprehensive_audio_analysis(audio_path)
        
        # Perform ensemble detection
        self.update_state(state="PROGRESS", meta={"progress": 20, "message": "Running detection models"})
        result = ensemble_detection(audio_path, "audio", detailed, confidence_threshold)
        
        self.update_state(state="PROGRESS", meta={"progress": 60, "message": "Analyzing results"})
        
        # Generate visualizations if detailed analysis
        visualization_paths = {}
        
        if detailed and not result.get("error"):
            self.update_state(state="PROGRESS", meta={"progress": 70, "message": "Generating visualizations"})
            
            try:
                # Import visualization functions
                from app.utils.visualization import (
                    generate_temporal_visualization,
                    generate_spectral_discontinuity_visualization,
                    generate_voice_consistency_visualization,
                    generate_silence_segments_visualization,
                    generate_confidence_gauge
                )
                
                # Directory for visualizations
                os.makedirs("visualizations", exist_ok=True)
                
                # 1. Temporal analysis visualization
                if "temporal_analysis" in result.get("detection_details", {}):
                    temporal_data = result["detection_details"]["temporal_analysis"]
                    
                    visualization_paths["temporal_analysis"] = generate_temporal_visualization(
                        temporal_data["timestamps"],
                        temporal_data["scores"],
                        temporal_data["threshold"],
                        "Audio Deepfake Confidence Over Time",
                        os.path.join("visualizations", f"temporal_{uuid.uuid4().hex}.png")
                    )
                
                # 2. Confidence gauge visualization
                if "combined_confidence_score" in result:
                    confidence = result["combined_confidence_score"]
                else:
                    confidence = result["confidence_score"]
                
                visualization_paths["confidence_gauge"] = generate_confidence_gauge(
                    confidence,
                    os.path.join("visualizations", f"confidence_{uuid.uuid4().hex}.png")
                )
                
                # 3. Spectral discontinuity visualization if available
                comprehensive_details = result.get("detection_details", {}).get("comprehensive_analysis", {})
                if comprehensive_details and "spectral_analysis" in comprehensive_details:
                    spectral_data = comprehensive_details["spectral_analysis"]
                    if "splice_times" in spectral_data and spectral_data["splice_times"]:
                        visualization_paths["spectral_discontinuity"] = generate_spectral_discontinuity_visualization(
                            audio_path,
                            spectral_data["splice_times"],
                            spectral_data.get("threshold", 0.5),
                            os.path.join("visualizations", f"spectral_{uuid.uuid4().hex}.png")
                        )
                
                # 4. Voice consistency visualization if available
                if comprehensive_details and "voice_consistency" in comprehensive_details:
                    vc_data = comprehensive_details["voice_consistency"]
                    if "segment_diffs" in vc_data:
                        # Create timestamps (just indices for segment differences)
                        segment_diff_timestamps = list(range(len(vc_data["segment_diffs"])))
                        
                        visualization_paths["voice_consistency"] = generate_voice_consistency_visualization(
                            segment_diff_timestamps,
                            vc_data["segment_diffs"],
                            vc_data.get("mean_segment_diff", 0),
                            os.path.join("visualizations", f"voice_consist_{uuid.uuid4().hex}.png")
                        )
                
                # 5. Silence segments visualization if available
                if comprehensive_details and "silence_analysis" in comprehensive_details:
                    silence_data = comprehensive_details["silence_analysis"]
                    if "silence_segments" in silence_data and silence_data["silence_segments"]:
                        visualization_paths["silence_analysis"] = generate_silence_segments_visualization(
                            audio_path,
                            silence_data["silence_segments"],
                            os.path.join("visualizations", f"silence_{uuid.uuid4().hex}.png")
                        )
                
            except Exception as e:
                self.update_state(state="PROGRESS", meta={"progress": 75, "message": f"Visualization error: {str(e)}"})
        
        # Create database record
        self.update_state(state="PROGRESS", meta={"progress": 80, "message": "Saving results"})
        
        db_result = DetectionResult(
            id=uuid.uuid4(),
            file_hash=file_hash,
            file_path=audio_path,
            media_type="audio",
            is_fake=result["is_fake"],
            confidence_score=result["confidence_score"],
            detection_details=result.get("detection_details", {}),
            models_used=result.get("models_used", {}),
            temporal_analysis_path=visualization_paths.get("temporal_analysis")
        )
        
        # Add visualization paths to detection details
        if visualization_paths and hasattr(db_result, "detection_details") and db_result.detection_details:
            db_result.detection_details["visualization_paths"] = visualization_paths
        
        self.session.add(db_result)
        self.session.commit()
        
        self.update_state(state="PROGRESS", meta={"progress": 100, "message": "Analysis complete"})
        
        return str(db_result.id)
    
    except Exception as e:
        # Log the error
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error in detect_audio task: {str(e)}")
        
        # Create error record in database
        try:
            db_result = DetectionResult(
                id=uuid.uuid4(),
                file_hash=file_hash,
                file_path=audio_path,
                media_type="audio",
                is_fake=False,
                confidence_score=0.0,
                detection_details={"error": str(e)},
                models_used={}
            )
            
            self.session.add(db_result)
            self.session.commit()
            
            return str(db_result.id)
        except Exception as db_error:
            logger.error(f"Error saving failure record: {str(db_error)}")
            raise e
