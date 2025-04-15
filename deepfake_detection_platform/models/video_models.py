import os
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Union
import time
import asyncio

# Import utility functions
from utils.model_loader import load_model
from utils.preprocessing import extract_faces, extract_audio_from_video

logger = logging.getLogger(__name__)

async def simulate_processing_time(processing_time: float = None):
    """
    Simulate processing time to mimic real model inference.
    
    Args:
        processing_time: Optional specific processing time in seconds. 
                        If None, a random time will be used.
    """
    # Use provided processing time or generate a random one
    if processing_time is None:
        processing_time = random.uniform(2.0, 5.0)  # Video processing usually takes longer
        
    await asyncio.sleep(processing_time)

async def analyze_video(video_data: Dict[str, Any], model: Optional[Any] = None) -> Dict[str, Any]:
    """
    Analyze a video file for deepfake detection.
    
    Args:
        video_data: Dictionary containing preprocessed video data
        
    Returns:
        Dictionary with analysis results
    """
    try:
        # Start timing the analysis
        start_time = time.time()
        logger.info("Starting video analysis")
        
        # Load the video model if not provided
        if model is None:
            model = await load_model("video")
            
        # Simulate model processing time
        # Adjust processing time based on model type if available
        if model and hasattr(model, 'processing_time'):
            processing_time = model.processing_time
        else:
            processing_time = random.uniform(2.0, 5.0)  # Video typically takes longer
            
        await simulate_processing_time(processing_time)
        
        # Generate deepfake detection results based on the provided model
        # In a real implementation, this would use the actual loaded model
        # For this demo, we'll simulate different results based on model ID
        model_id = getattr(model, 'id', None) if model else None
        
        # Generate scores based on model type
        if model_id == "slowfast_dfdc":
            # SlowFast is good at temporal inconsistencies
            fake_score = random.uniform(0.75, 0.95) if random.random() > 0.15 else random.uniform(0.05, 0.3)
        elif model_id == "timesformer_deepfake":
            # TimeSformer is excellent at detecting face swaps in video
            fake_score = random.uniform(0.85, 0.98) if random.random() > 0.1 else random.uniform(0.02, 0.2)
        elif model_id == "convlstm_lippingnet":
            # ConvLSTM specializes in lip sync issues
            fake_score = random.uniform(0.8, 0.97) if random.random() > 0.12 else random.uniform(0.03, 0.25)
        else:
            # Default behavior for unknown models
            fake_score = random.random()
        
        # Classification result
        is_fake = fake_score > 0.5
        classification = "fake" if is_fake else "real"
        confidence = fake_score if is_fake else 1.0 - fake_score
        
        # Feature analysis based on model type
        if model_id == "slowfast_dfdc":
            # SlowFast focuses on spatio-temporal consistency
            feature_scores = {
                "Temporal Consistency": random.uniform(0.8, 1.0) if not is_fake else random.uniform(0.0, 0.3),
                "Facial Movements": random.uniform(0.7, 0.9) if not is_fake else random.uniform(0.1, 0.4),
                "Motion Blur Patterns": random.uniform(0.75, 0.95) if not is_fake else random.uniform(0.05, 0.3),
                "Frame Transitions": random.uniform(0.8, 0.95) if not is_fake else random.uniform(0.05, 0.3),
                "Object Consistency": random.uniform(0.7, 0.9) if not is_fake else random.uniform(0.1, 0.4)
            }
        elif model_id == "timesformer_deepfake":
            # TimeSformer excels at long-term dependencies in video
            feature_scores = {
                "Temporal Consistency": random.uniform(0.85, 1.0) if not is_fake else random.uniform(0.0, 0.2),
                "Facial Movements": random.uniform(0.85, 0.98) if not is_fake else random.uniform(0.02, 0.2),
                "Motion Blur Patterns": random.uniform(0.7, 0.9) if not is_fake else random.uniform(0.1, 0.4),
                "Frame Transitions": random.uniform(0.8, 0.95) if not is_fake else random.uniform(0.05, 0.3),
                "Object Consistency": random.uniform(0.75, 0.95) if not is_fake else random.uniform(0.05, 0.3)
            }
        elif model_id == "convlstm_lippingnet":
            # ConvLSTM LippingNet specializes in audio-visual sync
            feature_scores = {
                "Temporal Consistency": random.uniform(0.7, 0.9) if not is_fake else random.uniform(0.1, 0.4),
                "Facial Movements": random.uniform(0.8, 0.95) if not is_fake else random.uniform(0.05, 0.3),
                "Motion Blur Patterns": random.uniform(0.65, 0.85) if not is_fake else random.uniform(0.15, 0.45),
                "Frame Transitions": random.uniform(0.7, 0.9) if not is_fake else random.uniform(0.1, 0.4),
                "Object Consistency": random.uniform(0.65, 0.85) if not is_fake else random.uniform(0.15, 0.45)
            }
        else:
            # Default random feature scoring for unknown models
            feature_scores = {
                "Temporal Consistency": random.uniform(0.0, 1.0),
                "Facial Movements": random.uniform(0.0, 1.0),
                "Motion Blur Patterns": random.uniform(0.0, 1.0),
                "Frame Transitions": random.uniform(0.0, 1.0),
                "Object Consistency": random.uniform(0.0, 1.0)
            }
            
        # Generate prediction result
        prediction_result = {
            "prediction": classification,
            "confidence": confidence,
            "details": {
                "technical_explanation": f"Video analyzed using {model_id} model",
                "features": feature_scores,
                "analysis_duration": time.time() - start_time
            }
        }
        
        # Calculate analysis duration
        duration = time.time() - start_time
        logger.info(f"Video analysis completed in {duration:.2f} seconds")
        
        # Add duration to the result
        prediction_result["analysis_duration"] = duration
        
        # Add additional information to the details if not already present
        if "details" in prediction_result and isinstance(prediction_result["details"], dict):
            details = prediction_result["details"]
            
            # Add video metadata if available
            if "frame_count" in video_data:
                if "metadata" not in details:
                    details["metadata"] = {}
                
                metadata = details["metadata"]
                metadata["frame_count"] = video_data.get("frame_count", 0)
                metadata["fps"] = video_data.get("fps", 0)
                
                # Calculate duration if possible
                if video_data.get("fps", 0) > 0 and video_data.get("frame_count", 0) > 0:
                    video_duration = video_data["frame_count"] / video_data["fps"]
                    metadata["duration"] = f"{int(video_duration // 60)}:{int(video_duration % 60):02d}"
            
            # Add temporal analysis (frame-by-frame predictions)
            if "temporal_analysis" not in details:
                # Generate frame-by-frame confidence scores
                frame_scores = generate_frame_scores(
                    processed_frames.shape[0],
                    prediction_result.get("prediction", ""),
                    prediction_result.get("confidence", 0.5)
                )
                details["temporal_analysis"] = frame_scores
        
        return prediction_result
        
    except Exception as e:
        logger.error(f"Error during video analysis: {str(e)}", exc_info=True)
        
        # Return a fallback result in case of error
        return {
            "prediction": "Error",
            "confidence": 0.0,
            "details": {
                "technical_explanation": f"An error occurred during analysis: {str(e)}",
                "features": {},
                "error": str(e)
            }
        }

def generate_frame_scores(num_frames: int, prediction: str, overall_confidence: float) -> Dict[str, Any]:
    """
    Generate synthetic frame-by-frame confidence scores for visualization.
    
    Args:
        num_frames: Number of frames
        prediction: Overall prediction label
        overall_confidence: Overall confidence score
        
    Returns:
        Dictionary with frame scores and analysis
    """
    import random
    
    # Set seed for reproducible randomness based on the prediction and confidence
    seed = int(hash(prediction) % 1000000) + int(overall_confidence * 1000)
    random.seed(seed)
    
    # Generate base frame scores centered around the overall confidence
    is_fake = prediction.lower() == "fake"
    
    # Frame indices (0 to num_frames-1)
    frame_indices = list(range(num_frames))
    
    # Base confidence level with random variation
    base_level = overall_confidence
    
    # Amount of random variation depends on the confidence level
    # Less confident predictions have more variation
    variation = 0.15 - (abs(overall_confidence - 0.5) * 0.1)
    
    # Generate random variations around the base level
    variations = [random.uniform(-variation, variation) for _ in range(num_frames)]
    
    # Add some temporal coherence (smooth variations)
    smoothed_variations = []
    window_size = 5
    
    for i in range(num_frames):
        # Calculate average of nearby variations for smoothing
        window_start = max(0, i - window_size // 2)
        window_end = min(num_frames, i + window_size // 2 + 1)
        smoothed = sum(variations[window_start:window_end]) / (window_end - window_start)
        smoothed_variations.append(smoothed)
    
    # Calculate frame scores
    frame_scores = [max(0.05, min(0.95, base_level + var)) for var in smoothed_variations]
    
    # For fake predictions, add some "artifact spikes"
    if is_fake and overall_confidence > 0.6:
        # Add 2-4 spikes where the fakeness is more apparent
        num_spikes = random.randint(2, 4)
        for _ in range(num_spikes):
            spike_pos = random.randint(0, num_frames - 1)
            spike_width = random.randint(3, 8)
            
            # Create a local spike around the position
            for i in range(max(0, spike_pos - spike_width), min(num_frames, spike_pos + spike_width + 1)):
                distance = abs(i - spike_pos)
                if distance < spike_width:
                    # Increase the frame score, with more increase closer to the spike center
                    boost = 0.2 * (1 - distance / spike_width)
                    frame_scores[i] = min(0.95, frame_scores[i] + boost)
    
    # For real predictions with high confidence, smooth out the scores even more
    elif not is_fake and overall_confidence > 0.7:
        # Apply additional smoothing
        for _ in range(2):  # Apply smoothing twice
            new_scores = frame_scores.copy()
            for i in range(1, num_frames - 1):
                new_scores[i] = (frame_scores[i-1] + frame_scores[i] + frame_scores[i+1]) / 3
            frame_scores = new_scores
    
    # Find keyframes (local maxima and minima)
    keyframes = []
    
    if num_frames > 10:
        for i in range(2, num_frames - 2):
            # Check if this is a local maximum
            if (frame_scores[i] > frame_scores[i-1] and 
                frame_scores[i] > frame_scores[i-2] and 
                frame_scores[i] > frame_scores[i+1] and 
                frame_scores[i] > frame_scores[i+2]):
                keyframes.append({
                    "frame": i,
                    "score": frame_scores[i],
                    "type": "peak" if is_fake else "authentic"
                })
            
            # Check if this is a local minimum
            elif (frame_scores[i] < frame_scores[i-1] and 
                  frame_scores[i] < frame_scores[i-2] and 
                  frame_scores[i] < frame_scores[i+1] and 
                  frame_scores[i] < frame_scores[i+2]):
                keyframes.append({
                    "frame": i,
                    "score": frame_scores[i],
                    "type": "authentic" if is_fake else "peak"
                })
    
    # Keep only the most significant keyframes (highest/lowest scores)
    keyframes.sort(key=lambda x: abs(x["score"] - 0.5), reverse=True)
    max_keyframes = 5
    keyframes = keyframes[:max_keyframes]
    
    # Sort keyframes by frame number for display
    keyframes.sort(key=lambda x: x["frame"])
    
    return {
        "frame_indices": frame_indices,
        "frame_scores": frame_scores,
        "keyframes": keyframes,
        "average_score": sum(frame_scores) / len(frame_scores) if frame_scores else 0
    }

async def analyze_video_with_audio(video_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze a video file using both visual and audio features.
    
    Args:
        video_data: Dictionary containing preprocessed video data
        
    Returns:
        Dictionary with analysis results
    """
    try:
        # 1. Analyze video frames
        video_result = await analyze_video(video_data)
        
        # 2. Extract and analyze audio if available
        audio_result = None
        video_path = video_data.get("original_path")
        
        if video_path and os.path.exists(video_path):
            # Extract audio from video
            audio_path = extract_audio_from_video(video_path)
            
            if audio_path and os.path.exists(audio_path):
                try:
                    # Import the audio analysis function
                    from models.audio_models import analyze_audio
                    
                    # Preprocess the audio
                    from utils.preprocessing import preprocess_audio
                    audio_data = await preprocess_audio(audio_path)
                    
                    # Analyze the audio
                    audio_result = await analyze_audio(audio_data)
                    
                    # Clean up the temporary audio file
                    os.remove(audio_path)
                    
                except Exception as e:
                    logger.error(f"Error analyzing audio from video: {str(e)}", exc_info=True)
        
        # 3. Combine results if audio analysis was successful
        if audio_result and isinstance(audio_result, dict) and "prediction" in audio_result:
            # Combine video and audio results
            combined_result = combine_results(video_result, audio_result)
            return combined_result
        else:
            # Return only video results
            return video_result
            
    except Exception as e:
        logger.error(f"Error during combined video/audio analysis: {str(e)}", exc_info=True)
        return await analyze_video(video_data)  # Fallback to video-only analysis

def combine_results(video_result: Dict[str, Any], audio_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine video and audio analysis results.
    
    Args:
        video_result: Dictionary with video analysis results
        audio_result: Dictionary with audio analysis results
        
    Returns:
        Dictionary with combined results
    """
    # Get predictions and confidence scores
    video_pred = video_result.get("prediction", "Unknown")
    audio_pred = audio_result.get("prediction", "Unknown")
    
    video_conf = video_result.get("confidence", 0.5)
    audio_conf = audio_result.get("confidence", 0.5)
    
    # Determine overall prediction based on confidence-weighted voting
    # Give more weight to video prediction (70% video, 30% audio)
    video_weight = 0.7
    audio_weight = 0.3
    
    # Calculate scores for "Fake" class
    fake_score = 0
    if video_pred.lower() == "fake":
        fake_score += video_weight * video_conf
    else:
        fake_score += video_weight * (1 - video_conf)
        
    if audio_pred.lower() == "fake":
        fake_score += audio_weight * audio_conf
    else:
        fake_score += audio_weight * (1 - audio_conf)
    
    # Determine final prediction and confidence
    if fake_score > 0.5:
        combined_pred = "Fake"
        combined_conf = fake_score
    else:
        combined_pred = "Real"
        combined_conf = 1 - fake_score
    
    # Combine details
    combined_details = {}
    
    if "details" in video_result and isinstance(video_result["details"], dict):
        combined_details = video_result["details"]
    
    if "details" in audio_result and isinstance(audio_result["details"], dict):
        # Add audio-specific details
        if "technical_explanation" in combined_details and "technical_explanation" in audio_result["details"]:
            combined_details["technical_explanation"] += "\n\n**Audio Analysis:**\n" + audio_result["details"]["technical_explanation"]
        
        # Combine features
        if "features" in combined_details and "features" in audio_result["details"]:
            # Prefix audio features to avoid name collisions
            audio_features = {f"Audio: {k}": v for k, v in audio_result["details"]["features"].items()}
            combined_details["features"].update(audio_features)
        
        # Add audio-specific metadata
        if "metadata" in audio_result["details"]:
            if "metadata" not in combined_details:
                combined_details["metadata"] = {}
            
            # Prefix audio metadata to avoid name collisions
            audio_metadata = {f"Audio: {k}": v for k, v in audio_result["details"]["metadata"].items()}
            combined_details["metadata"].update(audio_metadata)
        
        # Add waveform data if available
        if "waveform_data" in audio_result["details"]:
            combined_details["audio_waveform"] = audio_result["details"]["waveform_data"]
        
        # Add spectral data if available
        if "spectral_data" in audio_result["details"]:
            combined_details["audio_spectral"] = audio_result["details"]["spectral_data"]
    
    # Create combined result
    combined_result = {
        "prediction": combined_pred,
        "confidence_score": combined_conf,
        "details": combined_details,
        "video_result": {
            "prediction": video_pred,
            "confidence": video_conf
        },
        "audio_result": {
            "prediction": audio_pred,
            "confidence": audio_conf
        }
    }
    
    # Add analysis duration (sum of video and audio analysis times)
    v_duration = video_result.get("analysis_duration", 0)
    a_duration = audio_result.get("analysis_duration", 0)
    combined_result["analysis_duration"] = v_duration + a_duration
    
    return combined_result
