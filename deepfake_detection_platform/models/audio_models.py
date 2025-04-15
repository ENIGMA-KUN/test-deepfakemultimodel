import os
import numpy as np
import random
import logging
import time
import asyncio
from typing import Dict, Any, Optional, Union

# Import utility functions
from utils.model_loader import load_model

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
        processing_time = random.uniform(1.0, 3.0)  # Audio processing usually takes longer
        
    await asyncio.sleep(processing_time)

async def analyze_audio(audio_data: Dict[str, Any], model: Optional[Any] = None) -> Dict[str, Any]:
    """
    Analyze an audio file for deepfake detection.
    
    Args:
        audio_data: Dictionary containing preprocessed audio data
        
    Returns:
        Dictionary with analysis results
    """
    try:
        # Start timing the analysis
        start_time = time.time()
        logger.info("Starting audio analysis")
        
        # Load the audio model if not provided
        if model is None:
            model = await load_model("audio")
            
        # Simulate model processing time
        # Adjust processing time based on model type if available
        if model and hasattr(model, 'processing_time'):
            processing_time = model.processing_time
        else:
            processing_time = random.uniform(1.0, 3.0)  # Audio typically takes longer
            
        await simulate_processing_time(processing_time)
        
        # Generate deepfake detection results based on the provided model
        # In a real implementation, this would use the actual loaded model
        # For this demo, we'll simulate different results based on model ID
        model_id = getattr(model, 'id', None) if model else None
        
        # Generate scores based on model type
        if model_id == "wavlm_asvspoof":
            # WavLM is very good at detecting voice cloning
            fake_score = random.uniform(0.8, 0.98) if random.random() > 0.15 else random.uniform(0.02, 0.2)
        elif model_id == "lcnn_rawnet2":
            # LCNN is good at synthetic speech detection
            fake_score = random.uniform(0.75, 0.95) if random.random() > 0.15 else random.uniform(0.05, 0.3)
        elif model_id == "mel_spectrogram_lstm":
            # Mel-spectrogram model is better at detecting frequency artifacts
            fake_score = random.uniform(0.7, 0.9) if random.random() > 0.2 else random.uniform(0.1, 0.4)
        else:
            # Default behavior for unknown models
            fake_score = random.random()
        
        # Classification result
        is_fake = fake_score > 0.5
        classification = "fake" if is_fake else "real"
        confidence = fake_score if is_fake else 1.0 - fake_score
        
        # Feature analysis based on model type
        if model_id == "wavlm_asvspoof":
            # WavLM excels at temporal consistency and prosody detection
            feature_scores = {
                "Voice Consistency": random.uniform(0.8, 1.0) if not is_fake else random.uniform(0.0, 0.3),
                "Prosody Features": random.uniform(0.85, 0.98) if not is_fake else random.uniform(0.05, 0.3),
                "Spectral Features": random.uniform(0.7, 0.9) if not is_fake else random.uniform(0.1, 0.4),
                "Temporal Patterns": random.uniform(0.8, 0.95) if not is_fake else random.uniform(0.0, 0.25),
                "Phoneme Transitions": random.uniform(0.75, 0.95) if not is_fake else random.uniform(0.05, 0.3)
            }
        elif model_id == "lcnn_rawnet2":
            # LCNN/RawNet2 focuses on raw waveform analysis
            feature_scores = {
                "Voice Consistency": random.uniform(0.7, 0.9) if not is_fake else random.uniform(0.1, 0.4),
                "Prosody Features": random.uniform(0.65, 0.85) if not is_fake else random.uniform(0.15, 0.45),
                "Spectral Features": random.uniform(0.85, 1.0) if not is_fake else random.uniform(0.0, 0.2),
                "Temporal Patterns": random.uniform(0.7, 0.9) if not is_fake else random.uniform(0.1, 0.4),
                "Phoneme Transitions": random.uniform(0.65, 0.85) if not is_fake else random.uniform(0.15, 0.45)
            }
        elif model_id == "mel_spectrogram_lstm":
            # Mel-spectrogram LSTM specializes in frequency domain analysis
            feature_scores = {
                "Voice Consistency": random.uniform(0.6, 0.85) if not is_fake else random.uniform(0.15, 0.45),
                "Prosody Features": random.uniform(0.7, 0.9) if not is_fake else random.uniform(0.1, 0.4),
                "Spectral Features": random.uniform(0.8, 1.0) if not is_fake else random.uniform(0.0, 0.2),
                "Temporal Patterns": random.uniform(0.75, 0.95) if not is_fake else random.uniform(0.05, 0.3),
                "Phoneme Transitions": random.uniform(0.65, 0.9) if not is_fake else random.uniform(0.1, 0.4)
            }
        else:
            # Default random feature scoring for unknown models
            feature_scores = {
                "Voice Consistency": random.uniform(0.0, 1.0),
                "Prosody Features": random.uniform(0.0, 1.0),
                "Spectral Features": random.uniform(0.0, 1.0),
                "Temporal Patterns": random.uniform(0.0, 1.0),
                "Phoneme Transitions": random.uniform(0.0, 1.0)
            }
            
        # Generate prediction result
        prediction_result = {
            "prediction": classification,
            "confidence": confidence,
            "details": {
                "technical_explanation": f"Audio analyzed using {model_id} model",
                "features": feature_scores,
                "analysis_duration": time.time() - start_time
            }
        }
        
        # Calculate analysis duration
        duration = time.time() - start_time
        logger.info(f"Audio analysis completed in {duration:.2f} seconds")
        
        # Add duration to the result
        prediction_result["analysis_duration"] = duration
        
        # Add additional information to the details if not already present
        if "details" in prediction_result and isinstance(prediction_result["details"], dict):
            details = prediction_result["details"]
            
            # Add audio information if available
            if "sample_rate" in audio_data:
                if "metadata" not in details:
                    details["metadata"] = {}
                
                details["metadata"]["sample_rate"] = f"{audio_data['sample_rate']} Hz"
                
                # Add waveform data for visualization
                if "waveform" in audio_data and "waveform_data" not in details:
                    # Use a downsampled version of the waveform for visualization
                    waveform = audio_data["waveform"]
                    downsampled = downsample_signal(waveform, max_points=1000)
                    details["waveform_data"] = downsampled.tolist()
                    
                # Add spectral analysis for visualization
                if "waveform" in audio_data and "spectral_data" not in details:
                    spectral_data = generate_spectral_features(
                        audio_data["waveform"], 
                        audio_data["sample_rate"]
                    )
                    details["spectral_data"] = spectral_data
        
        return prediction_result
        
    except Exception as e:
        logger.error(f"Error during audio analysis: {str(e)}", exc_info=True)
        
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

def downsample_signal(signal: np.ndarray, max_points: int = 1000) -> np.ndarray:
    """
    Downsample a signal to a maximum number of points for visualization.
    
    Args:
        signal: Input signal as numpy array
        max_points: Maximum number of points in the output
        
    Returns:
        Downsampled signal
    """
    # If signal is already smaller than max_points, return as is
    if len(signal) <= max_points:
        return signal
    
    # Calculate downsampling factor
    factor = len(signal) // max_points
    
    # Downsample by taking every nth point
    downsampled = signal[::factor]
    
    # Trim to exactly max_points
    return downsampled[:max_points]

def generate_spectral_features(signal: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    """
    Generate spectral features from an audio signal for visualization.
    
    Args:
        signal: Audio signal as numpy array
        sample_rate: Sample rate of the audio
        
    Returns:
        Dictionary with spectral features
    """
    try:
        # Try to use librosa for spectral analysis
        try:
            import librosa
            import librosa.display
            
            # Calculate spectral features
            # 1. Mel spectrogram (downsampled for visualization)
            n_fft = 2048
            hop_length = 512
            n_mels = 128
            
            mel_spec = librosa.feature.melspectrogram(
                y=signal, 
                sr=sample_rate, 
                n_fft=n_fft, 
                hop_length=hop_length, 
                n_mels=n_mels
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # 2. Spectral contrast
            contrast = librosa.feature.spectral_contrast(
                y=signal, 
                sr=sample_rate, 
                n_fft=n_fft,
                hop_length=hop_length
            )
            
            # 3. Chromagram
            chroma = librosa.feature.chroma_stft(
                y=signal, 
                sr=sample_rate, 
                n_fft=n_fft, 
                hop_length=hop_length
            )
            
            # Downsample for visualization
            max_frames = 100
            
            if mel_spec_db.shape[1] > max_frames:
                factor = mel_spec_db.shape[1] // max_frames
                mel_spec_db = mel_spec_db[:, ::factor][:, :max_frames]
                contrast = contrast[:, ::factor][:, :max_frames]
                chroma = chroma[:, ::factor][:, :max_frames]
            
            # Convert to lists for JSON serialization
            return {
                "mel_spectrogram": mel_spec_db.tolist(),
                "spectral_contrast": contrast.tolist(),
                "chromagram": chroma.tolist(),
                "sample_rate": sample_rate,
                "n_fft": n_fft,
                "hop_length": hop_length
            }
            
        except ImportError:
            # Fallback to simple FFT if librosa is not available
            import scipy.fftpack
            
            # Calculate simple spectrogram
            n_fft = 2048
            
            # Take a segment of the signal if it's very long
            if len(signal) > sample_rate * 10:  # Limit to 10 seconds
                signal = signal[:sample_rate * 10]
            
            # Calculate spectrogram
            spectrogram = []
            window_size = n_fft
            hop_length = window_size // 2
            
            for i in range(0, len(signal) - window_size, hop_length):
                window = signal[i:i + window_size]
                window = window * np.hanning(window_size)
                spectrum = np.abs(np.fft.rfft(window))
                spectrogram.append(spectrum.tolist())
            
            # Downsample for visualization
            max_frames = 100
            if len(spectrogram) > max_frames:
                factor = len(spectrogram) // max_frames
                spectrogram = spectrogram[::factor][:max_frames]
            
            return {
                "spectrogram": spectrogram,
                "sample_rate": sample_rate,
                "n_fft": n_fft
            }
            
    except Exception as e:
        logger.error(f"Error generating spectral features: {str(e)}", exc_info=True)
        
        # Return empty features in case of error
        return {
            "error": str(e)
        }

async def analyze_voice(audio_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze a voice recording for deepfake detection.
    This is a specialized version that focuses on voice-specific features.
    
    Args:
        audio_data: Dictionary containing preprocessed audio data
        
    Returns:
        Dictionary with analysis results
    """
    # This is a wrapper around the general audio analysis function
    # In a real implementation, this could include voice-specific feature extraction
    result = await analyze_audio(audio_data)
    
    # Add voice-specific analysis if available
    if "details" in result and isinstance(result["details"], dict):
        details = result["details"]
        
        # Add voice-specific features
        if "features" in details:
            # Rename or add voice-specific feature names
            features = details["features"]
            voice_features = {}
            
            # Map general audio features to voice-specific ones
            feature_mapping = {
                "Spectral Analysis": "Voice Frequency Analysis",
                "Temporal Coherence": "Speech Pattern Consistency",
                "Background Noise": "Background Noise",
                "Spectral Consistency": "Voice Timbre Consistency",
                "Formant Analysis": "Formant Distribution"
            }
            
            for old_name, new_name in feature_mapping.items():
                if old_name in features:
                    voice_features[new_name] = features[old_name]
                else:
                    # Generate a random score if feature not present
                    import random
                    voice_features[new_name] = random.uniform(0.3, 0.8)
            
            details["features"] = voice_features
    
    return result
