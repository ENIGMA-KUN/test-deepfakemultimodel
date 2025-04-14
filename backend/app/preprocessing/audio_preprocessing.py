import os
import numpy as np
import librosa
import soundfile as sf
import torch
from typing import Tuple, List, Dict, Any, Optional, Union
import logging

# Import the extract_audio_from_video function from utils for compatibility
from app.utils.audio_utils import extract_audio_from_video as _extract_audio_from_video

# Configure logging
logger = logging.getLogger(__name__)

# Create a pass-through function for compatibility
def extract_audio_from_video(video_path: str, output_path: Optional[str] = None) -> str:
    """
    Extract audio track from a video file - pass-through to utils function.
    
    Args:
        video_path: Path to the video file
        output_path: Path to save the extracted audio
        
    Returns:
        Path to the extracted audio file
    """
    return _extract_audio_from_video(video_path, output_path)


def load_audio(audio_path: str, sr: int = 16000, mono: bool = True) -> Tuple[np.ndarray, int]:
    """
    Load an audio file with specified parameters.
    
    Args:
        audio_path: Path to the audio file
        sr: Sample rate to load audio with
        mono: Whether to convert audio to mono
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr, mono=mono)
        return y, sr
    except Exception as e:
        logger.error(f"Error loading audio {audio_path}: {str(e)}")
        raise ValueError(f"Failed to load audio: {str(e)}")


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio to target sample rate.
    
    Args:
        audio: Audio data as numpy array
        orig_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled audio data
    """
    if orig_sr == target_sr:
        return audio
    
    try:
        resampled = librosa.resample(y=audio, orig_sr=orig_sr, target_sr=target_sr)
        return resampled
    except Exception as e:
        logger.error(f"Error resampling audio: {str(e)}")
        raise ValueError(f"Failed to resample audio: {str(e)}")


def extract_audio_features(audio_path: str, sr: int = 16000) -> Dict[str, Any]:
    """
    Extract comprehensive audio features for deepfake detection.
    
    Args:
        audio_path: Path to the audio file
        sr: Sample rate to use
        
    Returns:
        Dictionary containing extracted features
    """
    try:
        # Load audio
        y, sr = load_audio(audio_path, sr=sr)
        
        # Calculate duration
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        
        # Extract spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        # Extract temporal features
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        
        # Extract harmonic and percussive components
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harmonic_energy = np.sum(y_harmonic**2)
        percussive_energy = np.sum(y_percussive**2)
        harmonic_percussive_ratio = harmonic_energy / percussive_energy if percussive_energy > 0 else 0
        
        # Extract chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Calculate mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return {
            "duration": float(duration),
            "mfcc": {
                "mean": mfcc_mean.tolist(),
                "std": mfcc_std.tolist(),
            },
            "spectral": {
                "centroid_mean": float(np.mean(spectral_centroid)),
                "bandwidth_mean": float(np.mean(spectral_bandwidth)),
                "contrast_mean": np.mean(spectral_contrast, axis=1).tolist(),
                "rolloff_mean": float(np.mean(spectral_rolloff)),
            },
            "temporal": {
                "zero_crossing_rate_mean": float(np.mean(zero_crossing_rate)),
                "zero_crossing_rate_std": float(np.std(zero_crossing_rate)),
            },
            "harmonic_percussive": {
                "harmonic_energy": float(harmonic_energy),
                "percussive_energy": float(percussive_energy),
                "harmonic_percussive_ratio": float(harmonic_percussive_ratio),
            },
            "chroma_mean": np.mean(chroma, axis=1).tolist(),
            "mel_spectrogram_mean": np.mean(mel_spec_db, axis=1).tolist(),
        }
    except Exception as e:
        logger.error(f"Error extracting audio features: {str(e)}")
        return {"error": str(e)}


def segment_audio(audio_path: str, segment_length: float = 3.0, hop_length: float = 1.5) -> List[np.ndarray]:
    """
    Segment audio into overlapping chunks for analysis.
    
    Args:
        audio_path: Path to the audio file
        segment_length: Length of each segment in seconds
        hop_length: Hop length between segments in seconds
        
    Returns:
        List of audio segments as numpy arrays
    """
    try:
        # Load audio
        y, sr = load_audio(audio_path)
        
        # Convert seconds to samples
        segment_samples = int(segment_length * sr)
        hop_samples = int(hop_length * sr)
        
        # Segment the audio
        segments = []
        for start in range(0, len(y) - segment_samples + 1, hop_samples):
            segment = y[start:start + segment_samples]
            segments.append(segment)
        
        # If no segments were created (audio too short), use the whole audio
        if not segments and len(y) > 0:
            segments.append(y)
        
        return segments
    except Exception as e:
        logger.error(f"Error segmenting audio: {str(e)}")
        return []


def preprocess_for_model(audio_path: str, model_type: str = 'wav2vec2') -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
    """
    Preprocess audio for specific deepfake detection models.
    
    Args:
        audio_path: Path to the audio file
        model_type: Type of model ('wav2vec2', 'rawnet2', 'melspec')
        
    Returns:
        Preprocessed audio as a torch.Tensor (with batch dimension) and sample rate if needed
    """
    try:
        # Load audio
        y, sr = load_audio(audio_path, sr=16000)  # Standard 16kHz for audio models
        
        if model_type.lower() == 'wav2vec2':
            # Wav2Vec2 expects raw waveform input
            if len(y) > 16000 * 10:  # Limit to 10 seconds
                y = y[:16000 * 10]
            audio_tensor = torch.tensor(y).unsqueeze(0)  # Add batch dimension
            return audio_tensor, sr
        
        elif model_type.lower() == 'rawnet2':
            # RawNet2 expects raw waveform input with shape [B, 1, T]
            if len(y) > 16000 * 10:  # Limit to 10 seconds
                y = y[:16000 * 10]
            audio_tensor = torch.tensor(y).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            return audio_tensor
        
        elif model_type.lower() == 'melspec':
            # Convert to mel spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            S_db = librosa.power_to_db(S, ref=np.max)
            
            # Normalize
            S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
            
            # Convert to tensor with shape [B, C, H, W]
            audio_tensor = torch.tensor(S_db).unsqueeze(0).unsqueeze(0).float()
            return audio_tensor
        
        else:
            raise ValueError(f"Unsupported audio model type: {model_type}")
    
    except Exception as e:
        logger.error(f"Error preprocessing audio for model: {str(e)}")
        raise ValueError(f"Failed to preprocess audio: {str(e)}")


def analyze_voice_consistency(audio_path: str) -> Dict[str, Any]:
    """
    Analyze voice consistency over time, useful for deepfake detection.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Dictionary with consistency analysis results
    """
    try:
        # Load audio
        y, sr = load_audio(audio_path)
        
        # Extract features over time
        # Segment audio into chunks
        segment_length = sr * 1  # 1 second segments
        num_segments = len(y) // segment_length
        
        if num_segments < 2:
            return {
                "consistency_score": 1.0,
                "segments_analyzed": 1,
                "error": "Audio too short for consistency analysis"
            }
        
        # Extract MFCCs for each segment
        segment_mfccs = []
        for i in range(num_segments):
            segment = y[i * segment_length:(i + 1) * segment_length]
            mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
            segment_mfccs.append(np.mean(mfccs, axis=1))
        
        # Calculate variance between segments
        segment_mfccs = np.array(segment_mfccs)
        variance = np.mean(np.var(segment_mfccs, axis=0))
        
        # Calculate consistency score (inverse of variance, normalized)
        consistency_score = 1.0 / (1.0 + variance)
        
        # Calculate segment-to-segment differences
        segment_diffs = []
        for i in range(len(segment_mfccs) - 1):
            diff = np.mean(np.abs(segment_mfccs[i+1] - segment_mfccs[i]))
            segment_diffs.append(diff)
        
        return {
            "consistency_score": float(consistency_score),
            "segments_analyzed": num_segments,
            "variance": float(variance),
            "mean_segment_diff": float(np.mean(segment_diffs)),
            "max_segment_diff": float(np.max(segment_diffs)),
            "segment_diffs": [float(d) for d in segment_diffs]
        }
    
    except Exception as e:
        logger.error(f"Error analyzing voice consistency: {str(e)}")
        return {
            "consistency_score": 0.0,
            "segments_analyzed": 0,
            "error": str(e)
        }


def extract_pitch_contour(audio_path: str) -> Dict[str, Any]:
    """
    Extract pitch contour and related features for voice analysis.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Dictionary with pitch analysis results
    """
    try:
        # Load audio
        y, sr = load_audio(audio_path)
        
        # Extract pitch
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        
        # Get most prominent pitch at each time step
        pitch_contour = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            pitch_contour.append(pitch)
        
        # Calculate pitch statistics
        valid_pitches = [p for p in pitch_contour if p > 0]
        
        if not valid_pitches:
            return {
                "error": "No valid pitch detected in audio",
                "pitch_mean": 0,
                "pitch_std": 0,
                "pitch_range": 0
            }
        
        pitch_mean = np.mean(valid_pitches)
        pitch_std = np.std(valid_pitches)
        pitch_range = np.max(valid_pitches) - np.min(valid_pitches)
        
        # Calculate pitch stability (how much the pitch changes)
        pitch_changes = []
        for i in range(len(valid_pitches) - 1):
            change = abs(valid_pitches[i+1] - valid_pitches[i])
            pitch_changes.append(change)
        
        mean_pitch_change = np.mean(pitch_changes) if pitch_changes else 0
        
        return {
            "pitch_mean": float(pitch_mean),
            "pitch_std": float(pitch_std),
            "pitch_range": float(pitch_range),
            "mean_pitch_change": float(mean_pitch_change)
        }
    except Exception as e:
        logger.error(f"Error extracting pitch contour: {str(e)}")
        return {
            "error": str(e),
            "pitch_mean": 0,
            "pitch_std": 0,
            "pitch_range": 0
        }


def detect_silence_segments(audio_path: str, min_silence_duration: float = 0.5, threshold_db: float = -60) -> Dict[str, Any]:
    """
    Detect silence segments in audio which can indicate unnatural editing or splicing.
    
    Args:
        audio_path: Path to the audio file
        min_silence_duration: Minimum silence duration in seconds to consider
        threshold_db: Threshold in dB below which audio is considered silence
        
    Returns:
        Dictionary with silence analysis results
    """
    try:
        # Load audio
        y, sr = load_audio(audio_path)
        
        # Convert threshold to amplitude
        threshold_amp = librosa.db_to_amplitude(threshold_db)
        
        # Calculate amplitude envelope
        amplitude_envelope = np.abs(y)
        
        # Find silence regions
        is_silence = amplitude_envelope < threshold_amp
        
        # Identify contiguous silence segments
        silence_segments = []
        in_silence = False
        silence_start = 0
        
        for i, silent in enumerate(is_silence):
            if silent and not in_silence:
                # Start of a silence segment
                in_silence = True
                silence_start = i
            elif not silent and in_silence:
                # End of a silence segment
                in_silence = False
                duration = (i - silence_start) / sr
                if duration >= min_silence_duration:
                    silence_segments.append({
                        "start": float(silence_start / sr),
                        "end": float(i / sr),
                        "duration": float(duration)
                    })
        
        # If the audio ends with silence
        if in_silence:
            duration = (len(y) - silence_start) / sr
            if duration >= min_silence_duration:
                silence_segments.append({
                    "start": float(silence_start / sr),
                    "end": float(len(y) / sr),
                    "duration": float(duration)
                })
        
        # Calculate silence statistics
        total_duration = len(y) / sr
        silence_durations = [segment["duration"] for segment in silence_segments]
        total_silence_duration = sum(silence_durations)
        silence_percentage = (total_silence_duration / total_duration) * 100 if total_duration > 0 else 0
        
        return {
            "silence_count": len(silence_segments),
            "silence_segments": silence_segments,
            "total_silence_duration": float(total_silence_duration),
            "silence_percentage": float(silence_percentage),
            "avg_silence_duration": float(np.mean(silence_durations)) if silence_durations else 0,
            "max_silence_duration": float(np.max(silence_durations)) if silence_durations else 0
        }
    except Exception as e:
        logger.error(f"Error detecting silence segments: {str(e)}")
        return {
            "error": str(e),
            "silence_count": 0,
            "silence_segments": []
        }


def analyze_spectral_discontinuities(audio_path: str) -> Dict[str, Any]:
    """
    Analyze spectral discontinuities that might indicate splicing or editing.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Dictionary with spectral discontinuity analysis
    """
    try:
        # Load audio
        y, sr = load_audio(audio_path)
        
        # Calculate spectrogram
        S = np.abs(librosa.stft(y))
        
        # Calculate frame-to-frame spectral difference
        spectral_diff = np.zeros(S.shape[1] - 1)
        for i in range(len(spectral_diff)):
            spectral_diff[i] = np.sqrt(np.sum((S[:, i+1] - S[:, i])**2))
        
        # Normalize differences
        if np.max(spectral_diff) > 0:
            spectral_diff = spectral_diff / np.max(spectral_diff)
        
        # Detect sudden large changes (potential splices)
        threshold = np.mean(spectral_diff) + 2 * np.std(spectral_diff)
        potential_splices = np.where(spectral_diff > threshold)[0]
        
        # Convert to time positions
        hop_length = 512  # Default hop length in librosa.stft
        splice_times = [(p * hop_length) / sr for p in potential_splices]
        
        # Calculate statistics
        discontinuity_score = float(np.mean(spectral_diff))
        max_discontinuity = float(np.max(spectral_diff))
        
        return {
            "discontinuity_score": discontinuity_score,
            "max_discontinuity": max_discontinuity,
            "potential_splices": len(splice_times),
            "splice_times": [float(t) for t in splice_times],
            "mean_spectral_diff": float(np.mean(spectral_diff)),
            "std_spectral_diff": float(np.std(spectral_diff))
        }
    except Exception as e:
        logger.error(f"Error analyzing spectral discontinuities: {str(e)}")
        return {
            "error": str(e),
            "discontinuity_score": 0,
            "potential_splices": 0
        }


def comprehensive_audio_analysis(audio_path: str) -> Dict[str, Any]:
    """
    Perform comprehensive analysis of audio for deepfake detection.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Dictionary with comprehensive analysis results
    """
    results = {}
    
    try:
        # First check if the file is valid
        if not os.path.exists(audio_path):
            return {"error": f"Audio file not found: {audio_path}"}
        
        # Extract basic audio features
        results["audio_features"] = extract_audio_features(audio_path)
        
        # Analyze voice consistency
        results["voice_consistency"] = analyze_voice_consistency(audio_path)
        
        # Extract pitch contour
        results["pitch_analysis"] = extract_pitch_contour(audio_path)
        
        # Detect silence segments
        results["silence_analysis"] = detect_silence_segments(audio_path)
        
        # Analyze spectral discontinuities
        results["spectral_analysis"] = analyze_spectral_discontinuities(audio_path)
        
        # Get audio duration
        try:
            y, sr = load_audio(audio_path)
            results["duration"] = float(len(y) / sr)
        except Exception as e:
            results["duration_error"] = str(e)
        
        # Calculate overall authenticity score based on various factors
        # This is a simplified scoring approach and can be refined
        authenticity_factors = []
        
        # Factor 1: Voice consistency (higher is more consistent)
        if "voice_consistency" in results and "consistency_score" in results["voice_consistency"]:
            authenticity_factors.append(results["voice_consistency"]["consistency_score"])
        
        # Factor 2: Inverse of spectral discontinuities (lower discontinuity score is better)
        if "spectral_analysis" in results and "discontinuity_score" in results["spectral_analysis"]:
            disc_score = results["spectral_analysis"]["discontinuity_score"]
            # Invert and scale to 0-1 range, assuming max score is 1
            authenticity_factors.append(1.0 - min(disc_score, 1.0))
        
        # Factor 3: Pitch stability (lower mean_pitch_change is more stable)
        if "pitch_analysis" in results and "mean_pitch_change" in results["pitch_analysis"]:
            pitch_change = results["pitch_analysis"]["mean_pitch_change"]
            # Scale to 0-1 range, assuming values typically under 50
            pitch_stability = 1.0 - min(pitch_change / 50.0, 1.0)
            authenticity_factors.append(pitch_stability)
        
        # Calculate overall score if we have factors
        if authenticity_factors:
            results["authenticity_score"] = float(np.mean(authenticity_factors))
        else:
            results["authenticity_score"] = 0.5  # Default score
            results["authenticity_score_warning"] = "Limited data for scoring"
        
        return results
    
    except Exception as e:
        logger.error(f"Error in comprehensive audio analysis: {str(e)}")
        results["error"] = str(e)
        return results
