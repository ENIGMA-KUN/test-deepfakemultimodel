# Audio Processing Code Consolidation Guide

## Problem Description

After reviewing the codebase, we've identified significant code duplication between `app/utils/audio_utils.py` and `app/preprocessing/audio_preprocessing.py`. This redundancy creates several problems:

1. **Maintenance Burden**: Changes to functionality need to be made in multiple places
2. **Inconsistent Implementations**: Functions with the same purpose may behave differently
3. **Confusion for Developers**: Unclear which version of a function should be used
4. **Increased Likelihood of Bugs**: Fixes applied to one version might not be applied to others

### Examples of Duplicated Functionality

Both files contain similar implementations of:

1. `check_voice_consistency` in `audio_utils.py` and `analyze_voice_consistency` in `audio_preprocessing.py`
2. `analyze_audio_features` in `audio_utils.py` and `extract_audio_features` in `audio_preprocessing.py`
3. `extract_audio_from_video` in both files (though `audio_preprocessing.py` uses a pass-through to `audio_utils.py`)

## Solution Approach

We'll follow a consolidated architecture where:

1. `audio_utils.py` will contain **low-level utilities** for basic audio operations
2. `audio_preprocessing.py` will contain **higher-level preprocessing** functions that build on the utilities

### Step 1: Refactor `audio_utils.py`

Keep only the most essential, low-level functions:

```python
# audio_utils.py
import os
import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)


def is_valid_audio(audio_path: str) -> bool:
    """
    Check if the file is a valid audio file.
    """
    try:
        y, sr = librosa.load(audio_path, sr=None, duration=1.0)  # Just load a small portion
        return True
    except Exception as e:
        logger.error(f"Invalid audio file: {str(e)}")
        return False


def extract_audio_from_video(video_path: str, output_path: Optional[str] = None) -> str:
    """
    Extract audio track from a video file.
    """
    try:
        import subprocess
        import shutil
        
        # Check if ffmpeg is available
        if shutil.which("ffmpeg") is None:
            logger.error("ffmpeg is not installed or not in PATH. Cannot extract audio from video.")
            raise RuntimeError("ffmpeg not found. Please install ffmpeg to enable audio extraction from videos.")
        
        if output_path is None:
            # Create temporary file
            output_path = video_path + ".wav"
        
        # Use ffmpeg to extract audio
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-q:a", "0",
            "-map", "a",
            output_path,
            "-y"  # Overwrite existing file if any
        ]
        
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if process.returncode != 0:
            error_msg = process.stderr.decode('utf-8', errors='replace')
            logger.error(f"ffmpeg error: {error_msg}")
            raise RuntimeError(f"ffmpeg failed to extract audio: {error_msg}")
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error extracting audio from video: {str(e)}")
        return ""


def load_audio(audio_path: str, sr: int = 16000, mono: bool = True) -> Tuple[np.ndarray, int]:
    """
    Load an audio file with specified parameters.
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
    """
    if orig_sr == target_sr:
        return audio
    
    try:
        resampled = librosa.resample(y=audio, orig_sr=orig_sr, target_sr=target_sr)
        return resampled
    except Exception as e:
        logger.error(f"Error resampling audio: {str(e)}")
        raise ValueError(f"Failed to resample audio: {str(e)}")
```

### Step 2: Enhance `audio_preprocessing.py`

Update `audio_preprocessing.py` to use the utilities from `audio_utils.py` and add more advanced functionality:

```python
# audio_preprocessing.py
import os
import numpy as np
import librosa
import torch
from typing import Tuple, List, Dict, Any, Optional, Union
import logging

# Import the utilities from audio_utils
from app.utils.audio_utils import (
    is_valid_audio,
    extract_audio_from_video, 
    load_audio, 
    resample_audio
)

# Configure logging
logger = logging.getLogger(__name__)


def extract_audio_features(audio_path: str, sr: int = 16000) -> Dict[str, Any]:
    """
    Extract comprehensive audio features for deepfake detection.
    """
    try:
        # Load audio using the utility function
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


def analyze_voice_consistency(audio_path: str) -> Dict[str, Any]:
    """
    Analyze voice consistency over time, useful for deepfake detection.
    """
    try:
        # Load audio using the utility function
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
```

### Step 3: Update Import References

Update all files that import from either module to use the correct module:

1. For basic operations:
```python
from app.utils.audio_utils import is_valid_audio, extract_audio_from_video, load_audio
```

2. For preprocessing operations:
```python
from app.preprocessing.audio_preprocessing import extract_audio_features, analyze_voice_consistency
```

### Step 4: Update `audio_models.py`

Update the audio models file to use the consolidated functions:

```python
# In preprocess_audio function in audio_models.py
from app.utils.audio_utils import load_audio
from app.preprocessing.audio_preprocessing import extract_audio_features
```

### Step 5: Update `audio_tasks.py`

Update the Celery tasks to use the new function locations:

```python
# In detect_audio function in audio_tasks.py
from app.preprocessing.audio_preprocessing import extract_audio_features, analyze_voice_consistency
```

## Testing Plan

After consolidating the code:

1. **Unit Testing**: Create unit tests for each function in both modules
2. **Integration Testing**: Test the audio detection pipeline with various audio files
3. **Regression Testing**: Ensure that the results match the previous implementation
4. **Edge Case Testing**: Test with very short audio files, corrupted files, etc.

## Benefits of Consolidation

1. **Improved Maintainability**: Single source of truth for each function
2. **Reduced Code Size**: Less code to maintain and debug
3. **Clearer Responsibility Boundaries**: Clear separation between utilities and preprocessing
4. **Better Performance**: Potential for optimization by avoiding duplicate calculations
5. **Consistent Results**: Ensures all parts of the application use the same implementation

## Conclusion

This consolidation effort will significantly improve the codebase structure and reduce the potential for bugs caused by inconsistent implementations of the same functionality. By clearly defining the responsibilities of each module, we make the codebase more maintainable and easier to understand for new developers.
