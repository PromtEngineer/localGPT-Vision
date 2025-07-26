import os
import tempfile
import numpy as np
import soundfile as sf
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

# Model configurations
AVAILABLE_MODELS = {
    "whisper-tiny": {
        "repo": "mlx-community/whisper-tiny",
        "type": "whisper",
        "description": "Very fast, lower accuracy"
    },
    "whisper-base": {
        "repo": "mlx-community/whisper-base-mlx",
        "type": "whisper",
        "description": "Fast, decent accuracy"
    },
    "whisper-small": {
        "repo": "mlx-community/whisper-small-mlx",
        "type": "whisper",
        "description": "Good balance"
    },
    "whisper-medium": {
        "repo": "mlx-community/whisper-medium-mlx",
        "type": "whisper",
        "description": "High accuracy, slower"
    },
    "whisper-large-v3": {
        "repo": "mlx-community/whisper-large-v3-mlx",
        "type": "whisper",
        "description": "Best accuracy, slowest"
    },
    "distil-whisper-large-v3": {
        "repo": "mlx-community/distil-whisper-large-v3",
        "type": "whisper",
        "description": "Fast with high accuracy"
    }
}

class BaseTranscriber(ABC):
    """Base class for speech transcribers"""
    
    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        pass

class WhisperTranscriber(BaseTranscriber):
    """Whisper model transcriber using MLX"""
    
    def __init__(self, model_id: str):
        try:
            import mlx_whisper
            self.mlx_whisper = mlx_whisper
            self.model_path = model_id
        except ImportError:
            raise ImportError("mlx-whisper is required. Install with: pip install mlx-whisper")
    
    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file using Whisper"""
        result = self.mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=self.model_path
        )
        return result.get("text", "")

class SpeechTranscriber:
    """Main transcriber class with model management"""
    
    def __init__(self, model_name: str = "whisper-tiny"):
        self.model_name = model_name
        self.model = None
        self.load_model(model_name)
    
    def load_model(self, model_name: str):
        """Load the specified model"""
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = AVAILABLE_MODELS[model_name]
        model_type = config["type"]
        repo_id = config["repo"]
        
        print(f"Loading {model_name} model...")
        
        if model_type == "whisper":
            self.model = WhisperTranscriber(repo_id)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model_name = model_name
        print(f"Model {model_name} loaded successfully")
    
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio array"""
        if self.model is None:
            raise RuntimeError("No model loaded")
        
        # Write audio to temporary file (required by models)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio, sample_rate)
            temp_path = tmp.name
        
        try:
            # Transcribe
            text = self.model.transcribe(temp_path)
            return text.strip()
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def list_models(self) -> Dict[str, Any]:
        """List available models"""
        return AVAILABLE_MODELS