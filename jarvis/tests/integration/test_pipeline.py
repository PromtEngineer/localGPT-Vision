import pytest
import numpy as np
from src.pipeline import TranscriptionPipeline

class TestPipeline:
    @pytest.fixture
    def pipeline(self):
        return TranscriptionPipeline(model_name="whisper-tiny")
    
    def test_transcription_flow(self, pipeline, tmp_path):
        # Create test audio
        duration = 3  # seconds
        sample_rate = 16000
        t = np.linspace(0, duration, duration * sample_rate)
        
        # Simple tone
        audio = np.sin(2 * np.pi * 440 * t) * 0.5
        
        # Save to file
        import soundfile as sf
        test_file = tmp_path / "test.wav"
        sf.write(test_file, audio, sample_rate)
        
        # Transcribe
        result = pipeline.transcribe_file(str(test_file))
        
        # Should return empty for pure tone
        assert isinstance(result, str)