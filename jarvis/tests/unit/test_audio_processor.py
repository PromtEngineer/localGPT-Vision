import pytest
import numpy as np
from src.audio.processor import AudioProcessor

class TestAudioProcessor:
    def test_normalize_audio(self):
        processor = AudioProcessor()
        audio = np.array([0.5, -0.5, 0.25, -0.25])
        normalized = processor.normalize_audio(audio)
        assert normalized.max() == 1.0 or normalized.min() == -1.0
    
    def test_silence_detection(self):
        processor = AudioProcessor()
        
        # Test silence
        silence = np.zeros(16000)
        assert processor.is_silence(silence)
        
        # Test non-silence
        tone = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
        assert not processor.is_silence(tone)
    
    def test_resample(self):
        processor = AudioProcessor()
        
        # Create 1 second of audio at 44100 Hz
        original_sr = 44100
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, original_sr))
        
        # Resample to 16000 Hz
        resampled = processor.resample(audio, original_sr, 16000)
        
        # Check length
        expected_length = 16000
        assert abs(len(resampled) - expected_length) < 10