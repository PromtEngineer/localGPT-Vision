import numpy as np
from scipy import signal
from typing import Tuple, Optional

class AudioProcessor:
    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate
        self.silence_threshold = 0.01
        self.min_audio_length = 0.5  # seconds
        
    def process(
        self, 
        audio: np.ndarray, 
        sample_rate: int
    ) -> Tuple[np.ndarray, bool]:
        """Process audio for transcription"""
        # Resample if needed
        if sample_rate != self.target_sample_rate:
            audio = self.resample(audio, sample_rate, self.target_sample_rate)
        
        # Normalize audio
        audio = self.normalize_audio(audio)
        
        # Check if audio is too short
        if len(audio) < self.target_sample_rate * self.min_audio_length:
            return audio, False
        
        # Apply noise reduction
        audio = self.reduce_noise(audio)
        
        # Check for silence
        is_silence = self.is_silence(audio)
        
        return audio, not is_silence
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range"""
        max_val = np.abs(audio).max()
        if max_val > 0:
            return audio / max_val
        return audio
    
    def resample(
        self, 
        audio: np.ndarray, 
        orig_sr: int, 
        target_sr: int
    ) -> np.ndarray:
        """Resample audio to target sample rate"""
        if orig_sr == target_sr:
            return audio
        
        # Calculate resample ratio
        resample_ratio = target_sr / orig_sr
        new_length = int(len(audio) * resample_ratio)
        
        # Use scipy's resample
        return signal.resample(audio, new_length)
    
    def reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """Apply basic noise reduction using spectral subtraction"""
        # Apply high-pass filter to remove low-frequency noise
        if len(audio) > 13:  # Minimum length for filter
            b, a = signal.butter(4, 100, 'hp', fs=self.target_sample_rate)
            audio = signal.filtfilt(b, a, audio)
        return audio
    
    def is_silence(self, audio: np.ndarray) -> bool:
        """Detect if audio is silence"""
        rms = np.sqrt(np.mean(audio**2))
        return rms < self.silence_threshold
    
    def apply_voice_activity_detection(
        self, 
        audio: np.ndarray
    ) -> np.ndarray:
        """Simple VAD to trim silence from beginning and end"""
        # Calculate energy for each frame
        frame_length = int(0.025 * self.target_sample_rate)  # 25ms frames
        hop_length = int(0.010 * self.target_sample_rate)    # 10ms hop
        
        energy = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            energy.append(np.sum(frame**2))
        
        energy = np.array(energy)
        threshold = np.mean(energy) * 0.1
        
        # Find voice activity regions
        voice_activity = energy > threshold
        
        if np.any(voice_activity):
            # Find first and last active frame
            first_active = np.argmax(voice_activity)
            last_active = len(voice_activity) - np.argmax(voice_activity[::-1])
            
            # Convert frame indices to sample indices
            start_sample = first_active * hop_length
            end_sample = min(last_active * hop_length + frame_length, len(audio))
            
            return audio[start_sample:end_sample]
        
        return audio