import sounddevice as sd
import numpy as np
from typing import Callable, Optional
import threading
import queue

class AudioRecorder:
    def __init__(
        self, 
        sample_rate: int = 16000,
        chunk_duration: float = 0.5,
        callback: Optional[Callable[[np.ndarray], None]] = None
    ):
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration)
        self.callback = callback
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
    def audio_callback(self, indata, frames, time, status):
        """Callback for sounddevice stream"""
        if status:
            print(f"Audio callback status: {status}")
        
        # Copy audio data to queue
        audio_chunk = indata[:, 0].copy()  # Get first channel
        self.audio_queue.put(audio_chunk)
        
        # Call user callback if provided
        if self.callback:
            self.callback(audio_chunk)
    
    def start_recording(self, device_id: Optional[int] = None):
        """Start audio recording"""
        self.is_recording = True
        
        # Start audio stream
        self.stream = sd.InputStream(
            device=device_id,
            channels=1,
            samplerate=self.sample_rate,
            callback=self.audio_callback,
            blocksize=self.chunk_size
        )
        self.stream.start()
        
    def stop_recording(self) -> np.ndarray:
        """Stop recording and return accumulated audio"""
        self.is_recording = False
        
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        # Collect all audio chunks
        audio_chunks = []
        while not self.audio_queue.empty():
            chunk = self.audio_queue.get()
            audio_chunks.append(chunk)
        
        if audio_chunks:
            return np.concatenate(audio_chunks)
        return np.array([])
    
    def get_audio_level(self) -> float:
        """Get current audio level for visualization"""
        if not self.audio_queue.empty():
            chunk = self.audio_queue.queue[-1]  # Peek at last item
            return float(np.abs(chunk).mean())
        return 0.0