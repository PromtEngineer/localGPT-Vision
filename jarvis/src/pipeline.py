import numpy as np
from typing import Optional, Callable
from .audio.recorder import AudioRecorder
from .audio.processor import AudioProcessor
from .models.transcriber import SpeechTranscriber
from .models.text_corrector import TextCorrector

class TranscriptionPipeline:
    """Main pipeline orchestrating transcription flow"""
    
    def __init__(
        self,
        model_name: str = "whisper-tiny",
        llm_model_name: str = "Phi-3.5-mini",
        sample_rate: int = 16000
    ):
        self.sample_rate = sample_rate
        self.audio_recorder = None
        self.audio_processor = AudioProcessor(sample_rate)
        self.transcriber = SpeechTranscriber(model_name)
        self.text_corrector = TextCorrector(llm_model_name)
        
    def set_model(self, model_name: str):
        """Change the transcription model"""
        self.transcriber.load_model(model_name)
    
    def set_llm_model(self, model_name: str):
        """Change the LLM model for text correction"""
        self.text_corrector.set_model(model_name)
    
    def start_recording(self, callback: Optional[Callable] = None, device_id: Optional[int] = None):
        """Start recording audio"""
        self.audio_recorder = AudioRecorder(
            sample_rate=self.sample_rate,
            callback=callback
        )
        self.audio_recorder.start_recording(device_id=device_id)
    
    def stop_recording(self) -> str:
        """Stop recording and return transcription"""
        if not self.audio_recorder:
            return ""
        
        # Stop recording
        final_audio = self.audio_recorder.stop_recording()
        
        # Use only the audio from stop_recording (which already includes all chunks)
        audio = final_audio
        
        # Process audio
        processed_audio, has_speech = self.audio_processor.process(
            audio, self.sample_rate
        )
        
        if not has_speech:
            return ""
        
        # Transcribe
        text = self.transcriber.transcribe(processed_audio, self.sample_rate)
        return text
    
    def transcribe_file(self, file_path: str) -> str:
        """Transcribe an audio file"""
        import soundfile as sf
        
        # Load audio
        audio, sr = sf.read(file_path)
        
        # Process
        processed_audio, has_speech = self.audio_processor.process(audio, sr)
        
        if not has_speech:
            return ""
        
        # Transcribe
        return self.transcriber.transcribe(processed_audio, self.sample_rate)
    
    def correct_text(
        self, 
        text: str, 
        context: Optional[str] = None
    ) -> str:
        """Apply AI correction to transcribed text"""
        return self.text_corrector.correct(text, context)
    
    def get_audio_level(self) -> float:
        """Get current audio level"""
        if self.audio_recorder:
            return self.audio_recorder.get_audio_level()
        return 0.0
    
    def process_stream(
        self,
        audio_chunk: np.ndarray,
        callback: Callable[[str], None]
    ):
        """Process audio stream in real-time"""
        # This would be used for streaming mode
        # Process chunk and transcribe incrementally
        pass