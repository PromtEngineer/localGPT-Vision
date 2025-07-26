import argparse
import sys
from pathlib import Path
from .pipeline import TranscriptionPipeline
from .models.transcriber import AVAILABLE_MODELS

def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Speech Transcription CLI"
    )
    
    parser.add_argument(
        "--audio",
        type=str,
        help="Path to audio file to transcribe"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="whisper-tiny",
        choices=list(AVAILABLE_MODELS.keys()),
        help="Model to use for transcription"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: stdout)"
    )
    
    parser.add_argument(
        "--no-correction",
        action="store_true",
        help="Skip AI text correction"
    )
    
    parser.add_argument(
        "--context",
        type=str,
        help="Context for better corrections"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models"
    )
    
    parser.add_argument(
        "--keep-fillers",
        action="store_true",
        help="Keep filler words in transcription"
    )
    
    args = parser.parse_args()
    
    # List models
    if args.list_models:
        print("\nAvailable models:")
        for name, config in AVAILABLE_MODELS.items():
            print(f"  {name}: {config['description']}")
        return 0
    
    # Validate audio file
    if not args.audio:
        parser.error("--audio is required")
    
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return 1
    
    # Initialize pipeline
    try:
        pipeline = TranscriptionPipeline(model_name=args.model)
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        return 1
    
    # Transcribe
    print(f"Transcribing {audio_path} with {args.model}...")
    try:
        text = pipeline.transcribe_file(str(audio_path))
        
        if not text:
            print("No speech detected in audio file")
            return 1
        
        # Apply correction if requested
        if not args.no_correction:
            print("Applying text correction...")
            text = pipeline.correct_text(
                text, 
                context=args.context
            )
        
        # Output result
        if args.output:
            with open(args.output, 'w') as f:
                f.write(text)
            print(f"Transcription saved to {args.output}")
        else:
            print("\nTranscription:")
            print("-" * 50)
            print(text)
            print("-" * 50)
        
        return 0
        
    except Exception as e:
        print(f"Error during transcription: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())