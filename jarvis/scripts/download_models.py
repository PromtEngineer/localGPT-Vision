"""Download required ML models"""

from huggingface_hub import snapshot_download
import os

# Models to download
MODELS = [
    # Whisper models for transcription
    "mlx-community/whisper-tiny",
    "mlx-community/whisper-base-mlx",
    
    # LLM models for text correction (at least one)
    "mlx-community/Phi-3.5-mini-instruct-4bit",
    "mlx-community/Qwen2.5-0.5B-Instruct-4bit",  # Tiny option
    # Add more as needed:
    # "mlx-community/gemma-2-2b-it-4bit",
    # "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
]

def download_models():
    """Download all required models"""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    
    for model_id in MODELS:
        print(f"Downloading {model_id}...")
        try:
            snapshot_download(
                repo_id=model_id,
                cache_dir=cache_dir,
                resume_download=True
            )
            print(f"✓ {model_id} downloaded successfully")
        except Exception as e:
            print(f"✗ Failed to download {model_id}: {e}")

if __name__ == "__main__":
    download_models()