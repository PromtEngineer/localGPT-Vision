from setuptools import setup, find_packages

setup(
    name="speech-transcription",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "sounddevice>=0.4.6",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "PyQt6>=6.5.0",
        "mlx>=0.5.0",
        "mlx-lm>=0.2.0",
        "mlx-whisper>=0.2.0",
        "soundfile>=0.12.0",
    ],
    entry_points={
        "console_scripts": [
            "speech-transcribe=src.cli:main",
            "speech-transcribe-gui=src.main:main",
        ],
    },
    python_requires=">=3.9",
)