# Speech Transcription Application for macOS

A privacy-focused, real-time speech transcription desktop application for macOS that runs entirely on-device using Apple Silicon optimization. Features speech-to-text transcription with AI-powered text correction, multiple UI modes, and support for various speech recognition models.

## Features

- 🎤 **Real-time Speech Transcription**: Capture and transcribe audio from your microphone
- 🔒 **Privacy-First**: All processing happens on-device, no data sent to cloud
- ⚡ **Apple Silicon Optimized**: Uses MLX framework for fast inference on M1/M2/M3 chips
- 🤖 **AI-Powered Correction**: Automatic removal of filler words and transcription errors
- 🎯 **Multiple Models**: Support for various Whisper models with different speed/accuracy tradeoffs
- 🖥️ **Multiple Interfaces**: GUI, CLI, and system tray modes
- ⌨️ **Keyboard Shortcuts**: Space to start/stop, Escape to cancel, Cmd+S to save

## Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.9 or higher
- Minimum 8GB RAM (16GB recommended)
- ~5GB disk space for models

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/speech-transcription-app.git
cd speech-transcription-app
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download ML models

```bash
python scripts/download_models.py
```

## Usage

### GUI Mode

Launch the graphical interface:

```bash
python -m src.main
```

Features:
- Click "Start Recording" or press Space to begin
- Real-time audio level visualization
- Automatic text correction (can be disabled)
- Save transcriptions with Cmd+S
- Dark mode interface

### CLI Mode

Transcribe audio files from the command line:

```bash
# Basic transcription
python -m src.cli --audio recording.wav

# With specific model
python -m src.cli --audio recording.wav --model whisper-medium

# With text correction and context
python -m src.cli --audio recording.wav --context "Technical meeting"

# Save to file
python -m src.cli --audio recording.wav --output transcript.txt

# List available models
python -m src.cli --list-models
```

### Available Models

- `whisper-tiny`: Very fast, lower accuracy
- `whisper-base`: Fast, decent accuracy
- `whisper-small`: Good balance
- `whisper-medium`: High accuracy, slower
- `whisper-large-v3`: Best accuracy, slowest
- `distil-whisper-large-v3`: Fast with high accuracy

## Development

### Running Tests

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run integration tests
make test-integration
```

### Code Formatting

```bash
# Format code with black and ruff
make format

# Run linters
make lint
```

### Building for Distribution

```bash
pip install -e .
```

This will install the package in development mode and create command-line scripts:
- `speech-transcribe`: CLI interface
- `speech-transcribe-gui`: GUI interface

## Project Structure

```
speech-transcription-app/
├── src/
│   ├── audio/              # Audio recording and processing
│   ├── models/             # ML models for transcription and correction
│   ├── gui/                # GUI components
│   ├── pipeline.py         # Main orchestration
│   ├── cli.py              # CLI interface
│   └── main.py             # GUI entry point
├── tests/                  # Unit and integration tests
├── scripts/                # Utility scripts
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Performance

- **Audio Latency**: < 100ms for recording start/stop
- **Transcription Speed**: > 5x real-time (10s audio in < 2s)
- **Memory Usage**: < 2GB with models loaded
- **Model Loading**: < 10s for initial load

## Troubleshooting

### Common Issues

1. **"Could not find the Qt platform plugin 'cocoa'"**
   - This is fixed automatically in the code, but if it persists:
   - Run with: `./run_gui.sh` instead of `python -m src.main`
   - Or set manually: `export QT_QPA_PLATFORM_PLUGIN_PATH=$(python -c "import PyQt6, os; print(os.path.join(os.path.dirname(PyQt6.__file__), 'Qt6', 'plugins', 'platforms'))")`

2. **"No module named 'mlx'"**
   - Ensure you're on a Mac with Apple Silicon
   - Reinstall with: `pip install --upgrade mlx mlx-lm mlx-whisper`

3. **Audio permission denied**
   - Go to System Preferences → Security & Privacy → Microphone
   - Grant permission to Terminal/your IDE

4. **Model download fails**
   - Check internet connection
   - Manually download from Hugging Face if needed

### Debug Mode

Run with verbose output:
```bash
python -m src.cli --audio test.wav --debug
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linters
5. Submit a pull request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- Apple's MLX team for the optimization framework
- OpenAI for Whisper models
- Hugging Face for model hosting