# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Real-Time Speech Transcription Application for macOS, designed to be privacy-focused with all processing done on-device using Apple Silicon optimization. The application features speech-to-text transcription with AI-powered text correction, multiple UI modes (GUI, CLI), and support for various speech recognition models.

### Original Requirements (PRD.md)
The project was built following a comprehensive Product Requirements Document that specified:
- Privacy-first approach with all processing on-device
- Real-time transcription performance (>5x real-time)
- Multiple STT model support (Whisper variants)
- AI-powered text correction using LLMs
- GUI and CLI interfaces
- Apple Silicon optimization using MLX framework

### Recent Feature Additions
1. **Microphone Selection**: Dropdown to choose from all available audio input devices
2. **Copy Functionality**: Separate copy buttons for raw and corrected text
3. **LLM Model Selection**: Dynamic selection of text correction models from GUI
4. **Enhanced Error Handling**: Proper thread cleanup and crash prevention
5. **Auto Qt Path Detection**: Automatic resolution of Qt platform plugin issues

## Architecture

### Core Components

1. **Audio Pipeline** (`src/audio/`)
   - `AudioRecorder`: Manages sounddevice streams for microphone capture with device selection
   - `AudioProcessor`: Handles normalization, resampling, noise reduction, and VAD

2. **Model Layer** (`src/models/`)
   - `SpeechTranscriber`: Abstraction over multiple STT models (Whisper variants)
   - `TextCorrector`: LLM-based correction using MLX models with dynamic model selection
   - Model dictionaries: `AVAILABLE_MODELS` (STT) and `AVAILABLE_LLM_MODELS` (correction)

3. **User Interfaces** (`src/gui/`)
   - `MainWindow`: PyQt6-based GUI with real-time visualization
     - Microphone selection dropdown with refresh capability
     - Model selection dropdowns for both STT and LLM models
     - Copy functionality for both raw and corrected text
     - Dark theme with audio level visualization
   - CLI interface in `src/cli.py`

4. **Pipeline** (`src/pipeline.py`)
   - Orchestrates the full transcription flow
   - Manages audio buffering without duplication
   - Supports dynamic model switching for both STT and LLM
   - Coordinates all components

### Key Design Decisions

- **Model Flexibility**: Support for multiple STT and LLM models with different accuracy/speed tradeoffs
- **Privacy-First**: All processing on-device, no cloud APIs
- **MLX Optimization**: Uses Apple's MLX framework for efficient inference on Apple Silicon
- **Dynamic Model Selection**: Users can choose models from the GUI without restart
- **Error Resilience**: Proper thread cleanup and error handling for stability

## Development Commands

### Project Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download required ML models
python scripts/download_models.py
```

### Running the Application
```bash
# GUI mode
python -m src.main

# CLI mode
python -m src.cli --audio recording.wav --model whisper-small

# With text correction
python -m src.cli --audio recording.wav --context "Technical discussion"
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run unit tests only
pytest tests/unit/ -v

# Run integration tests
pytest tests/integration/ -v

# Run specific test
pytest tests/unit/test_audio_processor.py::TestAudioProcessor::test_normalize_audio -v
```

### Development Workflow
```bash
# Format code
black src tests
ruff format src tests

# Lint code
ruff check src tests
mypy src

# Clean build artifacts
find . -type f -name "*.pyc" -delete
find . -type d -name "__pycache__" -delete
```

## Model Management

### Available STT Models
- **whisper-tiny**: Very fast, lower accuracy (39M params)
- **whisper-base**: Fast, decent accuracy (74M params)
- **whisper-small**: Good balance (244M params)
- **whisper-medium**: High accuracy, slower (769M params)
- **whisper-large-v3**: Best accuracy, slowest (1550M params)
- **distil-whisper-large-v3**: Fast with high accuracy (756M params)

### Available LLM Models for Text Correction
- **Qwen2.5-0.5B**: Tiny & fast (0.5B params, <2GB RAM)
- **Qwen2.5-1.5B**: Small & efficient (1.5B params, ~3GB RAM)
- **gemma-2-2b-it**: Compact & capable (2B params, ~4GB RAM)
- **Phi-3.5-mini**: Excellent quality (3.8B params, ~6GB RAM) - Default
- **Qwen2.5-7B**: Large & powerful (7B params, ~12GB RAM)
- **Mistral-7B-Instruct**: High quality (7B params, ~12GB RAM)

### Model Loading
Models are loaded on-demand from Hugging Face hub. Both `SpeechTranscriber` and `TextCorrector` support dynamic model switching via the GUI without restart.

## Audio Processing Pipeline

1. **Recording**: 16kHz mono audio captured in 0.5s chunks
   - Device selection with automatic default detection
   - Real-time audio level monitoring
   - Proper buffering without duplication
2. **Processing**: Normalization → Resampling → Noise reduction → VAD
3. **Transcription**: Audio written to temp file → Model inference → Text output
4. **Correction**: Optional LLM-based correction with context awareness
   - Custom prompt structure with context integration
   - No temperature/top_p parameters (unsupported by mlx_lm)

## GUI Architecture

- **Main Thread**: UI updates and user interaction
- **TranscriptionThread**: Background audio recording and level monitoring
- **Model Operations**: Synchronous but UI remains responsive via threading
- **Real-time Feedback**: Audio level visualization at 50ms intervals

## Performance Considerations

- Audio chunks processed at 0.5s intervals for low latency
- Models use MLX for Apple Silicon optimization
- Text correction uses default generation parameters (no temperature control)
- Memory-mapped model loading for faster startup
- Thread cleanup ensures no resource leaks on cancel/clear

## Common Issues and Fixes

### Qt Platform Plugin Error
**Problem**: "Could not find the Qt platform plugin 'cocoa'"
**Solution**: The code automatically detects and sets the Qt plugin path in `main.py`. If it persists:
```bash
# Use the provided run script
./run_gui.sh

# Or set manually
export QT_QPA_PLATFORM_PLUGIN_PATH=$(python -c "import PyQt6, os; print(os.path.join(os.path.dirname(PyQt6.__file__), 'Qt6', 'plugins', 'platforms'))")
```

### Duplicate Transcription
**Problem**: Transcribed text appears twice
**Solution**: Fixed by removing duplicate audio buffering in `pipeline.py`. The `stop_recording()` method now uses only the final audio from `AudioRecorder.stop_recording()`.

### LLM Temperature Error
**Problem**: "generate_step() got an unexpected keyword argument 'temperature'"
**Solution**: The mlx_lm library doesn't support temperature/top_p parameters. These have been removed from the `generate()` call in `TextCorrector`.

### Clear Button Crash
**Problem**: Clicking clear button terminates the application
**Solution**: Added proper error handling and thread cleanup in `clear_text()` method. The method now safely cancels any running recording before clearing.

### Text Correction Prompt
The LLM receives this structured prompt:
```
You are a helpful assistant that corrects transcription errors.
Context: {context if provided}

Please correct any transcription errors in the following text, removing filler words, fixing grammar, and improving clarity while preserving the original meaning:

{transcribed_text}

Corrected text:
```

### Whisper Model Repository Names
**Problem**: 404 errors when loading Whisper models
**Solution**: The mlx-community organization uses inconsistent naming for Whisper models:
- `whisper-tiny` - no suffix needed
- `whisper-base-mlx` - requires "-mlx" suffix
- `whisper-small-mlx` - requires "-mlx" suffix  
- `whisper-medium-mlx` - requires "-mlx" suffix
- `whisper-large-v3-mlx` - requires "-mlx" suffix
- `distil-whisper-large-v3` - no suffix needed

## Common Patterns

### Adding New STT Models
1. Add model config to `AVAILABLE_MODELS` in `src/models/transcriber.py`
2. Ensure model is Whisper-based (current implementation)
3. Update model download script if needed

### Adding New LLM Models
1. Add model config to `AVAILABLE_LLM_MODELS` in `src/models/text_corrector.py`
2. Include repo, description, and size fields
3. Test with mlx_lm.generate() compatibility
4. Update download script for default models

### Modifying Audio Processing
- All audio processing goes through `AudioProcessor.process()`
- Maintain 16kHz target sample rate for model compatibility
- Ensure processed audio is normalized to [-1, 1] range
- Avoid duplicate buffering in the pipeline

### GUI Customization
- Dark theme defined in `MainWindow.init_ui()`
- Keyboard shortcuts in `setup_shortcuts()`
- Audio visualization via `update_level()` callback
- Model dropdowns update pipeline without restart
- Copy buttons use QApplication.clipboard()

## Best Practices & Lessons Learned

### Thread Safety
- Always use proper thread cleanup in GUI operations
- Set `is_recording` flag before cleanup to prevent race conditions
- Use `wait()` with timeout before `terminate()` for graceful shutdown

### Audio Buffer Management
- Avoid duplicate buffering between recorder and pipeline
- Use the final audio from `stop_recording()` which includes all chunks
- Maintain consistent sample rates throughout the pipeline

### Model Parameter Compatibility
- mlx_lm's `generate()` doesn't support temperature/top_p parameters
- Use default generation settings for consistent behavior
- Test parameter compatibility when adding new models

### Error Handling
- Wrap all GUI callbacks with try/except blocks
- Provide user-friendly error messages via status bar
- Use QMessageBox for critical errors only

### Development Workflow
1. Always check PRD.md for requirements alignment
2. Test model downloads before assuming availability
3. Verify audio permissions before recording
4. Run both GUI and CLI modes during testing
5. Check memory usage with different model sizes