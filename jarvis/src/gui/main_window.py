import sys
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLabel, QComboBox,
    QCheckBox, QLineEdit, QProgressBar, QFileDialog,
    QMessageBox, QSplitter, QApplication
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QAction, QKeySequence, QFont
from typing import Optional

from ..pipeline import TranscriptionPipeline
from ..models.transcriber import AVAILABLE_MODELS
from ..models.text_corrector import AVAILABLE_LLM_MODELS
import sounddevice as sd

class TranscriptionThread(QThread):
    """Background thread for transcription"""
    textReady = pyqtSignal(str)
    errorOccurred = pyqtSignal(str)
    levelUpdate = pyqtSignal(float)
    
    def __init__(self, pipeline: TranscriptionPipeline, device_id: Optional[int] = None):
        super().__init__()
        self.pipeline = pipeline
        self.device_id = device_id
        self.is_recording = False
    
    def run(self):
        """Run transcription in background"""
        try:
            self.is_recording = True
            self.pipeline.start_recording(device_id=self.device_id)
            
            # Update audio levels
            while self.is_recording:
                level = self.pipeline.get_audio_level()
                self.levelUpdate.emit(level)
                self.msleep(50)  # 50ms updates
                
        except Exception as e:
            self.errorOccurred.emit(str(e))
    
    def stop_recording(self):
        """Stop recording and get transcription"""
        self.is_recording = False
        text = self.pipeline.stop_recording()
        self.textReady.emit(text)

class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.pipeline = TranscriptionPipeline()
        self.transcription_thread = None
        self.init_ui()
        self.setup_shortcuts()
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Speech Transcription")
        self.setGeometry(100, 100, 900, 700)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        layout = QVBoxLayout(central_widget)
        
        # Header section
        header_layout = QHBoxLayout()
        
        # Microphone selection
        mic_label = QLabel("Microphone:")
        self.mic_combo = QComboBox()
        self.refresh_microphones()
        self.mic_combo.currentIndexChanged.connect(self.on_mic_changed)
        
        # Add refresh button for microphones
        self.refresh_mic_button = QPushButton("🔄")
        self.refresh_mic_button.setMaximumWidth(30)
        self.refresh_mic_button.clicked.connect(self.refresh_microphones)
        self.refresh_mic_button.setToolTip("Refresh microphone list")
        
        header_layout.addWidget(mic_label)
        header_layout.addWidget(self.mic_combo)
        header_layout.addWidget(self.refresh_mic_button)
        
        # Model selection
        model_label = QLabel("Model:")
        self.model_combo = QComboBox()
        for model_name, config in AVAILABLE_MODELS.items():
            self.model_combo.addItem(
                f"{model_name} - {config['description']}", 
                model_name
            )
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        
        header_layout.addWidget(model_label)
        header_layout.addWidget(self.model_combo)
        header_layout.addStretch()
        
        # Options
        self.auto_correct_check = QCheckBox("Auto-correct after recording")
        self.auto_correct_check.setChecked(True)
        header_layout.addWidget(self.auto_correct_check)
        
        layout.addLayout(header_layout)
        
        # Second header row for LLM model and context
        second_header_layout = QHBoxLayout()
        
        # LLM Model selection
        llm_label = QLabel("LLM Model:")
        self.llm_combo = QComboBox()
        for model_name, config in AVAILABLE_LLM_MODELS.items():
            self.llm_combo.addItem(
                f"{model_name} - {config['description']}", 
                model_name
            )
        # Set default to Phi-3.5-mini
        default_index = list(AVAILABLE_LLM_MODELS.keys()).index("Phi-3.5-mini")
        self.llm_combo.setCurrentIndex(default_index)
        self.llm_combo.currentIndexChanged.connect(self.on_llm_model_changed)
        
        second_header_layout.addWidget(llm_label)
        second_header_layout.addWidget(self.llm_combo)
        
        # Context input
        context_label = QLabel("Context:")
        self.context_input = QLineEdit()
        self.context_input.setPlaceholderText("e.g., Medical discussion, Technical meeting")
        second_header_layout.addWidget(context_label)
        second_header_layout.addWidget(self.context_input)
        
        layout.addLayout(second_header_layout)
        
        # Audio level indicator
        self.level_bar = QProgressBar()
        self.level_bar.setMaximum(100)
        self.level_bar.setTextVisible(False)
        self.level_bar.setFixedHeight(10)
        layout.addWidget(self.level_bar)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.record_button = QPushButton("Start Recording")
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_text)
        
        self.correct_button = QPushButton("Correct Text")
        self.correct_button.clicked.connect(self.correct_text)
        
        button_layout.addWidget(self.record_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.correct_button)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
        # Text display area with splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Raw transcription
        raw_container = QWidget()
        raw_layout = QVBoxLayout(raw_container)
        
        # Raw header with label and copy button
        raw_header = QHBoxLayout()
        raw_label = QLabel("Raw Transcription:")
        self.copy_raw_button = QPushButton("Copy")
        self.copy_raw_button.clicked.connect(self.copy_raw_text)
        raw_header.addWidget(raw_label)
        raw_header.addStretch()
        raw_header.addWidget(self.copy_raw_button)
        raw_layout.addLayout(raw_header)
        
        self.raw_text_edit = QTextEdit()
        self.raw_text_edit.setReadOnly(True)
        raw_layout.addWidget(self.raw_text_edit)
        
        # Corrected text
        corrected_container = QWidget()
        corrected_layout = QVBoxLayout(corrected_container)
        
        # Corrected header with label and copy button
        corrected_header = QHBoxLayout()
        corrected_label = QLabel("Corrected Text:")
        self.copy_corrected_button = QPushButton("Copy")
        self.copy_corrected_button.clicked.connect(self.copy_corrected_text)
        corrected_header.addWidget(corrected_label)
        corrected_header.addStretch()
        corrected_header.addWidget(self.copy_corrected_button)
        corrected_layout.addLayout(corrected_header)
        
        self.corrected_text_edit = QTextEdit()
        corrected_layout.addWidget(self.corrected_text_edit)
        
        splitter.addWidget(raw_container)
        splitter.addWidget(corrected_container)
        splitter.setSizes([450, 450])
        
        layout.addWidget(splitter)
        
        # Status bar
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        
        # Apply dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QLabel {
                color: #ffffff;
                font-size: 14px;
            }
            QTextEdit {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555;
                font-size: 14px;
                font-family: Monaco, Menlo, monospace;
            }
            QComboBox, QLineEdit {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555;
                padding: 5px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #ffffff;
                margin-right: 5px;
            }
            QPushButton {
                background-color: #0d7377;
                color: white;
                border: none;
                padding: 8px 16px;
                font-size: 14px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #14a085;
            }
            QCheckBox {
                color: #ffffff;
                font-size: 14px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QProgressBar {
                background-color: #3c3c3c;
                border: 1px solid #555;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
    
    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        # Space to start/stop recording
        record_action = QAction("Record", self)
        record_action.setShortcut(QKeySequence(Qt.Key.Key_Space))
        record_action.triggered.connect(self.toggle_recording)
        self.addAction(record_action)
        
        # Escape to cancel recording
        cancel_action = QAction("Cancel", self)
        cancel_action.setShortcut(QKeySequence(Qt.Key.Key_Escape))
        cancel_action.triggered.connect(self.cancel_recording)
        self.addAction(cancel_action)
        
        # Cmd+S to save
        save_action = QAction("Save", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_transcription)
        self.addAction(save_action)
    
    def toggle_recording(self):
        """Start or stop recording"""
        if self.transcription_thread and self.transcription_thread.isRunning():
            self.stop_recording()
        else:
            self.start_recording()
    
    def start_recording(self):
        """Start recording audio"""
        # Update UI
        self.record_button.setText("Stop Recording")
        self.record_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
        """)
        self.status_label.setText("Recording...")
        
        # Get selected model
        model_name = self.model_combo.currentData()
        if model_name != self.pipeline.transcriber.model_name:
            self.pipeline.set_model(model_name)
        
        # Clear previous text
        self.raw_text_edit.clear()
        
        # Get selected microphone
        device_id = self.mic_combo.currentData() if self.mic_combo.currentIndex() >= 0 else None
        
        # Start recording in background thread
        self.transcription_thread = TranscriptionThread(self.pipeline, device_id)
        self.transcription_thread.textReady.connect(self.on_transcription_ready)
        self.transcription_thread.errorOccurred.connect(self.on_error)
        self.transcription_thread.levelUpdate.connect(self.update_level)
        self.transcription_thread.start()
    
    def stop_recording(self):
        """Stop recording and transcribe"""
        if self.transcription_thread:
            self.transcription_thread.stop_recording()
            self.status_label.setText("Transcribing...")
            
            # Reset button
            self.record_button.setText("Start Recording")
            self.record_button.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    font-size: 16px;
                    font-weight: bold;
                    padding: 10px;
                    border-radius: 5px;
                }
            """)
    
    def cancel_recording(self):
        """Cancel current recording"""
        try:
            if self.transcription_thread and self.transcription_thread.isRunning():
                # Stop recording gracefully first
                self.transcription_thread.is_recording = False
                # Wait a bit for thread to finish
                if not self.transcription_thread.wait(500):  # 500ms timeout
                    # If it doesn't finish, terminate it
                    self.transcription_thread.terminate()
                    self.transcription_thread.wait()  # Wait for termination
                
                self.transcription_thread = None
                self.record_button.setText("Start Recording")
                self.record_button.setStyleSheet("""
                    QPushButton {
                        background-color: #4CAF50;
                        color: white;
                        font-size: 16px;
                        font-weight: bold;
                        padding: 10px;
                        border-radius: 5px;
                    }
                """)
                self.status_label.setText("Recording cancelled")
                self.level_bar.setValue(0)
        except Exception as e:
            print(f"Error in cancel_recording: {e}")
    
    def on_transcription_ready(self, text: str):
        """Handle transcription result"""
        self.raw_text_edit.setText(text)
        self.status_label.setText("Transcription complete")
        self.level_bar.setValue(0)
        
        # Auto-correct if enabled
        if self.auto_correct_check.isChecked() and text.strip():
            self.correct_text()
    
    def correct_text(self):
        """Correct the transcribed text"""
        raw_text = self.raw_text_edit.toPlainText()
        if not raw_text.strip():
            return
        
        self.status_label.setText("Correcting text...")
        context = self.context_input.text()
        
        # Run correction in background
        QTimer.singleShot(0, lambda: self._do_correction(raw_text, context))
    
    def _do_correction(self, text: str, context: str):
        """Perform text correction"""
        try:
            corrected = self.pipeline.correct_text(text, context)
            self.corrected_text_edit.setText(corrected)
            self.status_label.setText("Text correction complete")
        except Exception as e:
            self.on_error(f"Correction failed: {str(e)}")
    
    def update_level(self, level: float):
        """Update audio level indicator"""
        # Convert to percentage (0-100)
        percentage = min(int(level * 1000), 100)
        self.level_bar.setValue(percentage)
    
    def on_model_changed(self):
        """Handle model selection change"""
        model_name = self.model_combo.currentData()
        self.status_label.setText(f"Model changed to {model_name}")
    
    def clear_text(self):
        """Clear all text fields"""
        try:
            # Stop any running recording first
            if self.transcription_thread and self.transcription_thread.isRunning():
                self.cancel_recording()
            
            self.raw_text_edit.clear()
            self.corrected_text_edit.clear()
            self.context_input.clear()
            self.status_label.setText("Ready")
        except Exception as e:
            print(f"Error in clear_text: {e}")
            self.on_error(f"Clear failed: {str(e)}")
    
    def save_transcription(self):
        """Save transcription to file"""
        text = self.corrected_text_edit.toPlainText()
        if not text:
            text = self.raw_text_edit.toPlainText()
        
        if not text:
            QMessageBox.warning(self, "Warning", "No text to save")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Transcription", "", "Text Files (*.txt)"
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(text)
                self.status_label.setText(f"Saved to {filename}")
            except Exception as e:
                self.on_error(f"Save failed: {str(e)}")
    
    def on_error(self, error_msg: str):
        """Handle errors"""
        self.status_label.setText(f"Error: {error_msg}")
        QMessageBox.critical(self, "Error", error_msg)
    
    def refresh_microphones(self):
        """Refresh the list of available microphones"""
        self.mic_combo.clear()
        
        # Get list of audio input devices
        devices = sd.query_devices()
        current_default = sd.default.device[0]  # Default input device
        
        default_index = 0
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:  # Only input devices
                device_name = f"{device['name']} ({device['hostapi']})"
                self.mic_combo.addItem(device_name, i)
                
                # Track default device
                if i == current_default:
                    default_index = self.mic_combo.count() - 1
        
        # Set to default device
        self.mic_combo.setCurrentIndex(default_index)
    
    def on_mic_changed(self):
        """Handle microphone selection change"""
        if self.mic_combo.currentIndex() >= 0:
            device_id = self.mic_combo.currentData()
            device_name = self.mic_combo.currentText()
            self.status_label.setText(f"Microphone changed to {device_name}")
    
    def copy_raw_text(self):
        """Copy raw transcription to clipboard"""
        text = self.raw_text_edit.toPlainText()
        if text:
            clipboard = QApplication.clipboard()
            clipboard.setText(text)
            self.status_label.setText("Raw text copied to clipboard")
    
    def copy_corrected_text(self):
        """Copy corrected text to clipboard"""
        text = self.corrected_text_edit.toPlainText()
        if text:
            clipboard = QApplication.clipboard()
            clipboard.setText(text)
            self.status_label.setText("Corrected text copied to clipboard")
    
    def on_llm_model_changed(self):
        """Handle LLM model selection change"""
        model_name = self.llm_combo.currentData()
        self.status_label.setText(f"LLM model changed to {model_name}")
        # Update the pipeline with new LLM model
        self.pipeline.set_llm_model(model_name)