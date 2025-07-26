import sys
import os

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set Qt plugin path for macOS
import platform
if platform.system() == "Darwin":  # macOS
    # Try to find PyQt6 installation
    try:
        import PyQt6
        pyqt6_path = os.path.dirname(PyQt6.__file__)
        plugin_path = os.path.join(pyqt6_path, "Qt6", "plugins", "platforms")
        if os.path.exists(plugin_path):
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_path
    except:
        pass

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from .gui.main_window import MainWindow
from .gui.system_tray import SystemTray

def main():
    """Main application entry point"""
    # Enable high DPI support
    if hasattr(Qt.ApplicationAttribute, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
    
    app = QApplication(sys.argv)
    app.setApplicationName("Speech Transcription")
    app.setOrganizationName("SpeechTranscription")
    
    # Create main window
    window = MainWindow()
    
    # Create system tray
    tray = SystemTray()
    tray.showMainWindow.connect(window.show)
    tray.quitApp.connect(app.quit)
    
    # Show window
    window.show()
    
    # Run app
    sys.exit(app.exec())

if __name__ == "__main__":
    main()