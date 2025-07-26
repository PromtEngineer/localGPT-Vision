from PyQt6.QtWidgets import QSystemTrayIcon, QMenu, QApplication
from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtCore import QObject, pyqtSignal
import os

class SystemTray(QObject):
    """System tray integration for background operation"""
    
    showMainWindow = pyqtSignal()
    startRecording = pyqtSignal()
    quitApp = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tray_icon = None
        self.init_tray()
    
    def init_tray(self):
        """Initialize system tray icon"""
        # Create tray icon
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(self.get_icon())
        
        # Create menu
        tray_menu = QMenu()
        
        # Show action
        show_action = QAction("Show", self)
        show_action.triggered.connect(self.showMainWindow.emit)
        tray_menu.addAction(show_action)
        
        # Record action
        record_action = QAction("Quick Record", self)
        record_action.triggered.connect(self.startRecording.emit)
        tray_menu.addAction(record_action)
        
        tray_menu.addSeparator()
        
        # Quit action
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.quitApp.emit)
        tray_menu.addAction(quit_action)
        
        # Set menu and show
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()
        
        # Handle clicks
        self.tray_icon.activated.connect(self.on_tray_activated)
    
    def get_icon(self):
        """Get or create tray icon"""
        # Create a simple icon programmatically
        from PyQt6.QtGui import QPixmap, QPainter, QBrush
        from PyQt6.QtCore import Qt
        
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw microphone icon
        painter.setBrush(QBrush(Qt.GlobalColor.white))
        painter.setPen(Qt.PenStyle.NoPen)
        
        # Mic body
        painter.drawEllipse(10, 5, 12, 18)
        
        # Mic stand
        painter.drawRect(14, 23, 4, 5)
        
        # Mic base
        painter.drawRect(10, 28, 12, 2)
        
        painter.end()
        
        return QIcon(pixmap)
    
    def on_tray_activated(self, reason):
        """Handle tray icon activation"""
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.showMainWindow.emit()
    
    def set_recording_state(self, is_recording: bool):
        """Update tray icon for recording state"""
        if is_recording:
            self.tray_icon.setToolTip("Speech Transcription - Recording...")
            # Could update icon to show recording state
        else:
            self.tray_icon.setToolTip("Speech Transcription")