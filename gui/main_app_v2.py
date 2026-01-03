"""
Edge AI Video Analytics - Main GUI Application
Clean, modular version - each tab is in its own module
"""

import customtkinter as ctk
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gui_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Reduce noise
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
API_URL = "http://localhost:8002"


class EdgeAIManager(ctk.CTk):
    """Main application - coordinates all tabs"""
    
    def __init__(self):
        super().__init__()
        logger.info("Starting Edge AI Manager v2")
        
        self.title("Edge AI Video Analytics - Management Console v2")
        self.geometry("1400x800")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # State
        self.api_process = None
        self.training_process = None
        self.camera_running = False
        self.camera_thread = None
        
        # Initialize dataset manager (shared across tabs)
        from dataset_manager import DatasetManager
        self.dataset_manager = DatasetManager()
        
        # Create tabview
        self.tabview = ctk.CTkTabview(self, width=1380, height=780)
        self.tabview.pack(padx=10, pady=10, fill="both", expand=True)
        
        # Add tabs
        self.tab_api = self.tabview.add("🌐 API")
        self.tab_dataset = self.tabview.add("📁 Datasets")
        self.tab_training = self.tabview.add("🎯 Training")
        self.tab_export = self.tabview.add("📦 Export")
        self.tab_test = self.tabview.add("🧪 Test")
        self.tab_monitoring = self.tabview.add("📊 Monitoring")
        
        # Setup each tab using modular classes
        logger.info("Initializing Edge AI Manager")
        self._setup_all_tabs()
        
        # Bind close event
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        logger.info("Edge AI Manager initialized successfully")
    
    def _setup_all_tabs(self):
        """Setup all tabs using their respective modules"""
        from tabs.api_tab import APITab
        from tabs.dataset_tab import DatasetTab
        from tabs.training_tab import TrainingTab
        from tabs.export_tab import ExportTab
        from tabs.test_tab import TestTab
        from tabs.monitoring_tab import MonitoringTab
        
        # Initialize each tab
        self.api_tab_instance = APITab(self, self.tab_api)
        self.api_tab_instance.setup()
        
        self.dataset_tab_instance = DatasetTab(self, self.tab_dataset)
        self.dataset_tab_instance.dataset_manager = self.dataset_manager
        self.dataset_tab_instance.setup()
        
        self.training_tab_instance = TrainingTab(self, self.tab_training)
        self.training_tab_instance.setup()
        
        self.export_tab_instance = ExportTab(self, self.tab_export)
        self.export_tab_instance.setup()
        
        self.test_tab_instance = TestTab(self, self.tab_test)
        self.test_tab_instance.setup()
        
        self.monitoring_tab_instance = MonitoringTab(self, self.tab_monitoring)
        self.monitoring_tab_instance.setup()
        
        logger.info("All tabs initialized")
    
    def on_closing(self):
        """Handle application close"""
        logger.info("Closing application")
        
        # Cleanup - stop all processes
        if hasattr(self, 'api_tab_instance'):
            try:
                self.api_tab_instance.cleanup()
            except:
                pass
        
        if hasattr(self, 'training_tab_instance') and hasattr(self.training_tab_instance, 'training_process'):
            if self.training_tab_instance.training_process:
                self.training_tab_instance.training_process.terminate()
        
        if self.camera_running:
            self.camera_running = False
        
        self.destroy()


def main():
    """Entry point"""
    app = EdgeAIManager()
    app.mainloop()


if __name__ == "__main__":
    main()
