"""
Base Tab Class
All tabs inherit from this for common functionality
"""

import customtkinter as ctk


class BaseTab:
    """Base class for all GUI tabs"""
    
    def __init__(self, app, parent):
        """
        Initialize tab
        
        Args:
            app: Reference to main application instance
            parent: Parent frame (CTkFrame or CTkTabview tab)
        """
        self.app = app
        self.parent = parent
        self.widgets = {}  # Store references to important widgets
    
    def setup(self):
        """Setup tab UI - override in subclasses"""
        raise NotImplementedError("Subclasses must implement setup()")
    
    def cleanup(self):
        """Cleanup when tab is closed - override if needed"""
        pass
    
    def refresh(self):
        """Refresh tab data - override if needed"""
        pass
