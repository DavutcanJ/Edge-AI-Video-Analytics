"""
GUI Tabs Module
Each tab is a separate class for better maintainability
"""

from .api_tab import APITab
from .dataset_tab import DatasetTab
from .training_tab import TrainingTab
from .export_tab import ExportTab
from .test_tab import TestTab
from .monitoring_tab import MonitoringTab

__all__ = [
    'APITab',
    'DatasetTab',
    'TrainingTab', 
    'ExportTab',
    'TestTab',
    'MonitoringTab'
]
