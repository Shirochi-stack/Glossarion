"""
Test script for GlossaryManager_GUI.py
Tests the PySide6 conversion by creating a minimal environment
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication

# Mock TranslatorGUI class with minimal required attributes
class MockTranslatorGUI:
    def __init__(self):
        self.config = {
            'enable_auto_glossary': False,
            'append_glossary': False,
            'glossary_min_frequency': 2,
            'glossary_max_names': 100,
            'glossary_max_titles': 50,
            'glossary_batch_size': 10,
            'glossary_max_text_size': 0,
            'glossary_max_sentences': 200,
            'manual_glossary_temperature': 0.1,
            'manual_context_limit': 2,
            'glossary_fuzzy_threshold': 0.90,
            'custom_entry_types': {
                'character': {'enabled': True, 'has_gender': True},
                'term': {'enabled': True, 'has_gender': False}
            },
            'custom_glossary_fields': []
        }
        
        self.custom_glossary_fields = []
        self.custom_entry_types = self.config['custom_entry_types']
        
        # Prompts
        self.manual_glossary_prompt = "Test manual prompt"
        self.auto_glossary_prompt = "Test auto prompt"
        self.append_glossary_prompt = "Test append prompt"
        self.glossary_translation_prompt = "Test translation prompt"
        self.glossary_format_instructions = "Test format instructions"
        
        # Mock master (Tkinter window)
        self.master = None
    
    def append_log(self, message):
        """Mock log method"""
        print(f"[LOG] {message}")
    
    def save_config(self, show_message=True):
        """Mock save config"""
        print("[MOCK] Config saved")

# Import the mixin after defining the mock
from GlossaryManager_GUI import GlossaryManagerMixin

# Create a test class that combines the mock with the mixin
class TestGUI(MockTranslatorGUI, GlossaryManagerMixin):
    pass

if __name__ == "__main__":
    # Create QApplication (required for PySide6 dialogs)
    app = QApplication(sys.argv)
    
    # Create mock GUI instance
    gui = TestGUI()
    
    print("✅ GlossaryManager_GUI loaded successfully")
    print("✅ All imports resolved")
    print("✅ Opening Glossary Manager dialog...")
    
    try:
        # Open the glossary manager dialog
        gui.glossary_manager()
        print("✅ Dialog opened successfully!")
    except Exception as e:
        print(f"❌ Error opening dialog: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("✅ All tests passed!")
