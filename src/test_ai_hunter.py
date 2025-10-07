import sys
from PySide6.QtWidgets import QApplication
from ai_hunter_enhanced import AIHunterConfigGUI

app = QApplication(sys.argv)

# Create a minimal config
config = {
    'ai_hunter_config': {
        'enabled': True,
        'sample_size': 3000,
        'retry_attempts': 6,
        'disable_temperature_change': False,
        'thresholds': {
            'exact': 90,
            'text': 35,
            'semantic': 85,
            'structural': 85,
            'character': 90,
            'pattern': 80
        },
        'weights': {
            'exact': 1.5,
            'text': 1.2,
            'semantic': 1.0,
            'structural': 1.0,
            'character': 0.8,
            'pattern': 0.8
        },
        'detection_mode': 'weighted_average',
        'multi_method_requirements': {
            'methods_required': 3,
            'min_methods': ['semantic', 'structural']
        },
        'preprocessing': {
            'remove_html_spacing': True,
            'normalize_unicode': True,
            'ignore_case': True,
            'remove_extra_whitespace': True
        },
        'edge_filters': {
            'min_text_length': 500,
            'max_length_ratio': 1.3,
            'min_length_ratio': 0.7
        },
        'language_detection': {
            'enabled': False,
            'target_language': 'english',
            'threshold_characters': 500,
            'languages': {
                'english': ['en'],
                'japanese': ['ja', 'jp']
            }
        }
    }
}

try:
    gui = AIHunterConfigGUI(None, config, None)
    gui.show_ai_hunter_config()
    sys.exit(app.exec())
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
