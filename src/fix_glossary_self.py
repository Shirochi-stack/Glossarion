#!/usr/bin/env python3
"""Remove all 'self' references from GlossaryManager.py"""
import re

file_path = 'C:/Users/omarn/Projects/Glossarion/src/GlossaryManager.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Remove def __init__(self): block entirely
content = re.sub(r'def __init__\(self\):.*?(?=\ndef )', '', content, flags=re.DOTALL)

# Remove 'self, ' from function definitions
content = re.sub(r'def (\w+)\(self, ', r'def \1(', content)

# Remove 'self)' from function definitions (functions with only self parameter)
content = re.sub(r'def (\w+)\(self\):', r'def \1():', content)

# Replace self._results_lock with module-level _results_lock
content = content.replace('self._results_lock', '_results_lock')
content = content.replace('self._file_write_lock', '_file_write_lock')

# Replace self._filter_text_for_glossary with _filter_text_for_glossary
content = content.replace('self._filter_text_for_glossary', '_filter_text_for_glossary')
content = content.replace('self._', '_')

# Add module-level locks at the top if not already present
if '_results_lock = threading.Lock()' not in content:
    # Find the line after _last_api_submission_time
    content = content.replace(
        '_last_api_submission_time = 0',
        '_last_api_submission_time = 0\n_results_lock = threading.Lock()\n_file_write_lock = threading.Lock()'
    )

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print('✅ Removed all self references from GlossaryManager.py')
print('   - Converted methods to module functions')
print('   - Changed instance variables to module-level')
