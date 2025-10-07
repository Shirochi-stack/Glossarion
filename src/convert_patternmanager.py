#!/usr/bin/env python3
"""Convert PatternManager class to module-level constants for better pickling"""
import re

input_file = 'C:/Users/omarn/Projects/Glossarion/src/PatternManager.py'

with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
skip_next = False
in_class = False

for i, line in enumerate(lines):
    # Skip class definition and docstring
    if 'class PatternManager:' in line:
        in_class = True
        # Add comment explaining the change
        new_lines.append('# All constants moved to module level for ProcessPoolExecutor compatibility\n')
        new_lines.append('# (ProcessPoolExecutor workers need to pickle/unpickle these patterns)\n')
        new_lines.append('\n')
        continue
    
    # Skip the docstring after class def
    if in_class and '"""' in line and 'Centralized' in line:
        in_class = False
        continue
    
    # Remove 4-space indentation from class attributes
    if line.startswith('    ') and not line.strip().startswith('#'):
        new_lines.append(line[4:])
    else:
        new_lines.append(line)

with open(input_file, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print('✅ Converted PatternManager class to module-level constants')
print(f'   File: {input_file}')
