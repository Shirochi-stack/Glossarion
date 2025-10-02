#!/usr/bin/env python3
"""Fix all save handlers to use in-memory config"""

import re

# Read the file
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace in save handler functions
# These are inside function definitions that save config
replacements = 0

# Pattern to find load_config() calls inside save functions
# We want to replace self.load_config() with self.get_current_config_for_update()
# but ONLY in save handler functions

lines = content.split('\n')
new_lines = []
inside_save_function = False
function_depth = 0

for i, line in enumerate(lines):
    # Check if we're entering a save function
    if 'def save_' in line and ('config' in lines[i+1:i+10] if i < len(lines)-10 else False):
        inside_save_function = True
        function_depth = len(line) - len(line.lstrip())
    
    # Check if we're exiting the function (less indentation)
    if inside_save_function and line.strip() and not line.startswith(' ' * (function_depth + 4)):
        if not line.startswith(' ' * function_depth) or 'def ' in line:
            inside_save_function = False
    
    # Replace if we're in a save function
    if inside_save_function and 'current_config = self.load_config()' in line:
        new_line = line.replace('current_config = self.load_config()', 
                               'current_config = self.get_current_config_for_update()')
        new_lines.append(new_line)
        replacements += 1
        print(f"Line {i+1}: Replaced in save function")
    else:
        new_lines.append(line)

new_content = '\n'.join(new_lines)

# Write back
with open('app.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print(f"\n✅ Fixed {replacements} save handlers to use in-memory config")