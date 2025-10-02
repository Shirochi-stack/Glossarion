#!/usr/bin/env python3
"""Replace all self.config.get() with self.get_config_value()"""

import re

def fix_config_references():
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Count occurrences
    pattern = r'self\.config\.get\('
    count = len(re.findall(pattern, content))
    print(f"Found {count} occurrences of self.config.get(")
    
    # Don't replace the ones in __init__ before profiles are set (lines around 215)
    # and the one in get_config_value itself (around line 925)
    lines = content.split('\n')
    new_lines = []
    
    for i, line in enumerate(lines):
        line_no = i + 1
        # Skip the specific lines in __init__ and the definition
        if line_no in [215, 925]:  # Keep these as-is
            new_lines.append(line)
        else:
            # Replace self.config.get( with self.get_config_value(
            new_line = line.replace('self.config.get(', 'self.get_config_value(')
            new_lines.append(new_line)
    
    new_content = '\n'.join(new_lines)
    
    # Count after replacement
    new_count = len(re.findall(pattern, new_content))
    print(f"After replacement: {new_count} occurrences remaining")
    
    # Save
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ Replacement complete!")

if __name__ == "__main__":
    fix_config_references()