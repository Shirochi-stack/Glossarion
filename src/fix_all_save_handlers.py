#!/usr/bin/env python3
"""Fix specific lines to use in-memory config"""

# Lines to replace (all the save handler ones, not line 911)
lines_to_replace = [2175, 2203, 2222, 2248, 2267, 2286, 3002, 3026, 3048, 3069, 3367]

with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for line_num in lines_to_replace:
    # Line numbers are 1-indexed
    idx = line_num - 1
    if idx < len(lines):
        old_line = lines[idx]
        if 'current_config = self.load_config()' in old_line:
            new_line = old_line.replace('current_config = self.load_config()', 
                                      'current_config = self.get_current_config_for_update()')
            lines[idx] = new_line
            print(f"Line {line_num}: Replaced")

with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"\n✅ Fixed {len(lines_to_replace)} save handlers")