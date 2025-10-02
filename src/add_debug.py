#!/usr/bin/env python3
"""Add debug logging to save_config"""

with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the save_config method and add debug logging
for i, line in enumerate(lines):
    if 'def save_config(self, config):' in line:
        # Find the line where we save
        for j in range(i, min(i + 50, len(lines))):
            if 'with open(self.config_file' in lines[j]:
                # Insert debug before
                indent = '                '
                debug_lines = [
                    f'{indent}print(f"DEBUG save_config called with model={{config.get(\'model\')}}, batch_size={{config.get(\'batch_size\')}}")\n',
                    f'{indent}print(f"DEBUG self.config before={{self.config.get(\'model\') if hasattr(self, \'config\') else \'N/A\'}}")\n',
                    f'{indent}print(f"DEBUG self.decrypted_config before={{self.decrypted_config.get(\'model\') if hasattr(self, \'decrypted_config\') else \'N/A\'}}")\n',
                ]
                lines[j:j] = debug_lines
                
                # Also add after update
                for k in range(j + 10, min(j + 30, len(lines))):
                    if 'self.decrypted_config = decrypt_config' in lines[k]:
                        debug_lines2 = [
                            f'{indent}print(f"DEBUG self.config after={{self.config.get(\'model\')}}")\n',
                            f'{indent}print(f"DEBUG self.decrypted_config after={{self.decrypted_config.get(\'model\')}}")\n',
                        ]
                        lines[k+1:k+1] = debug_lines2
                        break
                break
        break

with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("✅ Added debug logging to save_config")