"""
Simple API Key Encryption Module for Glossarion
Encrypts only specific API key fields including multi-key support
"""

import os
import json
import base64
from cryptography.fernet import Fernet
from pathlib import Path


class APIKeyEncryption:
    """Simple encryption handler for API keys"""
    
    def __init__(self):
        self.key_file = Path('.glossarion_key')
        self.cipher = self._get_or_create_cipher()
        
        # Define which fields to encrypt
        self.api_key_fields = [
            'api_key',
            'replicate_api_key',
            # Add more field names here if needed
        ]
    
    def _get_or_create_cipher(self):
        """Get existing cipher or create new one"""
        if self.key_file.exists():
            try:
                key = self.key_file.read_bytes()
                return Fernet(key)
            except:
                pass
        
        # Generate new key
        key = Fernet.generate_key()
        self.key_file.write_bytes(key)
        
        # Hide file on Windows
        if os.name == 'nt':
            import ctypes
            ctypes.windll.kernel32.SetFileAttributesW(str(self.key_file), 2)
        else:
            # Restrict permissions on Unix
            os.chmod(self.key_file, 0o600)
        
        return Fernet(key)
    
    def encrypt_value(self, value):
        """Encrypt a single value"""
        try:
            encrypted = self.cipher.encrypt(value.encode())
            return f"ENC:{base64.b64encode(encrypted).decode()}"
        except:
            return value
    
    def decrypt_value(self, value):
        """Decrypt a single value"""
        if not isinstance(value, str) or not value.startswith('ENC:'):
            return value
        
        try:
            encrypted_data = base64.b64decode(value[4:])
            return self.cipher.decrypt(encrypted_data).decode()
        except:
            return value
    
    def encrypt_multi_keys(self, multi_keys):
        """Encrypt API keys in multi_api_keys array"""
        if not isinstance(multi_keys, list):
            return multi_keys
        
        encrypted_keys = []
        for key_entry in multi_keys:
            if isinstance(key_entry, dict):
                encrypted_entry = key_entry.copy()
                # Encrypt the api_key field in each entry
                if 'api_key' in encrypted_entry and encrypted_entry['api_key']:
                    value = encrypted_entry['api_key']
                    if isinstance(value, str) and not value.startswith('ENC:'):
                        encrypted_entry['api_key'] = self.encrypt_value(value)
                encrypted_keys.append(encrypted_entry)
            else:
                encrypted_keys.append(key_entry)
        
        return encrypted_keys
    
    def encrypt_config(self, config):
        """Encrypt specific API key fields including multi-key support"""
        encrypted = config.copy()
        
        # Encrypt regular API key fields
        for field in self.api_key_fields:
            if field in encrypted and encrypted[field]:
                value = encrypted[field]
                # Only encrypt if not already encrypted
                if isinstance(value, str) and not value.startswith('ENC:'):
                    encrypted[field] = self.encrypt_value(value)
        
        # Encrypt multi_api_keys if present
        if 'multi_api_keys' in encrypted:
            encrypted['multi_api_keys'] = self.encrypt_multi_keys(encrypted['multi_api_keys'])
        
        # Encrypt fallback_keys if present (reuse multi_keys method)
        if 'fallback_keys' in encrypted:
            encrypted['fallback_keys'] = self.encrypt_multi_keys(encrypted['fallback_keys'])
        
        return encrypted
    
    def decrypt_config(self, config):
        """Decrypt specific API key fields including multi-key support"""
        decrypted = config.copy()
        
        # Decrypt regular API key fields
        for field in self.api_key_fields:
            if field in decrypted and decrypted[field]:
                decrypted[field] = self.decrypt_value(decrypted[field])
        
        # Decrypt multi_api_keys if present
        if 'multi_api_keys' in decrypted:
            decrypted['multi_api_keys'] = self.decrypt_multi_keys(decrypted['multi_api_keys'])
        
        # Decrypt fallback_keys if present (reuse multi_keys method)
        if 'fallback_keys' in decrypted:
            decrypted['fallback_keys'] = self.decrypt_multi_keys(decrypted['fallback_keys'])
        
        return decrypted
    
    def decrypt_config(self, config):
        """Decrypt specific API key fields including multi-key support"""
        decrypted = config.copy()
        
        # Decrypt regular API key fields
        for field in self.api_key_fields:
            if field in decrypted and decrypted[field]:
                decrypted[field] = self.decrypt_value(decrypted[field])
        
        # Decrypt multi_api_keys if present
        if 'multi_api_keys' in decrypted:
            decrypted['multi_api_keys'] = self.decrypt_multi_keys(decrypted['multi_api_keys'])
        
        return decrypted


# Simple interface functions
_handler = None

def get_handler():
    global _handler
    if _handler is None:
        _handler = APIKeyEncryption()
    return _handler

def encrypt_config(config):
    """Encrypt API keys in config"""
    return get_handler().encrypt_config(config)

def decrypt_config(config):
    """Decrypt API keys in config"""
    return get_handler().decrypt_config(config)

def migrate_config_file(config_file='config.json'):
    """Migrate existing config to encrypted format"""
    try:
        # Read config
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Check if already encrypted
        handler = get_handler()
        needs_encryption = False
        
        # Check regular API key fields
        for field in handler.api_key_fields:
            if field in config and config[field]:
                if isinstance(config[field], str) and not config[field].startswith('ENC:'):
                    needs_encryption = True
                    break
        
        # Check multi_api_keys
        if 'multi_api_keys' in config and isinstance(config['multi_api_keys'], list):
            for key_entry in config['multi_api_keys']:
                if isinstance(key_entry, dict) and 'api_key' in key_entry:
                    if key_entry['api_key'] and not key_entry['api_key'].startswith('ENC:'):
                        needs_encryption = True
                        break
        
        if not needs_encryption:
            print("Config already encrypted or no API keys found.")
            return True
        
        # Backup
        backup_file = f"{config_file}.backup"
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"Created backup: {backup_file}")
        
        # Encrypt
        encrypted = encrypt_config(config)
        
        # Save
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(encrypted, f, ensure_ascii=False, indent=2)
        
        print("✅ Successfully encrypted API keys!")
        
        # Show summary
        if 'multi_api_keys' in config:
            print(f"  - Encrypted {len(config['multi_api_keys'])} multi-key entries")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    # Simple migration script
    import sys
    
    config_file = 'config.json'
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    
    if os.path.exists(config_file):
        print(f"Encrypting API keys in {config_file}...")
        migrate_config_file(config_file)
    else:
        print(f"Config file not found: {config_file}")
