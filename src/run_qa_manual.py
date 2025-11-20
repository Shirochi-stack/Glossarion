import sys
import os
sys.path.append(os.getcwd()) # Add current directory to path

from scan_html_folder import scan_html_folder

def run_test():
    folder_path = os.path.abspath("test")
    
    qa_settings = {
        'target_language': 'english',
        'check_missing_html_tag': True,
        'check_invalid_nesting': True,
        'check_paragraph_structure': True,
        'check_unclosed_html_tags': True,
        'check_translation_artifacts': True,
        'check_encoding_issues': True,
        'check_repetition': True,
        'min_file_length': 0,
        'report_format': 'detailed',
        'auto_save_report': False,
    }
    
    print(f"Starting QA Scan on {folder_path}")
    results = scan_html_folder(folder_path, qa_settings=qa_settings, mode='quick-scan')
    
    print("\n--- RESULTS ---")
    for res in results:
        print(f"File: {res['filename']}")
        if res['issues']:
            print(f"  Issues: {res['issues']}")
        else:
            print("  No issues found.")

if __name__ == "__main__":
    # Make sure to run from the src directory
    if not os.path.exists("test"):
        print("Test directory not found! defaulting to absolute path.")
        os.chdir(r"C:\Users\ADMIN\Dev\Glossarion\src")
        
    run_test()
