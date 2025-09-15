import os
import subprocess
import sys
import requests

def download_pyinstxtractor():
    """Downloads pyinstxtractor.py if it doesn't exist."""
    if not os.path.exists("pyinstxtractor.py"):
        print("Downloading pyinstxtractor.py...")
        url = "https://raw.githubusercontent.com/extremecoders-re/pyinstxtractor/master/pyinstxtractor.py"
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open("pyinstxtractor.py", "w") as f:
                f.write(response.text)
            print("Download complete.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading pyinstxtractor.py: {e}")
            sys.exit(1)

def recover_code(exe_path):
    """
    Recovers Python scripts from a PyInstaller executable.

    Args:
        exe_path (str): The path to the .exe file.
    """
    if not os.path.exists(exe_path):
        print(f"Error: The file '{exe_path}' was not found.")
        return

    download_pyinstxtractor()

    # 1. Extract the .exe file
    print(f"\n[+] Extracting '{exe_path}'...")
    try:
        subprocess.run([sys.executable, "pyinstxtractor.py", exe_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during extraction: {e}")
        return
    except FileNotFoundError:
        print("Error: Could not find pyinstxtractor.py. Make sure it's in the same directory.")
        return

    # 2. Find the extracted directory
    extracted_dir = f"{exe_path}_extracted"
    if not os.path.isdir(extracted_dir):
        print(f"Error: Extraction failed. Directory '{extracted_dir}' not found.")
        return
    print(f"Extraction successful. Files are in '{extracted_dir}'")

    # 3. Decompile .pyc files
    print("\n[+] Decompiling .pyc files...")
    output_dir = os.path.join(extracted_dir, "decompiled_scripts")
    os.makedirs(output_dir, exist_ok=True)

    decompiled_files = []
    for root, _, files in os.walk(extracted_dir):
        for file in files:
            if file.endswith(".pyc"):
                pyc_path = os.path.join(root, file)
                py_path = os.path.join(output_dir, file.replace(".pyc", ".py"))
                print(f"  - Decompiling {pyc_path} to {py_path}")
                try:
                    subprocess.run(
                        ["uncompyle6", "-o", py_path, pyc_path],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    decompiled_files.append(py_path)
                except subprocess.CalledProcessError as e:
                    print(f"    - Failed to decompile {file}: {e.stderr}")
                except FileNotFoundError:
                    print("Error: 'uncompyle6' not found. Make sure it's installed ('pip install uncompyle6').")
                    return


    if decompiled_files:
        print(f"\n[+] Decompilation complete!")
        print(f"Recovered scripts are in: '{output_dir}'")
    else:
        print("\n[-] No .pyc files were successfully decompiled.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python recover_script.py <your_program.exe>")
    else:
        recover_code(sys.argv[1])