#!/usr/bin/env python3
"""
development_windows.py - PassportPAL Web Application (Non-Docker) for Windows

This script:
  - Checks Python version (requires 3.10+)
  - Verifies Node.js/npm installation
  - Sets up a virtual environment in the backend folder and installs backend dependencies
  - Checks for and installs gdown if necessary
  - Downloads ML models from Google Drive using an integrated Python function
  - Installs frontend dependencies via npm
  - Starts the backend (using uvicorn) and waits for it to be ready
  - Starts the frontend (npm run dev)
  
Press Ctrl+C to stop the application.
"""

import os
import sys
import subprocess
import time
import requests
import shutil
import tempfile

def print_banner():
    banner = """
=================================
 PassportPAL Web Application 
 Non-Docker Version Launcher 
=================================
"""
    print(banner)

def check_python_version():
    if sys.version_info < (3, 10):
        print(f"Python 3.10+ required (found {sys.version_info.major}.{sys.version_info.minor}).")
        sys.exit(1)

def command_exists(cmd):
    return shutil.which(cmd) is not None

def get_npm_command():
    # On Windows, npm is typically available as npm.cmd rather than npm.
    npm = shutil.which("npm.cmd")
    if npm:
        return "npm.cmd"
    npm = shutil.which("npm")
    if npm:
        return "npm"
    print("Node.js/npm not found.")
    sys.exit(1)

def check_node_npm():
    if not command_exists("npm") and not command_exists("npm.cmd"):
        print("Node.js/npm not installed.")
        sys.exit(1)

def setup_virtualenv(backend_dir, py_cmd):
    venv_dir = os.path.join(backend_dir, "venv")
    if not os.path.exists(venv_dir):
        print("Setting up virtual environment...")
        subprocess.check_call([py_cmd, "-m", "venv", venv_dir], cwd=backend_dir)
    return venv_dir

def run_command(command, cwd=None, shell=False):
    try:
        subprocess.check_call(command, cwd=cwd, shell=shell)
    except subprocess.CalledProcessError:
        print(f"Command failed: {' '.join(command) if isinstance(command, list) else command}")
        sys.exit(1)

def install_backend_dependencies(venv_python, backend_dir):
    print("Upgrading pip...")
    run_command([venv_python, "-m", "pip", "install", "--upgrade", "pip"], cwd=backend_dir)
    print("Installing backend dependencies...")
    run_command([venv_python, "-m", "pip", "install", "-r", "requirements.txt"], cwd=backend_dir)

def check_and_install_gdown(venv_python, backend_dir):
    try:
        subprocess.check_call([venv_python, "-c", "import gdown"], cwd=backend_dir,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print("Installing gdown...")
        run_command([venv_python, "-m", "pip", "install", "gdown"], cwd=backend_dir)

def download_models():
    """
    Downloads the required ML models from Google Drive using gdown.
    Expected files:
      - custom_instance_segmentation.pt
      - custom_cnn_model_scripted.pt
      - custom_cnn_model_metadata.json
    """
    # Assume this script is in <root>/development/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
    backend_dir = os.path.join(root_dir, "backend")
    models_dir = os.path.join(backend_dir, "models")
    
    required_files = [
        "custom_instance_segmentation.pt",
        "custom_cnn_model_scripted.pt",
        "custom_cnn_model_metadata.json"
    ]
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created models directory: {models_dir}")
    
    if all(os.path.exists(os.path.join(models_dir, f)) for f in required_files):
        print("Models already exist, skipping download.")
        return
    
    print("Downloading models from Google Drive...")
    tmp_dir = tempfile.mkdtemp()
    
    try:
        download_url = "https://drive.google.com/drive/folders/1qG6xU7eGEwTXxQWP5L6s2zuJ7FXs3SQB?usp=sharing"
        subprocess.check_call([sys.executable, "-m", "gdown", download_url, "--folder"], cwd=tmp_dir)
        
        downloaded_dir = None
        for item in os.listdir(tmp_dir):
            full_path = os.path.join(tmp_dir, item)
            if os.path.isdir(full_path) and item.endswith("models"):
                downloaded_dir = full_path
                break
        
        if not downloaded_dir:
            print("Error: Could not find downloaded models directory.")
            sys.exit(1)
        
        for f in required_files:
            if not os.path.exists(os.path.join(downloaded_dir, f)):
                print(f"Error: Missing file '{f}' after download.")
                sys.exit(1)
        
        for f in required_files:
            shutil.move(os.path.join(downloaded_dir, f), os.path.join(models_dir, f))
        print(f"Models successfully downloaded and moved to {models_dir}!")
        
    except subprocess.CalledProcessError:
        print("Error: Failed to download files via gdown.")
        sys.exit(1)
    finally:
        shutil.rmtree(tmp_dir)

def install_frontend_dependencies(frontend_dir, npm_cmd):
    print("Installing frontend dependencies...")
    run_command([npm_cmd, "install"], cwd=frontend_dir)

def start_backend(backend_dir, venv_dir):
    # Use the PowerShell approach from your working PS1 script.
    activate_script = os.path.join(venv_dir, "Scripts", "Activate.ps1")
    # Construct the PowerShell command:
    # Change directory to backend, activate the virtual environment, then start uvicorn.
    backend_cmd = (
        f"cd '{backend_dir}'; "
        f"& '{activate_script}'; "
        f"python -m uvicorn main:app --host 0.0.0.0 --port 5000 --reload"
    )
    print("Starting backend...")
    subprocess.Popen(
        ["powershell", "-NoExit", "-Command", backend_cmd],
        cwd=backend_dir,
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )

def wait_for_backend(max_attempts=5):
    backend_ready = False
    attempt = 0
    print("Waiting for backend to be ready...")
    while not backend_ready and attempt < max_attempts:
        attempt += 1
        time.sleep(10)
        try:
            response = requests.get("http://localhost:5000/api/status", timeout=5)
            if response.status_code == 200:
                print("Backend is ready!")
                backend_ready = True
                break
        except Exception:
            print(f"Backend not ready yet (Attempt {attempt}/{max_attempts})...")
    if not backend_ready:
        print(f"Backend failed to start after {max_attempts} attempts.")
        sys.exit(1)

def start_frontend(frontend_dir):
    # Use a PowerShell command similar to the working PS1 script.
    frontend_cmd = f"cd '{frontend_dir}'; npm run dev"
    print("Starting frontend...")
    subprocess.Popen(
        ["powershell", "-NoExit", "-Command", frontend_cmd],
        cwd=frontend_dir,
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )

def main():
    print_banner()
    check_python_version()
    
    if command_exists("python"):
        py_cmd = "python"
    elif command_exists("python3"):
        py_cmd = "python3"
    else:
        print("Python is not installed.")
        sys.exit(1)
        
    check_node_npm()
    npm_cmd = get_npm_command()
    
    # Define directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
    backend_dir = os.path.join(root_dir, "backend")
    frontend_dir = os.path.join(root_dir, "frontend")
    
    # Set up virtual environment and install backend dependencies
    venv_dir = setup_virtualenv(backend_dir, py_cmd)
    venv_python = os.path.join(venv_dir, "Scripts", "python")
    install_backend_dependencies(venv_python, backend_dir)
    check_and_install_gdown(venv_python, backend_dir)
    
    # Download models using the integrated Python function
    download_models()
    
    # Install frontend dependencies via npm
    install_frontend_dependencies(frontend_dir, npm_cmd)
    
    # Start backend and wait for it to be live
    start_backend(backend_dir, venv_dir)
    wait_for_backend(max_attempts=5)
    
    # Start frontend
    start_frontend(frontend_dir)
    
    print("\nApplication running:")
    print("Frontend: http://localhost:5173")
    print("Backend: http://localhost:5000")
    print("\nPress Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping application...")
        sys.exit(0)

if __name__ == '__main__':
    main()
