#!/usr/bin/env python3
"""
start_unix.py - Integrated Script for Linux/Mac

This script downloads ML models (if needed) and starts the PassportPAL application using Docker Compose.
It performs the following steps:
  1. Ensures that the 'gdown' module is installed.
  2. Downloads required model files from Google Drive if they are missing.
  3. Checks that Docker is running.
  4. Warns if ports 80 or 5000 are in use.
  5. Uses Docker Compose (auto-detecting the proper command) to stop any running containers.
  6. Checks if Docker images exist and prompts the user for rebuild options.
  7. Starts the Docker Compose containers.
  8. Waits for the backend and frontend services to be ready—attempting 5 times, waiting 30 seconds between checks.

Potential failure conditions include missing dependencies, Docker not running, network issues during download,
permission issues running Docker commands, or Docker Compose command misconfiguration.
"""

import os
import sys
import subprocess
import tempfile
import shutil
import time
import socket
import requests

def get_docker_compose_cmd():
    try:
        subprocess.check_output(["docker", "compose", "version"], stderr=subprocess.STDOUT)
        return ["docker", "compose"]
    except subprocess.CalledProcessError:
        try:
            subprocess.check_output(["docker-compose", "version"], stderr=subprocess.STDOUT)
            return ["docker-compose"]
        except subprocess.CalledProcessError:
            print("Error: Docker Compose is not installed.")
            sys.exit(1)

DOCKER_COMPOSE_CMD = get_docker_compose_cmd()

def ensure_gdown():
    try:
        import gdown
    except ImportError:
        print("gdown module not found. Installing it now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        try:
            import gdown
        except ImportError:
            print("Error: gdown module installation failed.")
            sys.exit(1)

def ensure_requests():
    try:
        import requests
    except ImportError:
        print("requests module not found. Installing it now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
        try:
            import requests
        except ImportError:
            print("Error: requests module installation failed.")
            sys.exit(1)

def ensure_docker():
    try:
        subprocess.check_output(["docker", "--version"], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        print("Error: Docker does not appear to be running. Please start Docker.")
        sys.exit(1)

def download_models():
    root_dir = os.path.dirname(os.path.abspath(__file__))
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

def check_docker_running():
    try:
        subprocess.check_output(["docker", "info"], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        print("Error: Docker does not appear to be running. Please start Docker.")
        sys.exit(1)

def check_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def docker_compose_down(root_dir):
    try:
        subprocess.check_call(DOCKER_COMPOSE_CMD + ["down"], cwd=root_dir)
    except subprocess.CalledProcessError:
        print("Warning: Failed to stop existing containers.")

def get_docker_image_exists(image_name):
    try:
        output = subprocess.check_output(["docker", "images", image_name, "-q"]).decode().strip()
        return bool(output)
    except subprocess.CalledProcessError:
        return False

def docker_compose_build(targets=None, no_cache=False, root_dir="."):
    command = DOCKER_COMPOSE_CMD + ["build"]
    if no_cache:
        command.append("--no-cache")
    if targets:
        command.extend(targets)
    subprocess.check_call(command, cwd=root_dir)

def docker_compose_up(root_dir):
    subprocess.check_call(DOCKER_COMPOSE_CMD + ["up", "-d"], cwd=root_dir)

def wait_for_services(max_attempts=5):
    """
    Attempts to check if the backend and frontend services are live.
    It makes 5 attempts, waiting 30 seconds between each.
    The backend health check uses a 30-second timeout.
    """
    attempts = 0
    backend_ready = False
    frontend_ready = False

    while attempts < max_attempts and not (backend_ready and frontend_ready):
        attempts += 1
        print(f"Attempt {attempts} of {max_attempts} to check services...")
        try:
            backend_response = requests.get("http://localhost:5000/api/status", timeout=30)
            if backend_response.status_code == 200:
                if not backend_ready:
                    print("Backend is ready!")
                backend_ready = True
        except Exception:
            print("Backend not ready yet...")
        
        try:
            frontend_response = requests.get("http://localhost", timeout=30)
            if frontend_response.status_code == 200:
                if not frontend_ready:
                    print("Frontend is ready!")
                frontend_ready = True
        except Exception:
            print("Frontend not ready yet...")
        
        if not (backend_ready and frontend_ready):
            print("Waiting for 30 seconds before retrying...")
            time.sleep(30)
    
    return backend_ready, frontend_ready

def restart_container(container_name):
    try:
        subprocess.check_call(["docker", "restart", container_name])
        print(f"{container_name} container restarted.")
    except subprocess.CalledProcessError:
        print(f"Failed to restart {container_name} container.")

def main():
    ensure_gdown()
    ensure_requests()
    ensure_docker()
    download_models()
    check_docker_running()
    
    if check_port_in_use(80):
        print("Warning: Port 80 is already in use. The frontend service might fail to start.")
    if check_port_in_use(5000):
        print("Warning: Port 5000 is already in use. The backend service might fail to start.")
    
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    backend_image_exists = get_docker_image_exists("passport-pal-backend")
    frontend_image_exists = get_docker_image_exists("passport-pal-frontend")
    
    print("Stopping any existing containers...")
    docker_compose_down(root_dir)
    
    rebuild = "partial"
    if backend_image_exists or frontend_image_exists:
        print("\nExisting images found:")
        if backend_image_exists:
            print("- Backend image exists")
        if frontend_image_exists:
            print("- Frontend image exists")
        
        print("\nRebuild options:")
        print("1. Use existing images (fastest)")
        print("2. Rebuild only missing or failed images (recommended)")
        print("3. Force rebuild all images (slowest)")
        option = input("Select an option (1-3) [2]: ").strip()
        if option == "1":
            rebuild = "none"
        elif option == "3":
            rebuild = "all"
    
    if rebuild == "all":
        print("Removing existing images for complete rebuild...")
        if backend_image_exists:
            subprocess.call(["docker", "image", "rm", "passport-pal-backend:latest", "-f"])
        if frontend_image_exists:
            subprocess.call(["docker", "image", "rm", "passport-pal-frontend:latest", "-f"])
        print("Building all containers from scratch...")
        docker_compose_build(no_cache=True, root_dir=root_dir)
    elif rebuild == "partial":
        targets = []
        if not backend_image_exists:
            print("Backend image not found, will build it...")
            targets.append("backend")
        if not frontend_image_exists:
            print("Frontend image not found, will build it...")
            targets.append("frontend")
        if targets:
            print("Building only necessary containers...")
            docker_compose_build(targets=targets, root_dir=root_dir)
        else:
            print("All images exist, skipping build...")
    
    print("Starting containers...")
    docker_compose_up(root_dir)
    
    print("Waiting for services to start...")
    backend_ready, frontend_ready = wait_for_services()
    
    if backend_ready and not frontend_ready:
        print("Frontend failed to start properly, but backend is working.")
        print("You can still use the API directly at http://localhost:5000")
        subprocess.call(["docker", "logs", "passport-pal-frontend", "--tail", "20"])
        restart = input("\nDo you want to try restarting the frontend container? (y/n): ").strip().lower()
        if restart == "y":
            print("Restarting frontend container...")
            restart_container("passport-pal-frontend")
            print("Frontend container restarted. Try accessing http://localhost in your browser.")
    elif not backend_ready:
        print("Error: Backend failed to start properly.")
        subprocess.call(DOCKER_COMPOSE_CMD + ["logs", "backend"])
        sys.exit(1)
    
    print("\nPassportPAL application is now running!")
    if frontend_ready:
        print("Frontend: http://localhost")
    print("Backend API: http://localhost:5000")
    print("\nUseful Docker commands:")
    print("- View logs:        " + " ".join(DOCKER_COMPOSE_CMD) + " logs -f")
    print("- Stop application: " + " ".join(DOCKER_COMPOSE_CMD) + " down")
    print("- Cleanup space:    docker system prune -a")

if __name__ == '__main__':
    main()
