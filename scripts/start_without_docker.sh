#!/bin/bash
# PassportPAL Web Application - Start Without Docker
# This script starts both the frontend and backend without using Docker

# Get the current script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"

# Set environment variables
export NODE_ENV=development

# ANSI colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Show banner
echo -e "${CYAN}=================================${NC}"
echo -e "${CYAN}  PassportPAL Web Application   ${NC}"
echo -e "${CYAN}  Non-Docker Version Launcher   ${NC}"
echo -e "${CYAN}=================================${NC}"
echo ""

# Function to check if command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
MISSING_PREREQS=()

if ! command_exists python3; then
  MISSING_PREREQS+=("Python 3.10 or higher")
fi

if ! command_exists npm; then
  MISSING_PREREQS+=("Node.js and npm")
fi

if [ ${#MISSING_PREREQS[@]} -gt 0 ]; then
  echo -e "${RED}Error: Missing prerequisites:${NC}"
  for prereq in "${MISSING_PREREQS[@]}"; do
    echo -e "${RED} - $prereq${NC}"
  done
  echo -e "${RED}Please install missing prerequisites and try again.${NC}"
  exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_VER_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_VER_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ $PYTHON_VER_MAJOR -lt 3 ] || [ $PYTHON_VER_MAJOR -eq 3 -a $PYTHON_VER_MINOR -lt 10 ]; then
  echo -e "${RED}Error: Python 3.10 or higher is required. Found version $PYTHON_VERSION${NC}"
  exit 1
fi

# Create and activate Python virtual environment for the backend
echo -e "${CYAN}Setting up Python virtual environment...${NC}"

# Check if venv exists, create if it doesn't
VENV_DIR="$BACKEND_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
  echo -e "${YELLOW}Creating new Python virtual environment...${NC}"
  python3 -m venv "$VENV_DIR"
  if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to create virtual environment.${NC}"
    exit 1
  fi
fi

# Activate the virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"

# Install backend requirements
echo -e "${CYAN}Installing backend dependencies...${NC}"
cd "$BACKEND_DIR"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [ $? -ne 0 ]; then
  echo -e "${RED}Failed to install backend dependencies.${NC}"
  deactivate
  exit 1
fi

# Install gdown if it's not already installed
if ! python -c "import gdown" &> /dev/null; then
    echo -e "${YELLOW}Installing gdown for model downloads...${NC}"
    python -m pip install gdown || {
        echo -e "${RED}Failed to install gdown.${NC}"
        deactivate
        exit 1
    }
fi

# Download ML models
echo -e "${CYAN}Checking for ML models...${NC}"
MODEL_SCRIPT="$SCRIPT_DIR/download_models.sh"

if [ -f "$MODEL_SCRIPT" ]; then
  bash "$MODEL_SCRIPT"
  if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to download models. Please check your internet connection and try again.${NC}"
    deactivate
    exit 1
  fi
else
  echo -e "${RED}Model download script not found at: $MODEL_SCRIPT${NC}"
  deactivate
  exit 1
fi

# Setup frontend
echo -e "${CYAN}Setting up frontend...${NC}"
cd "$FRONTEND_DIR"

# Install frontend dependencies
echo -e "${YELLOW}Installing frontend dependencies...${NC}"
npm install
if [ $? -ne 0 ]; then
  echo -e "${RED}Failed to install frontend dependencies.${NC}"
  deactivate
  exit 1
fi

# Start backend and frontend in separate processes
echo -e "${GREEN}Starting the application...${NC}"

# Start backend in a new terminal
echo -e "${YELLOW}Starting backend...${NC}"
if command_exists gnome-terminal; then
  gnome-terminal -- bash -c "cd '$BACKEND_DIR' && source '$VENV_DIR/bin/activate' && python -m uvicorn main:app --host 0.0.0.0 --port 5000 --reload; read -p 'Press Enter to close'"
elif command_exists xterm; then
  xterm -e "cd '$BACKEND_DIR' && source '$VENV_DIR/bin/activate' && python -m uvicorn main:app --host 0.0.0.0 --port 5000 --reload; read -p 'Press Enter to close'" &
elif command_exists terminal; then
  # macOS Terminal
  osascript -e "tell app \"Terminal\" to do script \"cd '$BACKEND_DIR' && source '$VENV_DIR/bin/activate' && python -m uvicorn main:app --host 0.0.0.0 --port 5000 --reload; read -p 'Press Enter to close'\""
else
  echo -e "${YELLOW}Couldn't find a suitable terminal emulator. Starting backend in background...${NC}"
  (cd "$BACKEND_DIR" && source "$VENV_DIR/bin/activate" && python -m uvicorn main:app --host 0.0.0.0 --port 5000 --reload) &
  BACKEND_PID=$!
fi

# Wait for backend to be ready
echo -e "${YELLOW}Waiting for backend to start...${NC}"
BACKEND_READY=false
MAX_ATTEMPTS=5
ATTEMPTS=0

while [ "$BACKEND_READY" = false ] && [ $ATTEMPTS -lt $MAX_ATTEMPTS ]; do
  ATTEMPTS=$((ATTEMPTS+1))
  
  if curl -s "http://localhost:5000/api/status" > /dev/null 2>&1; then
    BACKEND_STATUS=$(curl -s "http://localhost:5000/api/status")
    echo -e "${GREEN}Backend is ready!${NC}"
    BACKEND_READY=true
  else
    echo -e "${YELLOW}Waiting for backend to start (attempt $ATTEMPTS of $MAX_ATTEMPTS)...${NC}"
    if [ $ATTEMPTS -lt $MAX_ATTEMPTS ]; then
      echo -e "${YELLOW}Waiting 30 seconds before next check...${NC}"
      sleep 30
    fi
  fi
done

if [ "$BACKEND_READY" = false ]; then
  echo -e "${RED}Backend failed to start after $MAX_ATTEMPTS attempts. Please check the backend window for errors.${NC}"
  exit 1
fi

# Now start the frontend in a new terminal
echo -e "${YELLOW}Starting frontend...${NC}"
if command_exists gnome-terminal; then
  gnome-terminal -- bash -c "cd '$FRONTEND_DIR' && npm run dev; read -p 'Press Enter to close'"
elif command_exists xterm; then
  xterm -e "cd '$FRONTEND_DIR' && npm run dev; read -p 'Press Enter to close'" &
elif command_exists terminal; then
  # macOS Terminal
  osascript -e "tell app \"Terminal\" to do script \"cd '$FRONTEND_DIR' && npm run dev; read -p 'Press Enter to close'\""
else
  echo -e "${YELLOW}Couldn't find a suitable terminal emulator. Starting frontend in background...${NC}"
  (cd "$FRONTEND_DIR" && npm run dev) &
  FRONTEND_PID=$!
fi

# Show instructions
echo -e "\n${GREEN}Application is now running!${NC}"
echo -e "${CYAN}Frontend: http://localhost:3000${NC}"
echo -e "${CYAN}Backend: http://localhost:5000${NC}"
echo -e "\n${YELLOW}Press Ctrl+C to stop the application${NC}"

# Handle cleanup on exit
trap cleanup EXIT
cleanup() {
  echo -e "\n${YELLOW}Shutting down...${NC}"
  if [ ! -z "$BACKEND_PID" ]; then
    kill $BACKEND_PID 2>/dev/null
  fi
  if [ ! -z "$FRONTEND_PID" ]; then
    kill $FRONTEND_PID 2>/dev/null
  fi
  deactivate 2>/dev/null
}

# Keep the script running to maintain the processes
while true; do
  sleep 5
done 