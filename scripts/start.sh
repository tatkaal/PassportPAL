#!/bin/bash
# PassportPAL Web Application - Start Script
# This script manages the startup of the web application using Docker Compose

# Get the current script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
DOCKER_COMPOSE_FILE="$ROOT_DIR/docker-compose.yml"

# Set up color for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Print banner
echo -e "${CYAN}==================================${NC}"
echo -e "${CYAN}  PassportPAL Web Application    ${NC}"
echo -e "${CYAN}==================================${NC}"
echo -e "${CYAN}Starting...${NC}"
echo ""

# Check if Docker is installed and running
if ! command -v docker &>/dev/null; then
    echo -e "${RED}Error: Docker is not installed.${NC}"
    echo -e "${YELLOW}Please install Docker first: https://docs.docker.com/get-docker/${NC}"
    exit 1
fi

# Check if Docker service is running
if ! docker info &>/dev/null; then
    echo -e "${RED}Error: Docker service is not running.${NC}"
    echo -e "${YELLOW}Please start Docker service and try again.${NC}"
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &>/dev/null && ! docker compose version &>/dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed.${NC}"
    echo -e "${YELLOW}Please install Docker Compose first: https://docs.docker.com/compose/install/${NC}"
    exit 1
fi

# Check for existing model files
BACKEND_DIR="$ROOT_DIR/backend"
MODEL_SCRIPT="$SCRIPT_DIR/download_models.sh"

# Download models if needed
if [ -f "$MODEL_SCRIPT" ]; then
    echo -e "${CYAN}Checking for ML models...${NC}"
    bash "$MODEL_SCRIPT"
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to download models. Please check your internet connection and try again.${NC}"
        exit 1
    fi
fi

# Check ports availability
PORT_80_STATUS=$(netstat -tuln | grep ":80 ")
PORT_5000_STATUS=$(netstat -tuln | grep ":5000 ")

if [ ! -z "$PORT_80_STATUS" ]; then
    echo -e "${YELLOW}Warning: Port 80 is already in use by another process.${NC}"
    echo -e "${YELLOW}The frontend service may not start properly.${NC}"
fi

if [ ! -z "$PORT_5000_STATUS" ]; then
    echo -e "${YELLOW}Warning: Port 5000 is already in use by another process.${NC}"
    echo -e "${YELLOW}The backend service may not start properly.${NC}"
fi

# Stop existing containers
EXISTING_CONTAINERS=$(docker ps -a --filter "name=passport-pal-" --format "{{.Names}}")
if [ ! -z "$EXISTING_CONTAINERS" ]; then
    echo -e "${YELLOW}Stopping existing containers...${NC}"
    docker stop $(echo "$EXISTING_CONTAINERS") &>/dev/null
    docker rm $(echo "$EXISTING_CONTAINERS") &>/dev/null
fi

# Check for existing images
BACKEND_IMAGE=$(docker images passport-pal-backend:latest -q)
FRONTEND_IMAGE=$(docker images passport-pal-frontend:latest -q)

# Ask if we should rebuild images
REBUILD="no"
if [ ! -z "$BACKEND_IMAGE" ] || [ ! -z "$FRONTEND_IMAGE" ]; then
    echo -e "${YELLOW}Existing Docker images found. Do you want to rebuild them?${NC}"
    echo -e "${YELLOW}This will ensure you have the latest version, but will take longer.${NC}"
    echo -e "${YELLOW}[y/N]: ${NC}"
    read -r REBUILD
    
    if [[ $REBUILD =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Rebuilding images...${NC}"
        cd "$ROOT_DIR"
        docker-compose -f "$DOCKER_COMPOSE_FILE" build
    else
        echo -e "${YELLOW}Using existing images...${NC}"
    fi
fi

# Start the containers
echo -e "${GREEN}Starting containers...${NC}"
cd "$ROOT_DIR"

# Determine which docker-compose command to use
if docker compose version &>/dev/null; then
    DOCKER_COMPOSE_CMD="docker compose"
else
    DOCKER_COMPOSE_CMD="docker-compose"
fi

$DOCKER_COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" up -d

# Wait for services to be ready
echo -e "${YELLOW}Waiting for services to start...${NC}"

# Check if backend is ready
MAX_ATTEMPTS=30
ATTEMPT=0
BACKEND_READY=false

while [ $ATTEMPT -lt $MAX_ATTEMPTS ] && [ "$BACKEND_READY" = false ]; do
    ATTEMPT=$((ATTEMPT+1))
    echo -e "${YELLOW}Checking backend status (attempt $ATTEMPT/$MAX_ATTEMPTS)...${NC}"
    
    BACKEND_HEALTH=$(docker ps --filter "name=passport-pal-backend" --format "{{.Status}}" | grep -i "healthy")
    
    if [ ! -z "$BACKEND_HEALTH" ]; then
        echo -e "${GREEN}Backend is healthy!${NC}"
        BACKEND_READY=true
    else
        echo -e "${YELLOW}Backend not ready yet, waiting 5 seconds...${NC}"
        sleep 5
    fi
done

if [ "$BACKEND_READY" = false ]; then
    echo -e "${RED}Backend service did not become healthy within the expected time.${NC}"
    echo -e "${YELLOW}Showing backend logs:${NC}"
    $DOCKER_COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" logs backend
    
    echo -e "${YELLOW}Do you want to continue waiting for the backend? [Y/n]: ${NC}"
    read -r CONTINUE
    
    if [[ $CONTINUE =~ ^[Nn]$ ]]; then
        echo -e "${RED}Stopping services...${NC}"
        $DOCKER_COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" down
        exit 1
    fi
fi

# Check if frontend is ready
FRONTEND_READY=false
ATTEMPT=0

while [ $ATTEMPT -lt $MAX_ATTEMPTS ] && [ "$FRONTEND_READY" = false ]; do
    ATTEMPT=$((ATTEMPT+1))
    echo -e "${YELLOW}Checking frontend status (attempt $ATTEMPT/$MAX_ATTEMPTS)...${NC}"
    
    FRONTEND_STATUS=$(docker ps --filter "name=passport-pal-frontend" --format "{{.Status}}")
    
    if [[ $FRONTEND_STATUS == *"Up"* ]]; then
        # Try to connect to the frontend
        if curl -s http://localhost &>/dev/null; then
            echo -e "${GREEN}Frontend is up and running!${NC}"
            FRONTEND_READY=true
        else
            echo -e "${YELLOW}Frontend container is running but the service is not responding yet.${NC}"
        fi
    else
        echo -e "${YELLOW}Frontend not ready yet, waiting 5 seconds...${NC}"
    fi
    
    if [ "$FRONTEND_READY" = false ]; then
        sleep 5
    fi
done

if [ "$FRONTEND_READY" = false ]; then
    echo -e "${RED}Frontend service did not become ready within the expected time.${NC}"
    echo -e "${YELLOW}Showing frontend logs:${NC}"
    $DOCKER_COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" logs frontend
    
    echo -e "${YELLOW}Do you want to restart the frontend? [Y/n]: ${NC}"
    read -r RESTART
    
    if [[ ! $RESTART =~ ^[Nn]$ ]]; then
        echo -e "${YELLOW}Restarting frontend...${NC}"
        $DOCKER_COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" restart frontend
        
        echo -e "${YELLOW}Frontend restarted. Please check http://localhost in your browser.${NC}"
    fi
fi

# Display access information
echo -e "${GREEN}===================================================${NC}"
echo -e "${GREEN}  PassportPAL Web Application is now running!      ${NC}"
echo -e "${GREEN}===================================================${NC}"
echo -e "${CYAN}Access the web interface at: ${NC}http://localhost"
echo -e "${CYAN}Backend API available at: ${NC}http://localhost:5000"
echo ""
echo -e "${YELLOW}To stop the application, run: docker-compose down${NC}"
echo -e "${YELLOW}or use the script: ./scripts/stop.sh${NC}"
echo ""
