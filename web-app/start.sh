#!/bin/bash
set -e

# Start the PassportPAL Web Application using Docker Compose
echo -e "\e[32mStarting PassportPAL Web Application...\e[0m"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "\e[31mDocker is not running. Please start Docker first.\e[0m"
    exit 1
fi

# Download models if they don't exist
echo -e "\e[36mChecking for ML models...\e[0m"
MODEL_SCRIPT="$(pwd)/backend/download_models.sh"
if [ -f "$MODEL_SCRIPT" ]; then
    # Make script executable
    chmod +x "$MODEL_SCRIPT"
    # Run the script
    bash "$MODEL_SCRIPT"
    if [ $? -ne 0 ]; then
        echo -e "\e[31mFailed to download models. Please check your internet connection and try again.\e[0m"
        exit 1
    fi
else
    echo -e "\e[31mModel download script not found at: $MODEL_SCRIPT\e[0m"
    exit 1
fi

# Check if required ports are available
PORT_80_IN_USE=$(netstat -tuln | grep -c ":80 ")
PORT_5000_IN_USE=$(netstat -tuln | grep -c ":5000 ")

if [ $PORT_80_IN_USE -gt 0 ]; then
    echo -e "\e[33mWarning: Port 80 is already in use. The frontend service might fail to start.\e[0m"
fi

if [ $PORT_5000_IN_USE -gt 0 ]; then
    echo -e "\e[33mWarning: Port 5000 is already in use. The backend service might fail to start.\e[0m"
fi

# Check Docker disk usage before starting
echo -e "\e[36mChecking Docker disk usage before starting...\e[0m"
docker system df

# Stop any existing containers
echo -e "\e[36mStopping any existing containers...\e[0m"
docker compose down || docker-compose down

# Check existing images and ask for rebuild option
BACKEND_EXISTS=$(docker images --format "{{.Repository}}" | grep -c "web-app-backend" || echo "0")
FRONTEND_EXISTS=$(docker images --format "{{.Repository}}" | grep -c "web-app-frontend" || echo "0")

if [ "$BACKEND_EXISTS" -gt 0 ] || [ "$FRONTEND_EXISTS" -gt 0 ]; then
    echo -e "\e[36m\nExisting images found:\e[0m"
    if [ "$BACKEND_EXISTS" -gt 0 ]; then echo -e "- Backend image exists"; fi
    if [ "$FRONTEND_EXISTS" -gt 0 ]; then echo -e "- Frontend image exists"; fi
    
    echo -e "\e[36m\nBuild options:\e[0m"
    echo -e "1. Use existing images (fastest)"
    echo -e "2. Rebuild only missing images (recommended)"
    echo -e "3. Force rebuild all images (slowest)"
    
    read -p "Select an option (1-3) [2]: " BUILD_OPTION
    
    case $BUILD_OPTION in
        1)
            # Use existing images
            BUILD_CMD="docker compose up -d"
            ;;
        3)
            # Force rebuild all
            echo -e "\e[36mRemoving existing images for complete rebuild...\e[0m"
            docker image rm web-app-frontend:latest web-app-backend:latest -f 2>/dev/null || true
            
            # Clean build cache for a fresh build
            echo -e "\e[36mCleaning build cache...\e[0m"
            docker builder prune -f
            
            BUILD_CMD="docker compose build --no-cache && docker compose up -d"
            ;;
        *)
            # Default: Selective rebuild (option 2 or invalid input)
            BUILD_CMD="docker compose up --build -d"
            ;;
    esac
else
    # No existing images, build from scratch
    BUILD_CMD="docker compose up --build -d"
fi

# Build and start containers using the selected option
echo -e "\e[36mBuilding and starting containers...\e[0m"
eval $BUILD_CMD

# Wait for services to be ready
echo -e "\e[36mWaiting for services to start...\e[0m"
MAX_ATTEMPTS=30  # Reduced from 60 to speed up feedback
ATTEMPTS=0
BACKEND_READY=false
FRONTEND_READY=false

while [ $ATTEMPTS -lt $MAX_ATTEMPTS ] && { [ "$BACKEND_READY" = false ] || [ "$FRONTEND_READY" = false ]; }; do
    ATTEMPTS=$((ATTEMPTS+1))
    
    # Check backend health
    if curl -s http://localhost:5000/api/status > /dev/null 2>&1; then
        BACKEND_READY=true
        echo -e "\e[32mBackend is ready!\e[0m"
    else
        echo -e "\e[33mWaiting for backend...\e[0m"
    fi

    # Check frontend health
    if curl -s http://localhost > /dev/null 2>&1; then
        FRONTEND_READY=true
        echo -e "\e[32mFrontend is ready!\e[0m"
    else
        # Only check logs every 3 attempts to reduce output
        if [ $((ATTEMPTS % 3)) -eq 0 ]; then
            echo -e "\e[33mChecking frontend logs...\e[0m"
            docker logs passport-pal-frontend --tail 5
        fi
    fi

    if [ "$BACKEND_READY" = false ] || [ "$FRONTEND_READY" = false ]; then
        sleep 2
    fi
done

# Check service status
if [ "$BACKEND_READY" = false ]; then
    echo -e "\e[31mError: Backend failed to start properly.\e[0m"
    docker compose logs backend || docker-compose logs backend
    exit 1
fi

if [ "$FRONTEND_READY" = false ]; then
    echo -e "\e[33mWarning: Frontend failed to start properly, but backend is working.\e[0m"
    echo -e "\e[33mYou can still use the API directly at http://localhost:5000\e[0m"
    docker logs passport-pal-frontend --tail 20
    
    # Ask if user wants to force restart frontend
    read -p "Do you want to try restarting the frontend container? (y/n): " RESTART
    if [ "$RESTART" = "y" ]; then
        echo -e "\e[36mRestarting frontend container...\e[0m"
        docker restart passport-pal-frontend
        echo -e "\e[32mFrontend container restarted. Try accessing http://localhost in your browser.\e[0m"
    fi
fi

# Check Docker disk usage after build
echo -e "\e[36m\nDocker disk usage after build:\e[0m"
docker system df

echo -e "\n\e[32mPassportPAL application is now running!\e[0m"
if [ "$FRONTEND_READY" = true ]; then
    echo -e "\e[33mFrontend: http://localhost\e[0m"
fi
echo -e "\e[33mBackend API: http://localhost:5000\e[0m"

echo -e "\n\e[36mUseful Docker commands:\e[0m"
echo -e "\e[36m- View logs:        docker compose logs -f\e[0m"
echo -e "\e[36m- Stop application: docker compose down\e[0m"
echo -e "\e[36m- Cleanup space:    docker system prune -a\e[0m"
echo -e "\e[36m- Remove cache:     docker builder prune -f\e[0m"
