#!/bin/bash

# Start the PassportPAL Web Application using Docker Compose
echo -e "\e[32mStarting PassportPAL Web Application...\e[0m"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "\e[31mDocker is not running. Please start Docker first.\e[0m"
    exit 1
fi

# Check if images already exist
needs_build=false
if ! docker images -q passport-pal-backend > /dev/null || ! docker images -q passport-pal-frontend > /dev/null; then
    echo -e "\e[33mDocker images not found. Will build them...\e[0m"
    needs_build=true
fi

# Check if required ports are available
port80InUse=$(netstat -tuln | grep ":80 " | wc -l)
port5000InUse=$(netstat -tuln | grep ":5000 " | wc -l)

if [ $port80InUse -gt 0 ]; then
    echo -e "\e[33mWarning: Port 80 is already in use. The frontend service might fail to start.\e[0m"
fi

if [ $port5000InUse -gt 0 ]; then
    echo -e "\e[33mWarning: Port 5000 is already in use. The backend service might fail to start.\e[0m"
fi

# Verify model files exist
model_files=(
    "./backend/models/custom_instance_segmentation.pt"
    "./backend/models/custom_cnn_model_scripted.pt"
    "./backend/models/custom_cnn_model_metadata.json"
)

missing_models=false
for file in "${model_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "\e[33mWarning: Model file not found: $file\e[0m"
        missing_models=true
    fi
done

if [ "$missing_models" = true ]; then
    echo -e "\e[33mSome model files are missing. The application might not work correctly.\e[0m"
    read -p "Do you want to continue anyway? (y/n) " continue
    if [ "$continue" != "y" ]; then
        exit 1
    fi
fi

# Make script executable if it's not
chmod +x "$0"

# Start containers
if [ "$needs_build" = true ]; then
    echo -e "\e[36mBuilding and starting containers...\e[0m"
    docker-compose up --build -d
else
    echo -e "\e[36mStarting containers from existing images...\e[0m"
    docker-compose up -d
fi

# Check if containers started successfully
backend_running=$(docker ps --filter "name=passport-pal-backend" --format "{{.Names}}" | grep "passport-pal-backend")
frontend_running=$(docker ps --filter "name=passport-pal-frontend" --format "{{.Names}}" | grep "passport-pal-frontend")

if [ -z "$backend_running" ]; then
    echo -e "\e[31mError: Backend container failed to start. Check docker logs for details:\e[0m"
    docker logs passport-pal-backend
    exit 1
fi

if [ -z "$frontend_running" ]; then
    echo -e "\e[31mError: Frontend container failed to start. Check docker logs for details:\e[0m"
    docker logs passport-pal-frontend
    exit 1
fi

echo -e "\n\e[32mPassportPAL application is now running!\e[0m"
echo -e "\e[33mFrontend: http://localhost\e[0m"
echo -e "\e[33mBackend API: http://localhost:5000\e[0m"
echo -e "\n\e[36mUseful Docker commands:\e[0m"
echo -e "\e[36m- Stop application:  docker-compose down\e[0m"
echo -e "\e[36m- View logs:        docker-compose logs\e[0m"
echo -e "\e[36m- Rebuild images:   docker-compose up --build -d\e[0m"
