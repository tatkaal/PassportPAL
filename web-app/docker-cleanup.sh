#!/bin/bash
# Docker Cleanup Script for PassportPAL
# This script helps manage Docker resources and reduce storage usage

echo -e "\e[36mPassportPAL Docker Cleanup Utility\e[0m"
echo -e "\e[36m=================================\e[0m"

# Function to show Docker disk usage
show_docker_disk_usage() {
    echo -e "\n\e[36mCurrent Docker Disk Usage:\e[0m"
    docker system df
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "\e[31mDocker is not running. Please start Docker first.\e[0m"
    exit 1
fi

# Display current usage
show_docker_disk_usage

# Menu for cleanup options
echo -e "\n\e[32mCleanup Options:\e[0m"
echo -e "1. Remove unused containers (safe)"
echo -e "2. Remove unused images (safe)"
echo -e "3. Remove build cache (safe, rebuilds will be slower)"
echo -e "4. Remove all unused Docker objects (containers, images, networks, cache)"
echo -e "5. Deep clean - prune system and rebuild PassportPAL (rebuilds everything)"
echo -e "6. Exit without cleaning"

read -p $'\nSelect an option (1-6): ' OPTION

case $OPTION in
    1)
        echo -e "\n\e[36mRemoving unused containers...\e[0m"
        docker container prune -f
        ;;
    2)
        echo -e "\n\e[36mRemoving unused images...\e[0m"
        docker image prune -f
        ;;
    3)
        echo -e "\n\e[36mRemoving build cache...\e[0m"
        docker builder prune -f
        ;;
    4)
        echo -e "\n\e[36mRemoving all unused Docker objects...\e[0m"
        docker system prune -f
        ;;
    5)
        echo -e "\n\e[33mPerforming deep clean (this will remove ALL unused Docker resources)...\e[0m"
        
        # Stop PassportPAL containers if running
        echo -e "\e[36mStopping PassportPAL containers if running...\e[0m"
        docker compose down || docker-compose down
        
        # Prune everything including volumes (with confirmation)
        read -p "This will remove ALL unused Docker resources, including volumes. Continue? (y/n): " CONFIRM
        if [ "$CONFIRM" = "y" ]; then
            echo -e "\e[36mRemoving all unused Docker resources...\e[0m"
            docker system prune -a -f --volumes
            
            # Clean Docker BuildKit cache
            echo -e "\e[36mCleaning Docker BuildKit cache...\e[0m"
            docker builder prune -a -f
        else
            echo -e "\e[33mDeep clean cancelled.\e[0m"
        fi
        ;;
    6)
        echo -e "\e[36mExiting without cleaning.\e[0m"
        exit 0
        ;;
    *)
        echo -e "\e[31mInvalid option selected. Exiting.\e[0m"
        exit 1
        ;;
esac

# Show disk usage after cleanup
echo -e "\n\e[32mDocker Disk Usage After Cleanup:\e[0m"
show_docker_disk_usage

# Tips for Docker usage
echo -e "\n\e[32mTips for Efficient Docker Usage:\e[0m"
echo -e "- Run this cleanup script regularly to prevent disk space issues"
echo -e "- Use 'docker build --no-cache' only when necessary as it increases build time"
echo -e "- Consider using .dockerignore to exclude unnecessary files from builds"
echo -e "- Run 'docker system df' to check Docker disk usage at any time"

echo -e "\n\e[32mCleanup completed!\e[0m" 