#!/bin/bash
# PassportPAL Docker Clean-up Script for Linux/Mac
# This script safely removes PassportPAL Docker containers, images, and volumes

# Set up color for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Print banner
echo -e "${CYAN}=================================${NC}"
echo -e "${CYAN}  PassportPAL Docker Clean-up   ${NC}"
echo -e "${CYAN}=================================${NC}"
echo ""

# Stop and remove containers
echo -e "${YELLOW}Stopping and removing PassportPAL containers...${NC}"
CONTAINERS=$(docker ps -a --filter "name=passport-pal-" --format "{{.Names}}")
if [ ! -z "$CONTAINERS" ]; then
    docker stop $CONTAINERS
    docker rm $CONTAINERS
    echo -e "${GREEN}PassportPAL containers removed.${NC}"
else
    echo -e "${GREEN}No PassportPAL containers found.${NC}"
fi

# Remove images
echo -e "\n${YELLOW}Removing PassportPAL Docker images...${NC}"
IMAGES=$(docker images "passport-pal-*" --format "{{.Repository}}:{{.Tag}}")
if [ ! -z "$IMAGES" ]; then
    docker rmi $IMAGES -f
    echo -e "${GREEN}PassportPAL images removed.${NC}"
else
    echo -e "${GREEN}No PassportPAL images found.${NC}"
fi

# Remove dangling images
echo -e "\n${YELLOW}Cleaning up dangling images...${NC}"
DANGLING_IMAGES=$(docker images -f "dangling=true" -q)
if [ ! -z "$DANGLING_IMAGES" ]; then
    docker rmi $DANGLING_IMAGES -f
    echo -e "${GREEN}Dangling images removed.${NC}"
else
    echo -e "${GREEN}No dangling images found.${NC}"
fi

# Prune Docker system (optional)
echo -e "\n${YELLOW}Would you like to run a full Docker system prune?${NC}"
echo -e "${YELLOW}This will remove all unused containers, networks, images, and volumes.${NC}"
read -p "Enter 'y' to confirm or any other key to skip: " CONFIRMATION

if [ "$CONFIRMATION" = "y" ]; then
    echo -e "\n${YELLOW}Running Docker system prune...${NC}"
    docker system prune --volumes -f
    echo -e "${GREEN}Docker system pruned.${NC}"
else
    echo -e "\n${YELLOW}Skipping Docker system prune.${NC}"
fi

echo -e "\n${GREEN}Docker clean-up completed successfully!${NC}" 