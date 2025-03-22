# Docker Cleanup Script for PassportPAL
# This script helps manage Docker resources and reduce storage usage

Write-Host "PassportPAL Docker Cleanup Utility" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan

# Function to show Docker disk usage
function Show-DockerDiskUsage {
    Write-Host "`nCurrent Docker Disk Usage:" -ForegroundColor Cyan
    docker system df
}

# Check if Docker is running
if (-not (Get-Process -Name "Docker Desktop" -ErrorAction SilentlyContinue)) {
    Write-Host "Docker Desktop is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Display current usage
Show-DockerDiskUsage

# Menu for cleanup options
Write-Host "`nCleanup Options:" -ForegroundColor Green
Write-Host "1. Remove unused containers (safe)" -ForegroundColor White
Write-Host "2. Remove unused images (safe)" -ForegroundColor White
Write-Host "3. Remove build cache (safe, rebuilds will be slower)" -ForegroundColor White
Write-Host "4. Remove all unused Docker objects (containers, images, networks, cache)" -ForegroundColor White
Write-Host "5. Deep clean - prune system and rebuild PassportPAL (rebuilds everything)" -ForegroundColor Yellow
Write-Host "6. Exit without cleaning" -ForegroundColor White

$option = Read-Host "`nSelect an option (1-6)"

switch ($option) {
    "1" {
        Write-Host "`nRemoving unused containers..." -ForegroundColor Cyan
        docker container prune -f
    }
    "2" {
        Write-Host "`nRemoving unused images..." -ForegroundColor Cyan
        docker image prune -f
    }
    "3" {
        Write-Host "`nRemoving build cache..." -ForegroundColor Cyan
        docker builder prune -f
    }
    "4" {
        Write-Host "`nRemoving all unused Docker objects..." -ForegroundColor Cyan
        docker system prune -f
    }
    "5" {
        Write-Host "`nPerforming deep clean (this will remove ALL unused Docker resources)..." -ForegroundColor Yellow
        
        # Stop PassportPAL containers if running
        Write-Host "Stopping PassportPAL containers if running..." -ForegroundColor Cyan
        docker compose down
        
        # Prune everything including volumes (with confirmation)
        $confirm = Read-Host "This will remove ALL unused Docker resources, including volumes. Continue? (y/n)"
        if ($confirm -eq "y") {
            Write-Host "Removing all unused Docker resources..." -ForegroundColor Cyan
            docker system prune -a -f --volumes
            
            # Clean Docker BuildKit cache
            Write-Host "Cleaning Docker BuildKit cache..." -ForegroundColor Cyan
            docker builder prune -a -f
        } else {
            Write-Host "Deep clean cancelled." -ForegroundColor Yellow
        }
    }
    "6" {
        Write-Host "Exiting without cleaning." -ForegroundColor Cyan
        exit 0
    }
    default {
        Write-Host "Invalid option selected. Exiting." -ForegroundColor Red
        exit 1
    }
}

# Show disk usage after cleanup
Write-Host "`nDocker Disk Usage After Cleanup:" -ForegroundColor Green
Show-DockerDiskUsage

# Tips for Docker usage
Write-Host "`nTips for Efficient Docker Usage:" -ForegroundColor Green
Write-Host "- Run this cleanup script regularly to prevent disk space issues" -ForegroundColor White
Write-Host "- Use 'docker build --no-cache' only when necessary as it increases build time" -ForegroundColor White
Write-Host "- Consider using .dockerignore to exclude unnecessary files from builds" -ForegroundColor White
Write-Host "- Run 'docker system df' to check Docker disk usage at any time" -ForegroundColor White

Write-Host "`nCleanup completed!" -ForegroundColor Green 