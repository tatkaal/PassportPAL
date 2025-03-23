# Manual Installation

1. **Install the required Python packages**

  ```bash
   pip install requests gdown
  ```

2. **Download ML Models:**
   - Download the following files manually from https://drive.google.com/drive/folders/1qG6xU7eGEwTXxQWP5L6s2zuJ7FXs3SQB?usp=sharing:
     - `custom_instance_segmentation.pt`
     - `custom_cnn_model_scripted.pt`
     - `custom_cnn_model_metadata.json`
   - Place the downloaded files in the `backend/models` directory.

3. **Ensure Docker is Running:**

   - On **Windows/Mac**, launch Docker Desktop.
   - On **Linux**, ensure the Docker daemon is running (you might need to run Docker commands with `sudo` or add your user to the Docker group).

4. **Check Port Availability:**

   Ensure that ports **80** and **5000** are not in use by other services

5. **Build and Start the Application with Docker Compose:**

   - Ensure you are in the project root (where `docker-compose.yml` is located).
   - Stop any existing containers:
     ```powershell
     docker compose down
     ```
     or
     ```bash
     docker-compose down
     ```
   - Build the containers:
     ```powershell
     docker compose build
     ```
     or
     ```bash
     docker-compose build
     ```
   - Start the containers in detached mode:
     ```powershell
     docker compose up -d
     ```
     or
     ```bash
     docker-compose up -d
     ```

6. **Verify Application Health:**

   - Check the backend status by opening your browser and navigating to:
     ```
     http://localhost:5000/api/status
     ```
   - Now check the frontend by visiting:
     ```
     http://localhost
     ```