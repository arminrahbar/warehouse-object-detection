# Use the official Python 3.12 slim-based image as the base image
# "slim" is a minimal version of Python, reducing image size and dependencies.
FROM python:3.12-slim  

# Update package lists to ensure access to the latest versions
RUN apt update

# Install FFmpeg for multimedia processing (video/audio handling)
RUN apt install -y ffmpeg  

# Install additional system dependencies:
# - python3: Ensures Python is available (usually included in base image)
# - python3-pip: Pip package manager for installing Python dependencies
# - python3-tk: Required for GUI-related operations (some libraries may need it)
# - libgtk2.0-dev & pkg-config: Needed for OpenCV GUI functions and image processing
RUN apt-get update && apt-get install -y python3 python3-pip python3-tk libgtk2.0-dev pkg-config

# Set the working directory inside the container to "/app"
# All subsequent commands (e.g., COPY, RUN) will be executed relative to this directory.
WORKDIR /app  

# Copy the "requirements.txt" file to the container to manage dependencies
COPY requirements.txt requirements.txt  

# Install Python dependencies from "requirements.txt"
RUN pip3 install --no-cache-dir -r requirements.txt  

# Copy the entire project directory to the container's "/app" folder
# Files and directories listed in ".dockerignore" will be excluded from the copy process.
COPY . /app/  

# Expose port 23000 for external connections (useful for web servers or streaming apps)
EXPOSE 23000  

# Define the default command to run when the container starts
# This runs the Python application "app.py"
CMD ["python3", "app.py"]  
