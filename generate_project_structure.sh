#!/bin/bash

# Script to create the directory structure for the biocoder-edge project

# --- Set the project name ---
PROJECT_NAME="biocoder-edge"

# --- Create the root project directory ---
# echo "Creating root directory: $PROJECT_NAME/"
# mkdir -p "$PROJECT_NAME"
# cd "$PROJECT_NAME" || exit

# --- Create top-level files ---
echo "Creating top-level files..."
# touch .gitignore
# touch README.md
touch requirements.txt
# touch LICENSE
touch main.py

# --- Create config directory and file ---
echo "Creating config directory..."
mkdir -p config
touch config/config.yaml

# --- Create src directory and module subdirectories ---
echo "Creating src directory and modules..."
mkdir -p src/motion_detector
mkdir -p src/animal_analyzer
mkdir -p src/data_uploader

# Create __init__.py files to make them Python packages
touch src/__init__.py
touch src/motion_detector/__init__.py
touch src/animal_analyzer/__init__.py
touch src/data_uploader/__init__.py

# Create the main module files
touch src/motion_detector/detector.py
touch src/animal_analyzer/analyzer.py
touch src/data_uploader/uploader.py

# --- Create scripts directory for helper tools ---
echo "Creating scripts directory..."
mkdir -p scripts
touch scripts/test_camera.py
touch scripts/test_yolo.py
touch scripts/test_db_connection.py
touch scripts/view_output.py

# --- Create data directory with subfolders ---
echo "Creating data directory..."
mkdir -p data/pending_upload
mkdir -p data/uploaded

# Use .gitkeep to ensure Git tracks the empty directories
touch data/pending_upload/.gitkeep
touch data/uploaded/.gitkeep

# --- Final confirmation message ---
echo ""
echo "Project structure for '$PROJECT_NAME' created successfully."
echo "You can now initialize a git repository here with 'git init'."
echo ""
tree .
