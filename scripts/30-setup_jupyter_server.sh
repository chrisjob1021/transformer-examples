#!/bin/bash

# Exit on error
set -e

echo "Setting up Jupyter server..."

# Define the virtual environment directory
VENV_DIR="venv"
PROJ_DIR="$(pwd)"

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment not found at $VENV_DIR"
    echo "Please run scripts/create_venv.sh first to create the virtual environment."
    exit 1
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"
echo "Virtual environment activated."

# Install Jupyter
echo "Installing Jupyter..."
pip install jupyter

# Create Jupyter config if it doesn't exist
jupyter notebook --generate-config

# Set up Jupyter to run as a server
echo "Configuring Jupyter to run as a server..."
JUPYTER_CONFIG="$HOME/.jupyter/jupyter_notebook_config.py"

# Backup the config file if it exists
if [ -f "$JUPYTER_CONFIG" ]; then
    cp "$JUPYTER_CONFIG" "${JUPYTER_CONFIG}.backup"
    echo "Backed up existing Jupyter config to ${JUPYTER_CONFIG}.backup"
fi

# Append server settings to the config file
cat >> "$JUPYTER_CONFIG" << EOL

# Configuration for running Jupyter as a server
c.NotebookApp.ip = '0.0.0.0'  # Listen on all IPs
c.NotebookApp.open_browser = False  # Don't open a browser window
c.NotebookApp.port = 8888  # Port to use
c.NotebookApp.token = ''  # Disable token authentication
c.NotebookApp.password = ''  # Disable password authentication
EOL

# Create systemd service file
echo "Creating systemd service file..."
SERVICE_FILE="jupyter.service"
cat > $SERVICE_FILE << EOL
[Unit]
Description=Jupyter Notebook Server
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$PROJ_DIR
ExecStart=$PROJ_DIR/$VENV_DIR/bin/jupyter notebook --config=$JUPYTER_CONFIG
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOL

# Install the service file
echo "Installing systemd service..."
sudo mv $SERVICE_FILE /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable jupyter.service

echo "Jupyter server setup complete!"
echo "The Jupyter server has been configured to run as a systemd service."
echo 
echo "To check status: sudo systemctl status jupyter"
echo "To stop the server: sudo systemctl stop jupyter"
echo "Access the server at http://localhost:8888 or http://<your-ip>:8888"
echo "WARNING: Server is configured to run without password protection. This is not recommended for production environments."