#!/bin/bash

# Exit on error
set -e

echo "Creating Python virtual environment..."

# Check if Python 3.12 or greater is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
REQUIRED_VERSION="3.12"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "Error: Python version $REQUIRED_VERSION or greater is required. Found version $PYTHON_VERSION."
    echo "Please install Python $REQUIRED_VERSION or greater."
    exit 1
fi

echo "Found Python $PYTHON_VERSION, which meets the requirement (>= $REQUIRED_VERSION)."

# Define the virtual environment directory
VENV_DIR="venv"

# Check if virtual environment already exists
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    echo "To recreate it, delete the directory first: rm -rf $VENV_DIR"
    exit 0
fi

# Create the virtual environment
python3 -m venv "$VENV_DIR"

echo "Virtual environment created at $VENV_DIR"
echo "To activate it, run: source $VENV_DIR/bin/activate"

# Upgrade pip in the virtual environment
echo "Upgrading pip..."
"$VENV_DIR/bin/pip" install --upgrade pip

# Note: The virtual environment won't be activated in the current shell
# when running this script from the terminal because scripts run in a subshell
echo "Note: To activate the virtual environment in your current shell, you need to run:"
echo "source \"$VENV_DIR/bin/activate\""

# We can activate it for the remainder of this script, but it won't affect the parent shell
if [ -f "$VENV_DIR/bin/activate" ]; then
    # This only affects the current script execution
    source "$VENV_DIR/bin/activate" 2>/dev/null || true
fi
# Verify activation
echo "Virtual environment activated. Python path: $(which python)"

# Check if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Installing packages from requirements.txt..."
    "$VENV_DIR/bin/pip" install -r requirements.txt
    echo "Packages installed successfully."
else
    echo "No requirements.txt file found. Skipping package installation."
fi

echo "Done!"
