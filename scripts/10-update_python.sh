#!/bin/bash

# Script to install Python 3.12 using deadsnakes PPA

# Update package lists
echo "Updating package lists..."
sudo apt-get update

# Add deadsnakes PPA
echo "Adding deadsnakes PPA..."
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update

# Install Python 3.12
echo "Installing Python 3.12..."
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev

# Install pip for Python 3.12
echo "Installing pip for Python 3.12..."
curl -sS https://bootstrap.pypa.io/get-pip.py | sudo python3.12

# Set Python 3.12 as default
echo "Setting Python 3.12 as default..."
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
sudo update-alternatives --set python /usr/bin/python3.12
sudo update-alternatives --set python3 /usr/bin/python3.12

# Verify installation
echo "Verifying installation..."
python --version
python3 --version
pip3.12 --version

echo "Python 3.12 installation from deadsnakes PPA completed and set as default!"
