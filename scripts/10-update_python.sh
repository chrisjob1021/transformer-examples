#!/bin/bash

# Script to install Python 3.13 using deadsnakes PPA

# Update package lists
echo "Updating package lists..."
sudo apt-get update

# Add deadsnakes PPA
echo "Adding deadsnakes PPA..."
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update

# Install Python 3.13
echo "Installing Python 3.13..."
sudo apt-get install -y python3.13 python3.13-venv python3.13-dev

# Install pip for Python 3.13
echo "Installing pip for Python 3.13..."
curl -sS https://bootstrap.pypa.io/get-pip.py | sudo python3.13

# Set Python 3.13 as default
echo "Setting Python 3.13 as default..."
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.13 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1
sudo update-alternatives --set python /usr/bin/python3.13
sudo update-alternatives --set python3 /usr/bin/python3.13

# Verify installation
echo "Verifying installation..."
python --version
python3 --version
pip3.13 --version

echo "Python 3.13 installation from deadsnakes PPA completed and set as default!"
