#!/bin/bash

echo "Setting up MNIST Digit Recognition Project..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p models outputs

echo "Setup complete!"
echo ""
echo "To run the project:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run EDA: python main.py --eda"
echo "3. Run CNN: python main.py --cnn"
echo "4. Run all: python main.py --all"
