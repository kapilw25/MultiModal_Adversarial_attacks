#!/bin/bash
# Complete setup script for MultiModal VLM Pipeline
# Installs all dependencies needed to run test_vlm.py successfully

set -e  # Exit on any error

echo "=================================================="
echo "MultiModal VLM Pipeline - Complete Environment Setup"
echo "=================================================="

# Check if we're in a virtual environment
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "Currently in virtual environment: $VIRTUAL_ENV"
    echo "This will install all dependencies here."
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting. Please activate your desired environment first."
        exit 1
    fi
else
    echo "No virtual environment detected."
    echo "Creating main environment: venv_base"
    
    # Create base virtual environment
    python3 -m venv venv_base
    source venv_base/bin/activate
    echo "Created and activated venv_base"
fi

# Upgrade pip first
echo "Upgrading pip..."
pip install --upgrade pip

# Install all main requirements needed for test_vlm.py
echo "Installing complete requirements from requirements.txt..."
echo "This may take several minutes..."
pip install -r requirements.txt

echo ""
echo "✅ Complete environment ready!"
echo ""
echo "Next steps:"
echo "1. Activate environment: source venv_base/bin/activate"
echo "2. Run pipeline: python3 local_model/test_vlm.py"
echo ""
echo "The pipeline will automatically:"
echo "  - Create Florence-2 environment (transformers==4.44.2) when needed"
echo "  - Run most VLMs directly in this environment (latest transformers)"
echo "  - Handle all model-specific requirements transparently"
echo ""
echo "All dependencies installed - ready to run any VLM model!"