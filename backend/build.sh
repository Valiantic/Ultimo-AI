#!/usr/bin/env bash
# Render build script for FastAPI backend

set -o errexit

# Configure Cargo to use a writable directory
export CARGO_HOME=/opt/render/project/.cargo
export CARGO_TARGET_DIR=/opt/render/project/target

# Create directories if they don't exist
mkdir -p $CARGO_HOME
mkdir -p $CARGO_TARGET_DIR

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
