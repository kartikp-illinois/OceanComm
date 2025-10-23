#!/bin/bash

# Build script for FIR Filter UDP application

echo "Building FIR Filter UDP Application..."
echo "========================================"

# Check if CMake is installed
if ! command -v cmake &> /dev/null; then
    echo "Error: CMake is not installed"
    exit 1
fi

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir build
fi

# Navigate to build directory
cd build

# Configure with CMake
echo "Configuring project with CMake..."
cmake .. || { echo "CMake configuration failed"; exit 1; }

# Build the project
echo "Building project..."
make || { echo "Build failed"; exit 1; }

echo ""
echo "Build successful!"
echo "========================================"
echo "Executable location: ./build/fir_filter_udp"
echo ""
echo "To run the application:"
echo "  ./build/fir_filter_udp"
echo ""
echo "To run tests:"
echo "  1. Start the server: ./build/fir_filter_udp"
echo "  2. In another terminal: cd test && python3 test_fir_filter.py"
echo ""