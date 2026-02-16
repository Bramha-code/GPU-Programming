#!/bin/bash

# Build script for Linux/Mac

echo "==================================="
echo "GPU Programming - Build Script"
echo "==================================="
echo ""

# Check if NVCC is available
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: NVCC not found in PATH"
    echo "Please install CUDA Toolkit"
    echo "Ubuntu/Debian: sudo apt install nvidia-cuda-toolkit"
    echo "Arch: sudo pacman -S cuda"
    exit 1
fi

echo "CUDA Toolkit found:"
nvcc --version
echo ""

# Create output directory
mkdir -p bin

# Compilation flags  
NVCC_FLAGS="-O3 -arch=sm_75"

echo "Building examples..."
echo ""

# Build basic examples
echo "[1/4] Building vector addition..."
nvcc $NVCC_FLAGS 01_basics/add.cu -o bin/add || exit 1

echo "[2/4] Building vector addition benchmark..."
nvcc $NVCC_FLAGS 01_basics/add_bench.cu -o bin/add_bench || exit 1

echo "[3/4] Building matrix-vector multiplication..."
nvcc $NVCC_FLAGS 01_basics/matvec.cu -o bin/matvec || exit 1

echo "[4/4] Building matrix multiplication..."
nvcc $NVCC_FLAGS 01_basics/matmul.cu -o bin/matmul || exit 1

echo ""
echo "==================================="
echo "Build completed successfully!"
echo "==================================="
echo ""
echo "Executables are in the 'bin' directory:"
ls -1 bin/
echo ""
echo "To run an example:"
echo "  ./bin/add"
echo "  ./bin/matmul"
echo "  ./bin/add_bench"
echo ""
