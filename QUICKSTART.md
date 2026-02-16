# GPU Programming - Quick Start Guide

## Prerequisites

Before running these examples, ensure you have:
1. **NVIDIA GPU** with CUDA support
2. **CUDA Toolkit** installed (version 11.0+)
3. **C++ Compiler** (g++ or MSVC)

## Windows Setup

### Install CUDA Toolkit

1. Download CUDA Toolkit from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads)
2. Run the installer
3. Add CUDA to your PATH:
   ```
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
   ```

### Verify Installation

```cmd
nvcc --version
nvidia-smi
```

## Linux Setup

```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-get update
sudo apt-get -y install cuda

# Arch Linux
sudo pacman -S cuda

# Verify
nvcc --version
nvidia-smi
```

## Compilation

### Manual Compilation

```bash
# Compile vector addition
nvcc add.cu -o add

# With optimization
nvcc -O3 -arch=sm_75 add.cu -o add

# Run
./add
```

### Using Makefile

```bash
# Build all basic examples
make basics

# Build specific target
make add

# Run example
make run_add

# Clean
make clean
```

## Running Examples

### 1. Vector Addition
```bash
nvcc 01_basics/add.cu -o add
./add
```

**Expected Output:**
```
0 0 0
1 2 3
2 4 6
...
```

### 2. Matrix Multiplication
```bash
nvcc 01_basics/matmul.cu -o matmul
./matmul
```

### 3. Benchmarks
```bash
nvcc 01_basics/add_bench.cu -o add_bench
./add_bench
```

**Expected Output:**
```
p = 0 cpu time: 156 gpu time: 2340
p = 1 cpu time: 184 gpu time: 2456
...
p = 20 cpu time: 15680000 gpu time: 45600
```

## Common Issues

### Issue: `nvcc: command not found`
**Solution**: Add CUDA to your PATH

### Issue: `undefined reference to cudaMalloc`
**Solution**: Use `nvcc` instead of `g++`

### Issue: No CUDA-capable device
**Solution**: Ensure you have an NVIDIA GPU and drivers installed

```bash
nvidia-smi  # Check GPU is detected
```

### Issue: Architecture mismatch
**Solution**: Update CUDA architecture in Makefile:
```makefile
# For RTX 30 series
CUDA_ARCH = -arch=sm_86

# For GTX 10 series
CUDA_ARCH = -arch=sm_61
```

## Architecture Codes

| GPU Series | Architecture Code |
|------------|-------------------|
| RTX 40xx   | sm_89             |
| RTX 30xx   | sm_86             |
| RTX 20xx   | sm_75             |
| GTX 16xx   | sm_75             |
| GTX 10xx   | sm_61             |
| GTX 9xx    | sm_52             |

Find your GPU's compute capability: [CUDA GPUs](https://developer.nvidia.com/cuda-gpus)

## Next Steps

1. ✅ Start with `add.cu` to understand basics
2. ✅ Explore `matmul.cu` for 2D grids
3. ✅ Run benchmarks to see GPU vs CPU performance
4. ✅ Study optimization examples
5. ✅ Build the MNIST neural network

## Resources

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [YouTube Course](https://youtube.com/playlist?list=PL3xCBlatwrsXCGW4SfEoLzKiMSUCE7S_X)
- [CUDA Samples](https://github.com/NVIDIA/cuda-samples)

Happy Coding! 🚀
