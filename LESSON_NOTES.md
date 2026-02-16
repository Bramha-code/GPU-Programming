# Lesson Notes - GPU Programming Course

## Lesson 1-5: Introduction & Fundamentals

### Hitting the Power Wall
- **Problem**: CPU clock speeds plateaued around 2004
- **Reason**: Power consumption grows cubically with frequency
- **Solution**: Parallel computing with GPUs

### Why GPUs?
- Thousands of cores vs dozens in CPUs  
- Optimized for throughput over latency
- Perfect for data-parallel workloads

### CUDA Programming Model
- **Host**: CPU and its memory
- **Device**: GPU and its memory
- **Kernel**: Function that runs on GPU

```cpp
__global__ void kernel(args) { }  // Runs on GPU
kernel<<<gridSize, blockSize>>>(args);  // Launch from CPU
```

## Lesson 6-10: Memory Management

### Memory Hierarchy
1. **Global Memory**: Slow, large (GB)
2. **Shared Memory**: Fast, small (KB per block)
3. **Registers**: Fastest, tiny (per thread)
4. **Constant Memory**: Read-only, cached

### Memory Operations
```cpp
// Allocate device memory
float* d_array;
cudaMalloc(&d_array, size * sizeof(float));

// Copy host → device
cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice);

// Copy device → host
cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost);

// Free device memory
cudaFree(d_array);
```

## Lesson 11-15: Thread Hierarchy

### Thread Organization
- **Thread**: Basic unit of execution
- **Warp**: 32 threads executed together
- **Block**: Group of threads (up to 1024)
- **Grid**: Collection of blocks

### Thread Indexing
```cpp
// 1D indexing
int idx = blockIdx.x * blockDim.x + threadIdx.x;

// 2D indexing
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

### Typical Block Sizes
- 1D: 256, 512, 1024
- 2D: 16x16, 32x32

## Lesson 16-20: Memory Coalescing

### Coalesced Access
**Good**: Adjacent threads access adjacent memory
```cpp
// Thread 0 → array[0]
// Thread 1 → array[1]  
// Thread 2 → array[2]
```

**Bad**: Strided or random access
```cpp
// Thread 0 → array[0]
// Thread 1 → array[64]  // Stride = 64
// Thread 2 → array[128]
```

### Performance Impact
- Coalesced: 1 memory transaction
- Stride-2: 2 memory transactions
- Stride-32: 32 memory transactions!

## Lesson 21-25: Shared Memory

### Using Shared Memory
```cpp
__global__ void kernel() {
    __shared__ float tile[TILE_SIZE];
    
    // Load into shared memory
    tile[threadIdx.x] = global_array[idx];
    __syncthreads();  // Wait for all threads
    
    // Use tile (fast!)
    float result = tile[threadIdx.x] * 2;
}
```

### Bank Conflicts
- Shared memory has 32 banks
- Avoid multiple threads accessing same bank

## Lesson 26-30: Matrix Multiplication Optimization

### Naive Implementation
```cpp
for (int i = 0; i < N; i++) {
    sum += A[row*N + i] * B[i*N + col];
}
```
**Problem**: Non-coalesced reads from B

### Tiled Implementation
1. Load tiles into shared memory
2. Compute on tiles
3. Accumulate results

**Speedup**: 10-50x over naive version!

## Lesson 31-35: Advanced Optimizations

### Vectorized Loads
```cpp
float4 data = *reinterpret_cast<float4*>(&array[idx]);
```
- Load 128 bits at once
- 2-4x faster than scalar loads

### Constant Memory
```cpp
__constant__ float const_array[SIZE];
```
- Cached, broadcast to all threads
- Best for read-only data accessed by all threads

### Warp Divergence
**Avoid**:
```cpp
if (threadIdx.x % 2 == 0) {  // Half threads idle!
    // do work
}
```

**Better**:
```cpp
if (blockIdx.x % 2 == 0) {  // Entire blocks active
    // do work
}
```

## Lesson 36-40: Neural Networks

### Forward Pass
```cpp
// Matrix multiplication: output = input * weights + bias
__global__ void forward(float* input, float* weights, float* bias, float* output) {
    // Compute output[row][col]
    float sum = bias[col];
    for (int i = 0; i < N; i++) {
        sum += input[row*N + i] * weights[i*M + col];
    }
    output[row*M + col] = sum;
}
```

### Backward Pass
- Compute gradients with respect to inputs
- Update weights using gradients

### Activation Functions
```cpp
// ReLU
output = max(0.0f, input);

// ReLU gradient
gradient = input > 0 ? 1.0f : 0.0f;
```

## Lesson 41-45: MNIST Classification

### Network Architecture
```
Input (784) → Dense (128) → ReLU → Dense (10) → Softmax
```

### Softmax
```cpp
float max_val = findMax(logits);
float sum = 0;
for (int i = 0; i < N; i++) {
    output[i] = exp(logits[i] - max_val);
    sum += output[i];
}
for (int i = 0; i < N; i++) {
    output[i] /= sum;
}
```

### Cross-Entropy Loss
```cpp
loss = -sum(target * log(prediction))
```

## Lesson 46-49: Optimization Techniques

### Kernel Fusion
Combine multiple kernels to reduce memory transfers:
```cpp
// Instead of: kernel1(); kernel2();
// Use: fused_kernel();
```

### Asynchronous Execution
```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);
kernel<<<grid, block, 0, stream>>>();
```

### Profiling
```bash
# Use NVIDIA Nsight
nvprof ./program

# Or NVIDIA Nsight Compute
ncu ./program
```

## Key Takeaways

1. **Memory is king**: Optimize memory access first
2. **Coalesce**: Keep adjacent threads accessing adjacent memory
3. **Shared memory**: Use for frequently accessed data
4. **Occupancy**: Keep GPU busy with enough threads
5. **Profile**: Measure, don't guess!

## Performance Checklist

- [ ] Memory accesses coalesced?
- [ ] Using shared memory where appropriate?
- [ ] Minimizing host-device transfers?
- [ ] Avoiding warp divergence?
- [ ] Enough threads for high occupancy?
- [ ] Profiled with nvprof/ncu?

## Common Patterns

### Reduction (Sum)
```cpp
__shared__ float partial_sum[BLOCK_SIZE];
// Each thread loads and sums
// Tree-based reduction in shared memory
```

### Scan (Prefix Sum)
```cpp
// Parallel prefix sum using shared memory
// Hillis-Steele or Blelloch algorithm
```

### Histogram
```cpp
// Atomic operations or privatization technique
atomicAdd(&histogram[bin], 1);
```

---

**Resources for Deep Dive:**
- CUDA C Programming Guide
- CUDA Best Practices Guide  
- "Programming Massively Parallel Processors" by Kirk & Hwu
