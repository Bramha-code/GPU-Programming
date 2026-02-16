NVCC = nvcc
NVCC_FLAGS = -O3

# Detect CUDA architecture
# Common values: sm_60 (Pascal), sm_70 (Volta), sm_75 (Turing), sm_80 (Ampere), sm_86 (RTX 30xx)
CUDA_ARCH = -arch=sm_75

.PHONY: all clean basics optimization advanced

all: basics optimization

basics: add add_bench matvec matmul

optimization: matmul_bench

add: 01_basics/add.cu
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $< -o $@

add_bench: 01_basics/add_bench.cu
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $< -o $@

matvec: 01_basics/matvec.cu
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $< -o $@

matmul: 01_basics/matmul.cu
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $< -o $@

matmul_bench: 02_optimization/matmul_bench.cu
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $< -o $@

run_add: add
	./add

run_matmul: matmul
	./matmul

run_bench: add_bench
	./add_bench

clean:
	rm -f add add_bench matvec matmul matmul_bench
	rm -f *.o

help:
	@echo "Available targets:"
	@echo "  all          - Build all examples"
	@echo "  basics       - Build basic examples (add, matvec, matmul)"
	@echo "  optimization - Build optimization examples"
	@echo "  run_add      - Run vector addition"
	@echo "  run_matmul   - Run matrix multiplication"
	@echo "  run_bench    - Run benchmarks"
	@echo "  clean        - Remove all built files"
