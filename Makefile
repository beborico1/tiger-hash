# Compiler settings
NVCC = nvcc
CC = gcc
CCFLAGS = -O3 -Wall -I.
NVCCFLAGS = -O3 -I.

# Default CUDA architecture (can be overridden from command line)
CUDA_ARCH ?= -arch=sm_60

# Source files
CUDA_SOURCES = tiger_bruteforce_gpu.cu tiger_gpu.cu
C_SOURCES = tiger.c tiger_impl.c tiger_tables.c

# Object files
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)
C_OBJECTS = $(C_SOURCES:.c=.o)

# Target executables
TARGETS = tiger_bruteforce_gpu tiger_hash_compare

# Default target
all: $(TARGETS)

# Main GPU bruteforce executable
tiger_bruteforce_gpu: $(CUDA_OBJECTS) $(C_OBJECTS)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -o $@ $^

# Hash comparison executable
tiger_hash_compare: tiger_hash_compare.cu $(C_OBJECTS)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -o $@ $^

# Pattern rule for CUDA source files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@

# Pattern rule for C source files
%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

# Clean target
clean:
	rm -f $(TARGETS) *.o

# Testing targets
test_cpu: tiger_test
	./tiger_test

test_gpu: tiger_bruteforce_gpu
	./tiger_bruteforce_gpu

# Help target
help:
	@echo "Available targets:"
	@echo "  all               - Build all executables"
	@echo "  tiger_bruteforce_gpu - Build GPU bruteforce implementation"
	@echo "  tiger_hash_compare   - Build hash comparison tool"
	@echo "  clean             - Remove built files"
	@echo "  test_cpu          - Run CPU tests"
	@echo "  test_gpu          - Run GPU tests"
	@echo ""
	@echo "Options:"
	@echo "  CUDA_ARCH        - Set CUDA architecture (default: sm_60)"
	@echo "  Example: make CUDA_ARCH='-arch=sm_75'"

# Phony targets
.PHONY: all clean help test_cpu test_gpu

# Dependencies
tiger_bruteforce_gpu.o: tiger_gpu.h tiger.h tiger_common.h
tiger_gpu.o: tiger_gpu.h tiger_common.h tiger_tables.h
tiger.o: tiger.h
tiger_impl.o: tiger.h tiger_tables.h
tiger_tables.o: tiger_tables.h