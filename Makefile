# Compiler settings
NVCC = nvcc
CC = gcc
CCFLAGS = -O3 -Wall -I.
NVCCFLAGS = -O3 -I.

# Default CUDA architecture for RTX 4070 Ti
CUDA_ARCH ?= -arch=sm_89

# Source files
CUDA_SOURCES = tiger_bruteforce_gpu.cu tiger_gpu.cu tiger_cuda_device.cu
C_SOURCES = tiger.c tiger_impl.c tiger_tables.c

# Object files
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)
C_OBJECTS = $(C_SOURCES:.c=.o)

# Target executables
TARGETS = tiger_bruteforce_gpu

# Default target
all: $(TARGETS)

# Main executable
tiger_bruteforce_gpu: $(CUDA_OBJECTS) $(C_OBJECTS)
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

.PHONY: all clean