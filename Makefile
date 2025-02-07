# Compiler settings
NVCC = nvcc
CC = gcc
CCFLAGS = -O3 -Wall -I.
NVCCFLAGS = -O3 -I.

# Default CUDA architecture for RTX 4070 Ti
CUDA_ARCH ?= -arch=sm_89

# Source files
CUDA_SOURCE = tiger_bruteforce_gpu.cu
C_SOURCE = tiger.c

# Object files
CUDA_OBJECT = $(CUDA_SOURCE:.cu=.o)
C_OBJECT = $(C_SOURCE:.c=.o)

# Target executable
TARGET = tiger_bruteforce_gpu

all: $(TARGET)

# Compile CUDA source
$(CUDA_OBJECT): $(CUDA_SOURCE)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@

# Compile C source
$(C_OBJECT): $(C_SOURCE)
	$(CC) $(CCFLAGS) -c $< -o $@

$(TARGET): $(CUDA_OBJECT) $(C_OBJECT)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -o $@ $^

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean