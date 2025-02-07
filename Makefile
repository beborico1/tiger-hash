# Compiler settings
NVCC = nvcc
CC = gcc
CCFLAGS = -O3 -Wall -I.
NVCCFLAGS = -O3 -I.

# Default CUDA architecture for RTX 4070 Ti
CUDA_ARCH ?= -arch=sm_89

# Source files
CUDA_SOURCES = tiger_gpu.cu tiger_bruteforce_gpu.cu
C_SOURCES = tiger.c

# Object files
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)
C_OBJECTS = $(C_SOURCES:.c=.o)

# Target executable
TARGET = tiger_bruteforce_gpu

all: $(TARGET)

# Compile CUDA sources
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@

# Compile C sources
%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

$(TARGET): $(CUDA_OBJECTS) $(C_OBJECTS)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -o $@ $^

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean