# Compiler settings
NVCC = nvcc
CCFLAGS = -O3 -Wall -I.
NVCCFLAGS = -O3 -I.

# Default CUDA architecture for RTX 4070 Ti
CUDA_ARCH ?= -arch=sm_89

# Source files
CUDA_SOURCE = tiger_cuda.cu
C_SOURCES = tiger.c tiger_impl.c tiger_tables.c

# Object files
CUDA_OBJECT = $(CUDA_SOURCE:.cu=.o)
C_OBJECTS = $(C_SOURCES:.c=.o)

# Target executable
TARGET = tiger_bruteforce_gpu

all: $(TARGET)

$(TARGET): $(CUDA_OBJECT) $(C_OBJECTS)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -o $@ $^

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean