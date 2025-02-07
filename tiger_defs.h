// tiger_defs.h
#ifndef TIGER_DEFS_H
#define TIGER_DEFS_H

#include <stdint.h>

// Cross-platform L64 macro definition
#if defined(_MSC_VER)
#define L64(x) x##ui64
#else
#define L64(x) x##ULL
#endif

// Common utility macros
#define ZEND_SECURE_ZERO(p, n) memset((p), 0, (n))

// Common types for both CPU and GPU implementations
typedef struct
{
    uint64_t state[3];
    uint64_t passed;
    unsigned char buffer[64];
    uint32_t length;
} TIGER_COMMON_CTX;

#endif // TIGER_DEFS_H

// Makefile
NVCC = nvcc
    CFLAGS = -I.- O3
                      CUDA_ARCH = -arch = sm_35 #Adjust based on your GPU architecture

                 SOURCES = tiger_bruteforce_gpu.cu tiger_gpu.cu tiger.c tiger_impl.c tiger_tables.c
                               HEADERS = tiger.h tiger_gpu.h tiger_common.h tiger_defs.h tiger_tables.h
                                             TARGET = tiger_bruteforce_gpu

                                                      $(TARGET) : $(SOURCES) $(HEADERS)
                                                                      $(NVCC) $(CFLAGS) $(CUDA_ARCH) -
                                                                  o $ @$(SOURCES)

                                                                      clean : rm -
                                                                              f $(TARGET)

#Modified build command:
#make CUDA_ARCH = "-arch=sm_XX"(replace XX with your GPU architecture version)
#Common values : sm_35, sm_50, sm_60, sm_70, sm_75, sm_80, sm_86