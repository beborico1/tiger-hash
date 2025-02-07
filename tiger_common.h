// tiger_common.h
#ifndef TIGER_COMMON_H
#define TIGER_COMMON_H

#include <stdint.h>

// Define L64 macro for 64-bit constants across all platforms
#if defined(_MSC_VER)
#define L64(x) x##ui64
#else
#define L64(x) x##ULL
#endif

// Common types and structures for both CPU and GPU implementations
typedef struct
{
    uint64_t state[3];
    uint64_t passed;
    unsigned char buffer[64];
    uint32_t length;
} TIGER_COMMON_CTX;

// Function declarations for CPU implementation
void TIGERInit(TIGER_COMMON_CTX *context);
void TIGERUpdate(TIGER_COMMON_CTX *context, const unsigned char *input, size_t len);
void TIGER192Final(unsigned char digest[24], TIGER_COMMON_CTX *context);

// Function declarations for GPU implementation
#ifdef __CUDACC__
__device__ void TIGERInit_gpu(TIGER_COMMON_CTX *context);
__device__ void TIGERUpdate_gpu(TIGER_COMMON_CTX *context, const unsigned char *input, size_t len);
__device__ void TIGER192Final_gpu(unsigned char digest[24], TIGER_COMMON_CTX *context);
#endif

#endif // TIGER_COMMON_H