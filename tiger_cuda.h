#ifndef TIGER_CUDA_H
#define TIGER_CUDA_H

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // GPU Tiger context structure
    typedef struct
    {
        uint64_t state[3];
        uint64_t passed;
        unsigned char buffer[64];
        uint32_t length;
    } GPU_TIGER_CTX;

    // Device function declarations
    __device__ void TIGERInit_gpu(GPU_TIGER_CTX *context);
    __device__ void TIGERUpdate_gpu(GPU_TIGER_CTX *context, const unsigned char *input, size_t len);
    __device__ void TIGER192Final_gpu(unsigned char digest[24], GPU_TIGER_CTX *context);

    // Host wrapper function declarations
    __host__ void initialize_gpu_tables(void);
    __host__ void host_TIGERInit_gpu(GPU_TIGER_CTX *context);
    __host__ void host_TIGERUpdate_gpu(GPU_TIGER_CTX *context, const unsigned char *input, size_t len);
    __host__ void host_TIGER192Final_gpu(unsigned char digest[24], GPU_TIGER_CTX *context);

    // Utility functions
    __device__ unsigned long long atomicAdd64(unsigned long long *address, unsigned long long val);
    void checkCudaError(cudaError_t err, const char *msg);

#ifdef __cplusplus
}
#endif

#endif // TIGER_CUDA_H