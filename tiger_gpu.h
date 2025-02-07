#ifndef TIGER_GPU_H
#define TIGER_GPU_H

#include <stdint.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct
    {
        uint64_t state[3];
        uint64_t passed;
        unsigned char buffer[64];
        uint32_t length;
    } GPU_TIGER_CTX;

    __device__ void TIGERInit_gpu(GPU_TIGER_CTX *context);
    __device__ void TIGERUpdate_gpu(GPU_TIGER_CTX *context, const unsigned char *input, size_t len);
    __device__ void TIGER192Final_gpu(unsigned char digest[24], GPU_TIGER_CTX *context);

    void host_TIGERInit_gpu(GPU_TIGER_CTX *context);
    void host_TIGERUpdate_gpu(GPU_TIGER_CTX *context, const unsigned char *input, size_t len);
    void host_TIGER192Final_gpu(unsigned char digest[24], GPU_TIGER_CTX *context);

    void initialize_gpu_tables(void);
    void checkCudaError(cudaError_t err, const char *msg);

#ifdef __cplusplus
}
#endif

#endif // TIGER_GPU_H