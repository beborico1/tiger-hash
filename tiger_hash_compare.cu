#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "tiger.h"

// GPU Tiger context structure - should match the structure in tiger_cuda.c
typedef struct
{
    uint64_t state[3];
    uint64_t passed;
    unsigned char buffer[64];
    uint32_t length;
} GPU_TIGER_CTX;

// Function declarations for CUDA implementation
__device__ void TIGERInit_gpu(GPU_TIGER_CTX *context);
__device__ void TIGERUpdate_gpu(GPU_TIGER_CTX *context, const unsigned char *input, size_t len);
__device__ void TIGER192Final_gpu(unsigned char digest[24], GPU_TIGER_CTX *context);

void print_hash(unsigned char *hash)
{
    for (int i = 0; i < 24; i++)
    {
        printf("%02x", hash[i]);
    }
    printf("\n");
}

// Kernel definitions
__global__ void tiger_init_kernel(GPU_TIGER_CTX *context)
{
    TIGERInit_gpu(context);
}

__global__ void tiger_update_kernel(GPU_TIGER_CTX *context, const unsigned char *input, size_t len)
{
    TIGERUpdate_gpu(context, input, len);
}

__global__ void tiger_final_kernel(GPU_TIGER_CTX *context, unsigned char *digest)
{
    TIGER192Final_gpu(digest, context);
}

// Host wrapper functions
void host_TIGERInit_gpu(GPU_TIGER_CTX *context)
{
    GPU_TIGER_CTX *d_context;
    cudaError_t err;

    // Allocate device memory for context
    err = cudaMalloc(&d_context, sizeof(GPU_TIGER_CTX));
    checkCudaError(err, "Failed to allocate device memory for context");

    // Copy context to device
    err = cudaMemcpy(d_context, context, sizeof(GPU_TIGER_CTX), cudaMemcpyHostToDevice);
    checkCudaError(err, "Failed to copy context to device");

    // Launch kernel
    tiger_init_kernel<<<1, 1>>>(d_context);
    err = cudaGetLastError();
    checkCudaError(err, "Failed to launch init kernel");

    err = cudaDeviceSynchronize();
    checkCudaError(err, "Failed to synchronize after init kernel");

    // Copy result back to host
    err = cudaMemcpy(context, d_context, sizeof(GPU_TIGER_CTX), cudaMemcpyDeviceToHost);
    checkCudaError(err, "Failed to copy context back to host");

    // Clean up
    err = cudaFree(d_context);
    checkCudaError(err, "Failed to free device memory");
}

void host_TIGERUpdate_gpu(GPU_TIGER_CTX *context, const unsigned char *input, size_t len)
{
    GPU_TIGER_CTX *d_context;
    unsigned char *d_input;
    cudaError_t err;

    // Allocate device memory
    err = cudaMalloc(&d_context, sizeof(GPU_TIGER_CTX));
    checkCudaError(err, "Failed to allocate device memory for context");

    err = cudaMalloc(&d_input, len);
    checkCudaError(err, "Failed to allocate device memory for input");

    // Copy data to device
    err = cudaMemcpy(d_context, context, sizeof(GPU_TIGER_CTX), cudaMemcpyHostToDevice);
    checkCudaError(err, "Failed to copy context to device");

    err = cudaMemcpy(d_input, input, len, cudaMemcpyHostToDevice);
    checkCudaError(err, "Failed to copy input to device");

    // Launch kernel
    tiger_update_kernel<<<1, 1>>>(d_context, d_input, len);
    err = cudaGetLastError();
    checkCudaError(err, "Failed to launch update kernel");

    err = cudaDeviceSynchronize();
    checkCudaError(err, "Failed to synchronize after update kernel");

    // Copy result back to host
    err = cudaMemcpy(context, d_context, sizeof(GPU_TIGER_CTX), cudaMemcpyDeviceToHost);
    checkCudaError(err, "Failed to copy context back to host");

    // Clean up
    err = cudaFree(d_context);
    checkCudaError(err, "Failed to free device context memory");

    err = cudaFree(d_input);
    checkCudaError(err, "Failed to free device input memory");
}

void host_TIGER192Final_gpu(unsigned char digest[24], GPU_TIGER_CTX *context)
{
    GPU_TIGER_CTX *d_context;
    unsigned char *d_digest;
    cudaError_t err;

    // Allocate device memory
    err = cudaMalloc(&d_context, sizeof(GPU_TIGER_CTX));
    checkCudaError(err, "Failed to allocate device memory for context");

    err = cudaMalloc(&d_digest, 24);
    checkCudaError(err, "Failed to allocate device memory for digest");

    // Copy context to device
    err = cudaMemcpy(d_context, context, sizeof(GPU_TIGER_CTX), cudaMemcpyHostToDevice);
    checkCudaError(err, "Failed to copy context to device");

    // Launch kernel
    tiger_final_kernel<<<1, 1>>>(d_context, d_digest);
    err = cudaGetLastError();
    checkCudaError(err, "Failed to launch final kernel");

    err = cudaDeviceSynchronize();
    checkCudaError(err, "Failed to synchronize after final kernel");

    // Copy result back to host
    err = cudaMemcpy(digest, d_digest, 24, cudaMemcpyDeviceToHost);
    checkCudaError(err, "Failed to copy digest back to host");

    // Clean up
    err = cudaFree(d_context);
    checkCudaError(err, "Failed to free device context memory");

    err = cudaFree(d_digest);
    checkCudaError(err, "Failed to free device digest memory");
}

// Error checking helper function
void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

// Test function to compare CPU and GPU implementations
void test_implementations(const char *input)
{
    // CPU implementation
    TIGER_CTX cpu_ctx;
    unsigned char cpu_digest[24];

    TIGERInit(&cpu_ctx);
    TIGERUpdate(&cpu_ctx, (const unsigned char *)input, strlen(input));
    TIGER192Final(cpu_digest, &cpu_ctx);

    // GPU implementation
    GPU_TIGER_CTX gpu_ctx;
    unsigned char gpu_digest[24];

    host_TIGERInit_gpu(&gpu_ctx);
    host_TIGERUpdate_gpu(&gpu_ctx, (const unsigned char *)input, strlen(input));
    host_TIGER192Final_gpu(gpu_digest, &gpu_ctx);

    // Compare results
    printf("Input string: %s\n", input);
    printf("CPU hash: ");
    print_hash(cpu_digest);
    printf("GPU hash: ");
    print_hash(gpu_digest);

    int match = memcmp(cpu_digest, gpu_digest, 24) == 0;
    printf("Hash match: %s\n\n", match ? "YES" : "NO");
}

int main()
{
    // Test cases
    const char *test_cases[] = {
        "",                                                         // Empty string
        "a",                                                        // Single character
        "abc",                                                      // Classic test case
        "message digest",                                           // Longer string
        "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", // Long input
        "The quick brown fox jumps over the lazy dog"               // Sentence
    };

    int num_tests = sizeof(test_cases) / sizeof(test_cases[0]);

    printf("Running Tiger hash implementation comparison tests...\n\n");

    for (int i = 0; i < num_tests; i++)
    {
        test_implementations(test_cases[i]);
    }

    return 0;
}