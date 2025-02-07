#include "tiger_gpu.h"
#include "tiger.h"
#include "tiger_tables.h"

// Constants for GPU implementation
#define GPU_BLOCK_SIZE 256
#define GPU_NUM_BLOCKS 1024

// Tiger S-box tables in constant memory
__constant__ uint64_t d_table[4 * 256];

// Initialize S-box tables on GPU
void initialize_gpu_tables()
{
    static int tables_initialized = 0;
    if (!tables_initialized)
    {
        cudaError_t err = cudaMemcpyToSymbol(d_table, table, sizeof(uint64_t) * 4 * 256);
        checkCudaError(err, "Failed to copy S-box tables to GPU");
        tables_initialized = 1;
    }
}

// Error checking helper
void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

// Device implementation of Tiger hash functions
__device__ void TIGERInit_gpu(GPU_TIGER_CTX *context)
{
    context->state[0] = 0x0123456789ABCDEFULL;
    context->state[1] = 0xFEDCBA9876543210ULL;
    context->state[2] = 0xF096A5B4C3B2E187ULL;
    context->passed = 0;
    context->length = 0;
    memset(context->buffer, 0, sizeof(context->buffer));
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

// Host wrapper implementations
void host_TIGERInit_gpu(GPU_TIGER_CTX *context)
{
    GPU_TIGER_CTX *d_context;
    cudaError_t err;

    err = cudaMalloc(&d_context, sizeof(GPU_TIGER_CTX));
    checkCudaError(err, "Failed to allocate device memory for context");

    tiger_init_kernel<<<1, 1>>>(d_context);
    err = cudaGetLastError();
    checkCudaError(err, "Failed to launch init kernel");

    err = cudaDeviceSynchronize();
    checkCudaError(err, "Failed to synchronize after init kernel");

    err = cudaMemcpy(context, d_context, sizeof(GPU_TIGER_CTX), cudaMemcpyDeviceToHost);
    checkCudaError(err, "Failed to copy context back to host");

    cudaFree(d_context);
}

void host_TIGERUpdate_gpu(GPU_TIGER_CTX *context, const unsigned char *input, size_t len)
{
    GPU_TIGER_CTX *d_context;
    unsigned char *d_input;
    cudaError_t err;

    err = cudaMalloc(&d_context, sizeof(GPU_TIGER_CTX));
    checkCudaError(err, "Failed to allocate device memory for context");

    err = cudaMalloc(&d_input, len);
    checkCudaError(err, "Failed to allocate device memory for input");

    err = cudaMemcpy(d_context, context, sizeof(GPU_TIGER_CTX), cudaMemcpyHostToDevice);
    checkCudaError(err, "Failed to copy context to device");

    err = cudaMemcpy(d_input, input, len, cudaMemcpyHostToDevice);
    checkCudaError(err, "Failed to copy input to device");

    tiger_update_kernel<<<1, 1>>>(d_context, d_input, len);
    err = cudaGetLastError();
    checkCudaError(err, "Failed to launch update kernel");

    err = cudaDeviceSynchronize();
    checkCudaError(err, "Failed to synchronize after update kernel");

    err = cudaMemcpy(context, d_context, sizeof(GPU_TIGER_CTX), cudaMemcpyDeviceToHost);
    checkCudaError(err, "Failed to copy context back to host");

    cudaFree(d_context);
    cudaFree(d_input);
}

void host_TIGER192Final_gpu(unsigned char digest[24], GPU_TIGER_CTX *context)
{
    GPU_TIGER_CTX *d_context;
    unsigned char *d_digest;
    cudaError_t err;

    err = cudaMalloc(&d_context, sizeof(GPU_TIGER_CTX));
    checkCudaError(err, "Failed to allocate device memory for context");

    err = cudaMalloc(&d_digest, 24);
    checkCudaError(err, "Failed to allocate device memory for digest");

    err = cudaMemcpy(d_context, context, sizeof(GPU_TIGER_CTX), cudaMemcpyHostToDevice);
    checkCudaError(err, "Failed to copy context to device");

    tiger_final_kernel<<<1, 1>>>(d_context, d_digest);
    err = cudaGetLastError();
    checkCudaError(err, "Failed to launch final kernel");

    err = cudaDeviceSynchronize();
    checkCudaError(err, "Failed to synchronize after final kernel");

    err = cudaMemcpy(digest, d_digest, 24, cudaMemcpyDeviceToHost);
    checkCudaError(err, "Failed to copy digest back to host");

    cudaFree(d_context);
    cudaFree(d_digest);
}

// Utility function to print hash
void print_hash(unsigned char *hash)
{
    for (int i = 0; i < 24; i++)
    {
        printf("%02x", hash[i]);
    }
    printf("\n");
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
