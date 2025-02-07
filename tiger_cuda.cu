#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "tiger.h"

// Constants for Tiger hash
#define BLOCK_SIZE 256
#define NUM_BLOCKS 1024

// GPU Tiger context structure
typedef struct
{
    uint64_t state[3];
    uint64_t passed;
    unsigned char buffer[64];
    uint32_t length;
} GPU_TIGER_CTX;

// Tiger S-box tables in constant memory
__constant__ uint64_t d_table[4 * 256];

// Macros for Tiger hash GPU implementation
#define GPU_SAVE_ABC \
    aa = a;          \
    bb = b;          \
    cc = c;

#define GPU_ROUND(a, b, c, x, mul)                                                 \
    c ^= x;                                                                        \
    a -= d_table[(unsigned char)(c)] ^                                             \
         d_table[256 + (unsigned char)(((uint32_t)(c)) >> (2 * 8))] ^              \
         d_table[512 + (unsigned char)((c) >> (4 * 8))] ^                          \
         d_table[768 + (unsigned char)(((uint32_t)((c) >> (4 * 8))) >> (2 * 8))];  \
    b += d_table[768 + (unsigned char)(((uint32_t)(c)) >> (1 * 8))] ^              \
         d_table[512 + (unsigned char)(((uint32_t)(c)) >> (3 * 8))] ^              \
         d_table[256 + (unsigned char)(((uint32_t)((c) >> (4 * 8))) >> (1 * 8))] ^ \
         d_table[(unsigned char)(((uint32_t)((c) >> (4 * 8))) >> (3 * 8))];        \
    b *= mul;

#define GPU_PASS(a, b, c, mul)  \
    GPU_ROUND(a, b, c, x0, mul) \
    GPU_ROUND(b, c, a, x1, mul) \
    GPU_ROUND(c, a, b, x2, mul) \
    GPU_ROUND(a, b, c, x3, mul) \
    GPU_ROUND(b, c, a, x4, mul) \
    GPU_ROUND(c, a, b, x5, mul) \
    GPU_ROUND(a, b, c, x6, mul) \
    GPU_ROUND(b, c, a, x7, mul)

#define GPU_KEY_SCHEDULE              \
    x0 -= x7 ^ 0xA5A5A5A5A5A5A5A5ULL; \
    x1 ^= x0;                         \
    x2 += x1;                         \
    x3 -= x2 ^ ((~x1) << 19);         \
    x4 ^= x3;                         \
    x5 += x4;                         \
    x6 -= x5 ^ ((~x4) >> 23);         \
    x7 ^= x6;                         \
    x0 += x7;                         \
    x1 -= x0 ^ ((~x7) << 19);         \
    x2 ^= x1;                         \
    x3 += x2;                         \
    x4 -= x3 ^ ((~x2) >> 23);         \
    x5 ^= x4;                         \
    x6 += x5;                         \
    x7 -= x6 ^ 0x0123456789ABCDEFULL;

#define GPU_FEEDFORWARD \
    a ^= aa;            \
    b -= bb;            \
    c += cc;

// GPU implementation of tiger_compress
__device__ void tiger_compress_gpu(uint64_t *str, uint64_t *state)
{
    uint64_t a, b, c;
    uint64_t aa, bb, cc;
    uint64_t x0, x1, x2, x3, x4, x5, x6, x7;

    a = state[0];
    b = state[1];
    c = state[2];

    x0 = str[0];
    x1 = str[1];
    x2 = str[2];
    x3 = str[3];
    x4 = str[4];
    x5 = str[5];
    x6 = str[6];
    x7 = str[7];

    GPU_SAVE_ABC

    // Pass 1
    GPU_PASS(a, b, c, 5)
    GPU_KEY_SCHEDULE

    // Pass 2
    GPU_PASS(c, a, b, 7)
    GPU_KEY_SCHEDULE

    // Pass 3
    GPU_PASS(b, c, a, 9)

    GPU_FEEDFORWARD

    state[0] = a;
    state[1] = b;
    state[2] = c;
}

// GPU initialization function
__device__ void TIGERInit_gpu(GPU_TIGER_CTX *context)
{
    context->state[0] = 0x0123456789ABCDEFULL;
    context->state[1] = 0xFEDCBA9876543210ULL;
    context->state[2] = 0xF096A5B4C3B2E187ULL;
    context->passed = 0;
    context->length = 0;

    for (int i = 0; i < 64; i++)
    {
        context->buffer[i] = 0;
    }
}

// GPU update function
__device__ void TIGERUpdate_gpu(GPU_TIGER_CTX *context, const unsigned char *input, size_t len)
{
    if (context->length + len < 64)
    {
        for (size_t i = 0; i < len; i++)
        {
            context->buffer[context->length + i] = input[i];
        }
        context->length += len;
    }
    else
    {
        size_t i = 0;

        if (context->length)
        {
            size_t fill = 64 - context->length;
            for (size_t j = 0; j < fill; j++)
            {
                context->buffer[context->length + j] = input[j];
            }

            tiger_compress_gpu((uint64_t *)context->buffer, context->state);
            context->passed += 512;
            i = fill;
        }

        while (i + 64 <= len)
        {
            for (int j = 0; j < 64; j++)
            {
                context->buffer[j] = input[i + j];
            }
            tiger_compress_gpu((uint64_t *)context->buffer, context->state);
            context->passed += 512;
            i += 64;
        }

        context->length = len - i;
        for (size_t j = 0; j < context->length; j++)
        {
            context->buffer[j] = input[i + j];
        }
    }
}

// GPU final function
__device__ void TIGER192Final_gpu(unsigned char digest[24], GPU_TIGER_CTX *context)
{
    context->buffer[context->length] = 0x01;
    context->length++;

    for (size_t i = context->length; i < 64; i++)
    {
        context->buffer[i] = 0;
    }

    if (context->length > 56)
    {
        tiger_compress_gpu((uint64_t *)context->buffer, context->state);

        for (int i = 0; i < 64; i++)
        {
            context->buffer[i] = 0;
        }
    }

    uint64_t bits = context->passed + (context->length << 3);
    for (int i = 0; i < 8; i++)
    {
        context->buffer[56 + i] = (bits >> (i * 8)) & 0xFF;
    }

    tiger_compress_gpu((uint64_t *)context->buffer, context->state);

    for (int i = 0; i < 24; i++)
    {
        digest[i] = (context->state[i / 8] >> (8 * (i % 8))) & 0xFF;
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