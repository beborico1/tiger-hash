// tiger_gpu.cu
#include "tiger_gpu.h"
#include "tiger_tables.h"
#include <stdio.h>
#include <string.h>
#include "tiger_commmon.h"

// Tiger S-box tables in constant memory
__constant__ uint64_t d_table[4 * 256];

// Macros for GPU implementation
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

// GPU implementation of tiger_compress
__device__ static void tiger_compress_gpu(const uint64_t *str, uint64_t *state)
{
    uint64_t a, b, c, aa, bb, cc;
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

    GPU_PASS(a, b, c, 5)
    GPU_KEY_SCHEDULE

    GPU_PASS(c, a, b, 7)
    GPU_KEY_SCHEDULE

    GPU_PASS(b, c, a, 9)

    a ^= aa;
    b -= bb;
    c += cc;

    state[0] = a;
    state[1] = b;
    state[2] = c;
}

// Define the actual device functions that were declared in tiger_gpu.h
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

            tiger_compress_gpu((const uint64_t *)context->buffer, context->state);
            context->passed += 512;
            i = fill;
        }

        for (; i + 64 <= len; i += 64)
        {
            for (int j = 0; j < 64; j++)
            {
                context->buffer[j] = input[i + j];
            }
            tiger_compress_gpu((const uint64_t *)context->buffer, context->state);
            context->passed += 512;
        }

        context->length = len - i;
        for (size_t j = 0; j < context->length; j++)
        {
            context->buffer[j] = input[i + j];
        }
    }
}

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
        tiger_compress_gpu((const uint64_t *)context->buffer, context->state);
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

    tiger_compress_gpu((const uint64_t *)context->buffer, context->state);

    for (int i = 0; i < 24; i++)
    {
        digest[i] = (context->state[i / 8] >> (8 * (i % 8))) & 0xFF;
    }
}

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

void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

// Add these kernel functions:
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

void host_TIGERInit_gpu(GPU_TIGER_CTX *context)
{
    GPU_TIGER_CTX *d_context;
    cudaMalloc(&d_context, sizeof(GPU_TIGER_CTX));

    // Launch kernel
    dim3 block(1);
    dim3 grid(1);
    tiger_init_kernel<<<grid, block>>>(d_context);

    // Copy result back
    cudaMemcpy(context, d_context, sizeof(GPU_TIGER_CTX), cudaMemcpyDeviceToHost);
    cudaFree(d_context);
}

void host_TIGERUpdate_gpu(GPU_TIGER_CTX *context, const unsigned char *input, size_t len)
{
    GPU_TIGER_CTX *d_context;
    unsigned char *d_input;

    cudaMalloc(&d_context, sizeof(GPU_TIGER_CTX));
    cudaMalloc(&d_input, len);

    cudaMemcpy(d_context, context, sizeof(GPU_TIGER_CTX), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, input, len, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 block(1);
    dim3 grid(1);
    tiger_update_kernel<<<grid, block>>>(d_context, d_input, len);

    cudaMemcpy(context, d_context, sizeof(GPU_TIGER_CTX), cudaMemcpyDeviceToHost);

    cudaFree(d_context);
    cudaFree(d_input);
}

void host_TIGER192Final_gpu(unsigned char digest[24], GPU_TIGER_CTX *context)
{
    GPU_TIGER_CTX *d_context;
    unsigned char *d_digest;

    cudaMalloc(&d_context, sizeof(GPU_TIGER_CTX));
    cudaMalloc(&d_digest, 24);

    cudaMemcpy(d_context, context, sizeof(GPU_TIGER_CTX), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 block(1);
    dim3 grid(1);
    tiger_final_kernel<<<grid, block>>>(d_context, d_digest);

    cudaMemcpy(digest, d_digest, 24, cudaMemcpyDeviceToHost);

    cudaFree(d_context);
    cudaFree(d_digest);
}
