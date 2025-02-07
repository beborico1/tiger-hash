// tiger_cuda.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

// Constants
#define BLOCK_SIZE 256
#define NUM_BLOCKS 1024
#define CHARSET_SIZE 62

// Forward declarations
__device__ void tiger_compress_gpu(uint64_t *str, uint64_t *state);

// Device constant memory for lookup tables and charset
__constant__ uint64_t d_table[4 * 256];
__constant__ char d_charset[CHARSET_SIZE];
__constant__ unsigned char d_target[24];

// GPU context structure
typedef struct
{
    uint64_t state[3];
    uint64_t passed;
    unsigned char buffer[64];
    uint32_t length;
} GPU_TIGER_CTX;

// Round and key schedule macros
#define ROUND(a, b, c, x, mul)                                                         \
    {                                                                                  \
        c ^= x;                                                                        \
        a -= d_table[(unsigned char)(c)] ^                                             \
             d_table[256 + (unsigned char)(((uint32_t)(c)) >> (2 * 8))] ^              \
             d_table[512 + (unsigned char)((c) >> (4 * 8))] ^                          \
             d_table[768 + (unsigned char)(((uint32_t)((c) >> (4 * 8))) >> (2 * 8))];  \
        b += d_table[768 + (unsigned char)(((uint32_t)(c)) >> (1 * 8))] ^              \
             d_table[512 + (unsigned char)(((uint32_t)(c)) >> (3 * 8))] ^              \
             d_table[256 + (unsigned char)(((uint32_t)((c) >> (4 * 8))) >> (1 * 8))] ^ \
             d_table[(unsigned char)(((uint32_t)((c) >> (4 * 8))) >> (3 * 8))];        \
        b *= mul;                                                                      \
    }

#define KEY_SCHEDULE                      \
    {                                     \
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
        x7 -= x6 ^ 0x0123456789ABCDEFULL; \
    }

// Implementation of tiger_compress_gpu
__device__ void tiger_compress_gpu(uint64_t *str, uint64_t *state)
{
    uint64_t a, b, c, aa, bb, cc;
    uint64_t x0, x1, x2, x3, x4, x5, x6, x7;

    // Load state
    a = state[0];
    b = state[1];
    c = state[2];

    // Load message block
    x0 = str[0];
    x1 = str[1];
    x2 = str[2];
    x3 = str[3];
    x4 = str[4];
    x5 = str[5];
    x6 = str[6];
    x7 = str[7];

    // Save state
    aa = a;
    bb = b;
    cc = c;

    // Round 1
    ROUND(a, b, c, x0, 5)
    ROUND(b, c, a, x1, 5)
    ROUND(c, a, b, x2, 5)
    ROUND(a, b, c, x3, 5)
    ROUND(b, c, a, x4, 5)
    ROUND(c, a, b, x5, 5)
    ROUND(a, b, c, x6, 5)
    ROUND(b, c, a, x7, 5)

    KEY_SCHEDULE

    // Round 2
    ROUND(c, a, b, x0, 7)
    ROUND(a, b, c, x1, 7)
    ROUND(b, c, a, x2, 7)
    ROUND(c, a, b, x3, 7)
    ROUND(a, b, c, x4, 7)
    ROUND(b, c, a, x5, 7)
    ROUND(c, a, b, x6, 7)
    ROUND(a, b, c, x7, 7)

    KEY_SCHEDULE

    // Round 3
    ROUND(b, c, a, x0, 9)
    ROUND(c, a, b, x1, 9)
    ROUND(a, b, c, x2, 9)
    ROUND(b, c, a, x3, 9)
    ROUND(c, a, b, x4, 9)
    ROUND(a, b, c, x5, 9)
    ROUND(b, c, a, x6, 9)
    ROUND(c, a, b, x7, 9)

    // Feedforward
    a ^= aa;
    b -= bb;
    c += cc;

    state[0] = a;
    state[1] = b;
    state[2] = c;
}

// Implementation of TIGERInit_gpu
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

// Implementation of TIGERUpdate_gpu
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

// Implementation of TIGER192Final_gpu
__device__ void TIGER192Final_gpu(unsigned char digest[24], GPU_TIGER_CTX *context)
{
    context->buffer[context->length++] = 0x01;

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

// Implementation of atomicAdd64
__device__ unsigned long long atomicAdd64(unsigned long long *address, unsigned long long val)
{
    unsigned long long old = *address, assumed;
    do
    {
        assumed = old;
        old = atomicCAS(address, assumed, val + assumed);
    } while (assumed != old);
    return old;
}

// String generation for bruteforce
__device__ void generate_string(char *buffer, size_t length, uint64_t index)
{
    for (size_t i = 0; i < length; i++)
    {
        buffer[i] = d_charset[index % CHARSET_SIZE];
        index /= CHARSET_SIZE;
    }
    buffer[length] = '\0';
}

// Bruteforce kernel
__global__ void bruteforce_kernel(size_t length, uint64_t start_index, bool *found,
                                  char *result_string, unsigned long long *attempts)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * blockDim.x;
    uint64_t current_index = start_index + tid;

    char test_string[32];
    unsigned char hash[24];
    GPU_TIGER_CTX context;

    while (!(*found))
    {
        generate_string(test_string, length, current_index);

        TIGERInit_gpu(&context);
        TIGERUpdate_gpu(&context, (const unsigned char *)test_string, length);
        TIGER192Final_gpu(hash, &context);

        atomicAdd64(attempts, 1ULL);

        bool match = true;
        for (int i = 0; i < 24; i++)
        {
            if (hash[i] != d_target[i])
            {
                match = false;
                break;
            }
        }

        if (match)
        {
            *found = true;
            for (size_t i = 0; i <= length; i++)
            {
                result_string[i] = test_string[i];
            }
            return;
        }

        current_index += stride;
    }
}

// Utility function to check CUDA errors
void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

// Function to initialize tables in GPU memory
extern "C" void initialize_gpu_tables(const uint64_t *host_table, const char *charset)
{
    cudaError_t err;

    err = cudaMemcpyToSymbol(d_table, host_table, sizeof(uint64_t) * 4 * 256);
    checkCudaError(err, "Failed to copy table to GPU");

    err = cudaMemcpyToSymbol(d_charset, charset, CHARSET_SIZE);
    checkCudaError(err, "Failed to copy charset to GPU");
}