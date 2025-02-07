#include <stdio.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include "tiger.h"
#include "tiger_tables.h"

// Constants
#define BLOCK_SIZE 256
#define NUM_BLOCKS 1024
#define CHARSET_SIZE 62

// Device constant memory
__constant__ uint64_t d_table[4 * 256];
__constant__ char d_charset[CHARSET_SIZE];
__constant__ unsigned char d_target[24];

// Function declarations for CPU Tiger hash functions that are defined in tiger.c
extern "C"
{
    void TIGERInit(TIGER_CTX *context);
    void TIGERUpdate(TIGER_CTX *context, const unsigned char *input, size_t length);
    void TIGER192Final(unsigned char digest[24], TIGER_CTX *context);
}

// GPU context structure
typedef struct
{
    uint64_t state[3];
    uint64_t passed;
    unsigned char buffer[64];
    uint32_t length;
} GPU_TIGER_CTX;

// Round macros for GPU implementation
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

__device__ void tiger_compress_gpu(uint64_t *str, uint64_t *state)
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

    a ^= aa;
    b -= bb;
    c += cc;

    state[0] = a;
    state[1] = b;
    state[2] = c;
}

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

// Helper functions
__device__ void generate_string(char *buffer, size_t length, uint64_t index)
{
    for (size_t i = 0; i < length; i++)
    {
        buffer[i] = d_charset[index % CHARSET_SIZE];
        index /= CHARSET_SIZE;
    }
    buffer[length] = '\0';
}

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

void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

void initialize_gpu_tables(void)
{
    // Initialize charset
    char charset[CHARSET_SIZE];
    int idx = 0;
    for (char c = 'a'; c <= 'z'; c++)
        charset[idx++] = c;
    for (char c = 'A'; c <= 'Z'; c++)
        charset[idx++] = c;
    for (char c = '0'; c <= '9'; c++)
        charset[idx++] = c;

    cudaError_t err;
    err = cudaMemcpyToSymbol(d_table, tiger_table, sizeof(uint64_t) * 4 * 256);
    checkCudaError(err, "Failed to copy table to GPU");

    err = cudaMemcpyToSymbol(d_charset, charset, CHARSET_SIZE);
    checkCudaError(err, "Failed to copy charset to GPU");
}

bool bruteforce_gpu(const unsigned char *target_hash, size_t length, double time_limit,
                    char *result, uint64_t *total_attempts)
{
    bool *d_found;
    char *d_result;
    unsigned long long *d_attempts;
    bool h_found = false;
    unsigned long long h_attempts = 0;
    cudaError_t err;

    // Initialize CUDA memory
    err = cudaMemcpyToSymbol(d_target, target_hash, 24);
    checkCudaError(err, "Failed to copy target hash to constant memory");

    err = cudaMalloc(&d_found, sizeof(bool));
    checkCudaError(err, "Failed to allocate device memory for found flag");

    err = cudaMalloc(&d_result, 32);
    checkCudaError(err, "Failed to allocate device memory for result");

    err = cudaMalloc(&d_attempts, sizeof(unsigned long long));
    checkCudaError(err, "Failed to allocate device memory for attempts counter");

    err = cudaMemset(d_found, 0, sizeof(bool));
    checkCudaError(err, "Failed to initialize found flag");

    err = cudaMemset(d_attempts, 0, sizeof(unsigned long long));
    checkCudaError(err, "Failed to initialize attempts counter");

    uint64_t start_index = 0;
    clock_t start_time = clock();

    while ((double)(clock() - start_time) / CLOCKS_PER_SEC < time_limit && !h_found)
    {
        bruteforce_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(
            length, start_index, d_found, d_result, d_attempts);

        err = cudaGetLastError();
        checkCudaError(err, "Failed to launch bruteforce kernel");

        err = cudaMemcpy(&h_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost);
        checkCudaError(err, "Failed to copy found flag from device");

        start_index += NUM_BLOCKS * BLOCK_SIZE;

        // Periodically update attempts count
        if (start_index % (NUM_BLOCKS * BLOCK_SIZE * 100) == 0)
        {
            err = cudaMemcpy(&h_attempts, d_attempts, sizeof(unsigned long long),
                             cudaMemcpyDeviceToHost);
            checkCudaError(err, "Failed to copy attempts counter from device");
            *total_attempts = (uint64_t)h_attempts;
        }
    }

    // Get final attempt count
    err = cudaMemcpy(&h_attempts, d_attempts, sizeof(unsigned long long),
                     cudaMemcpyDeviceToHost);
    checkCudaError(err, "Failed to copy final attempts counter from device");
    *total_attempts = (uint64_t)h_attempts;

    // If found, copy the result string
    if (h_found)
    {
        err = cudaMemcpy(result, d_result, length + 1, cudaMemcpyDeviceToHost);
        checkCudaError(err, "Failed to copy result string from device");
    }

    // Cleanup
    cudaFree(d_found);
    cudaFree(d_result);
    cudaFree(d_attempts);

    return h_found;
}

int main()
{
    srand(time(NULL));
    initialize_gpu_tables();

    // Test parameters
    const size_t max_length = 8;    // Maximum string length to test
    const double time_limit = 10.0; // Time limit per length in seconds

    printf("Starting GPU bruteforce test\n");
    printf("Testing strings up to length %zu\n", max_length);
    printf("Time limit per length: %.1f seconds\n\n", time_limit);

    for (size_t length = 1; length <= max_length; length++)
    {
        // Create a random target string and its hash
        char target_string[32];
        unsigned char target_hash[24];
        TIGER_CTX context;

        // Generate random target string
        for (size_t i = 0; i < length; i++)
        {
            target_string[i] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"[rand() % 62];
        }
        target_string[length] = '\0';

        // Generate its hash
        TIGERInit(&context);
        TIGERUpdate(&context, (const unsigned char *)target_string, length);
        TIGER192Final(target_hash, &context);

        printf("\nTesting length %zu\n", length);
        printf("Target string: %s\n", target_string);
        printf("Target hash: ");
        for (int i = 0; i < 24; i++)
            printf("%02x", target_hash[i]);
        printf("\n");

        // Try to find it
        char result[32];
        uint64_t attempts = 0;
        bool found = bruteforce_gpu(target_hash, length, time_limit, result, &attempts);

        if (found)
        {
            printf("Found match: %s\n", result);
            printf("Attempts: %lu\n", attempts);
            printf("Speed: %.2f million hashes/second\n",
                   (attempts / time_limit) / 1000000.0);
        }
        else
        {
            printf("No match found within time limit\n");
            printf("Attempts: %lu\n", attempts);
            printf("Speed: %.2f million hashes/second\n",
                   (attempts / time_limit) / 1000000.0);
            break; // Stop if we can't find a match at this length
        }
    }

    return 0;
}