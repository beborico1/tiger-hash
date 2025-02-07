// tiger_cuda.c
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

// Constants
#define BLOCK_SIZE 256
#define NUM_BLOCKS 1024
#define CHARSET_LENGTH 62

// Tiger S-box tables in constant memory
__constant__ uint64_t d_table[4 * 256];

extern "C"
{
    extern const uint64_t tiger_table[4 * 256];
}

// GPU Context structure
typedef struct
{
    uint64_t state[3];
    uint64_t passed;
    unsigned char buffer[64];
    uint32_t length;
} GPU_TIGER_CTX;

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

__device__ void tiger_compress_gpu(uint64_t *str, uint64_t *state)
{
    uint64_t a, b, c, tmpa;
    uint64_t aa, bb, cc;
    uint64_t x0, x1, x2, x3, x4, x5, x6, x7;

    a = state[0];
    b = state[1];
    c = state[2];

    // Load str into x0-x7
    x0 = str[0];
    x1 = str[1];
    x2 = str[2];
    x3 = str[3];
    x4 = str[4];
    x5 = str[5];
    x6 = str[6];
    x7 = str[7];

    // Save abc
    GPU_SAVE_ABC

    // Pass 1
    GPU_PASS(a, b, c, 5)
    GPU_KEY_SCHEDULE

    // Pass 2
    GPU_PASS(c, a, b, 7)
    GPU_KEY_SCHEDULE

    // Pass 3
    GPU_PASS(b, c, a, 9)

    // Feedforward
    GPU_FEEDFORWARD

    state[0] = a;
    state[1] = b;
    state[2] = c;
}

__device__ void TIGERUpdate_gpu(GPU_TIGER_CTX *context, const unsigned char *input, size_t len)
{
    if (context->length + len < 64)
    {
        // If we don't have enough data to process a full block
        for (size_t i = 0; i < len; i++)
        {
            context->buffer[context->length + i] = input[i];
        }
        context->length += len;
    }
    else
    {
        size_t i = 0;

        // First, fill any partial block
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

        // Process full blocks
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

        // Store remaining bytes
        context->length = len - i;
        for (size_t j = 0; j < context->length; j++)
        {
            context->buffer[j] = input[i + j];
        }
    }
}

__device__ void TIGER192Final_gpu(unsigned char digest[24], GPU_TIGER_CTX *context)
{
    // Add padding bits
    context->buffer[context->length] = 0x01;
    context->length++;

    // Zero fill rest of block
    for (size_t i = context->length; i < 64; i++)
    {
        context->buffer[i] = 0;
    }

    // If length is > 56 bytes, process this block and create a new one
    if (context->length > 56)
    {
        tiger_compress_gpu((uint64_t *)context->buffer, context->state);

        // Create new zero-filled block
        for (int i = 0; i < 64; i++)
        {
            context->buffer[i] = 0;
        }
    }

    // Append bit length
    uint64_t bits = context->passed + (context->length << 3);
    for (int i = 0; i < 8; i++)
    {
        context->buffer[56 + i] = (bits >> (i * 8)) & 0xFF;
    }

    // Process final block
    tiger_compress_gpu((uint64_t *)context->buffer, context->state);

    // Output hash
    for (int i = 0; i < 24; i++)
    {
        digest[i] = (context->state[i / 8] >> (8 * (i % 8))) & 0xFF;
    }
}

// Generate random string on GPU
__device__ void generate_random_string_gpu(char *str, size_t length, curandState *state)
{
    const char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

    for (size_t i = 0; i < length; i++)
    {
        int key = curand(state) % CHARSET_LENGTH;
        str[i] = charset[key];
    }
    str[length] = '\0';
}

// Main bruteforce kernel
__global__ void bruteforce_kernel(
    unsigned char *target_hash,
    size_t length,
    volatile int *found,
    char *result,
    unsigned long long *attempts,
    curandState *rand_states)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize random number generator
    curand_init(clock64(), tid, 0, &rand_states[tid]);

    char test_string[20]; // Adjust size based on max length
    unsigned char current_hash[24];
    GPU_TIGER_CTX context;

    while (!(*found))
    {
        // Generate random string
        generate_random_string_gpu(test_string, length, &rand_states[tid]);

        // Calculate hash
        TIGERInit_gpu(&context);
        TIGERUpdate_gpu(&context, (unsigned char *)test_string, length);
        TIGER192Final_gpu(current_hash, &context);

        atomicAdd((unsigned long long *)attempts, 1ULL);

        // Compare with target
        bool match = true;
        for (int i = 0; i < 24; i++)
        {
            if (current_hash[i] != target_hash[i])
            {
                match = false;
                break;
            }
        }

        if (match)
        {
            *found = 1;
            memcpy(result, test_string, length);
            break;
        }
    }
}

// Host-side function to create target hash
void create_target_hash(unsigned char *target_hash, size_t length, char *original_string)
{
    GPU_TIGER_CTX context;
    generate_random_string(original_string, length); // Host-side random string generation

    TIGERInit_gpu(&context);
    TIGERUpdate_gpu(&context, (unsigned char *)original_string, length);
    TIGER192Final_gpu(target_hash, &context);

    printf("Created target hash from string: %s\n", original_string);
    printf("Target hash: ");
    for (int i = 0; i < 24; i++)
    {
        printf("%02x", target_hash[i]);
    }
    printf("\n\n");
}

// Main host function
int bruteforce_length(size_t length, double time_limit)
{
    // Allocate device memory
    unsigned char *d_target_hash;
    volatile int *d_found;
    char *d_result;
    unsigned long long *d_attempts;
    curandState *d_rand_states;

    cudaMalloc(&d_target_hash, 24);
    cudaMalloc(&d_found, sizeof(int));
    cudaMalloc(&d_result, 20); // Adjust based on max length
    cudaMalloc(&d_attempts, sizeof(unsigned long long));
    cudaMalloc(&d_rand_states, BLOCK_SIZE * NUM_BLOCKS * sizeof(curandState));

    // Create target hash
    unsigned char target_hash[24];
    char original_string[20];
    create_target_hash(target_hash, length, original_string);

    // Copy target hash to device
    cudaMemcpy(d_target_hash, target_hash, 24, cudaMemcpyHostToDevice);

    // Initialize found flag and attempts counter
    int host_found = 0;
    unsigned long long host_attempts = 0;
    cudaMemcpy((void *)d_found, &host_found, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_attempts, &host_attempts, sizeof(unsigned long long), cudaMemcpyHostToDevice);

    // Launch kernel
    clock_t start_time = clock();
    bruteforce_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(
        d_target_hash, length, d_found, d_result, d_attempts, d_rand_states);

    // Monitor progress
    char result[20];
    while ((double)(clock() - start_time) / CLOCKS_PER_SEC < time_limit)
    {
        cudaMemcpy(&host_found, (void *)d_found, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&host_attempts, d_attempts, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

        if (host_found)
        {
            cudaMemcpy(result, d_result, length, cudaMemcpyDeviceToHost);
            double time_spent = (double)(clock() - start_time) / CLOCKS_PER_SEC;

            printf("Found match!\n");
            printf("String: %s\n", result);
            printf("Attempts: %llu\n", host_attempts);
            printf("Time taken: %.2f seconds\n", time_spent);
            printf("Speed: %.2f million hashes/second\n\n",
                   (host_attempts / time_spent) / 1000000.0);

            cudaFree(d_target_hash);
            cudaFree(d_found);
            cudaFree(d_result);
            cudaFree(d_attempts);
            cudaFree(d_rand_states);
            return 1;
        }

        // Print progress
        if (host_attempts % 1000000 == 0)
        {
            double time_spent = (double)(clock() - start_time) / CLOCKS_PER_SEC;
            printf("\rAttempts: %llu | Time: %.2f s | Speed: %.2f MH/s",
                   host_attempts, time_spent, (host_attempts / time_spent) / 1000000.0);
            fflush(stdout);
        }
    }

    // Time limit reached without finding match
    printf("\nTime limit reached!\n");
    printf("Attempts: %llu\n", host_attempts);
    double time_spent = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    printf("Time taken: %.2f seconds\n", time_spent);
    printf("Speed: %.2f million hashes/second\n",
           (host_attempts / time_spent) / 1000000.0);
    printf("Could not find match within time limit\n\n");

    cudaFree(d_target_hash);
    cudaFree(d_found);
    cudaFree(d_result);
    cudaFree(d_attempts);
    cudaFree(d_rand_states);
    return 0;
}

int main()
{
    // Initialize Tiger tables in constant memory
    cudaMemcpyToSymbol(d_table, table, sizeof(table));

    size_t current_length = 1;
    double time_limit = 10.0; // 10 seconds

    printf("Starting progressive bruteforce test\n");
    printf("Time limit per length: %.1f seconds\n\n", time_limit);

    while (1)
    {
        printf("Testing length %zu:\n", current_length);
        int success = bruteforce_length(current_length, time_limit);

        if (!success)
        {
            printf("Stopping at length %zu as it could not be bruteforced within %.1f seconds\n",
                   current_length, time_limit);
            break;
        }

        current_length++;
    }

    return 0;
}

__device__ void TIGERInit_gpu(GPU_TIGER_CTX *context)
{
    context->state[0] = 0x0123456789ABCDEFULL;
    context->state[1] = 0xFEDCBA9876543210ULL;
    context->state[2] = 0xF096A5B4C3B2E187ULL;
    context->passed = 0;
    context->length = 0;

    // Clear buffer
    for (int i = 0; i < 64; i++)
    {
        context->buffer[i] = 0;
    }
}

// Add this before your main() function
void init_gpu_tables()
{
    // Copy Tiger tables to constant memory
    cudaError_t err = cudaMemcpyToSymbol(d_table, table, sizeof(table));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy Tiger tables to GPU: %s\n",
                cudaGetErrorString(err));
        exit(1);
    }
}