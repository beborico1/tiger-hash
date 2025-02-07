#include <stdio.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include "tiger.h"

// Constants for GPU bruteforce
#define THREADS_PER_BLOCK 256
#define NUM_BLOCKS 1024
#define CHARSET_SIZE 62 // a-z, A-Z, 0-9

__constant__ char d_charset[CHARSET_SIZE];
__constant__ unsigned char d_target[24];

// Host-side charset initialization
void initialize_charset()
{
    char charset[CHARSET_SIZE];
    int idx = 0;

    // Add lowercase letters
    for (char c = 'a'; c <= 'z'; c++)
    {
        charset[idx++] = c;
    }

    // Add uppercase letters
    for (char c = 'A'; c <= 'Z'; c++)
    {
        charset[idx++] = c;
    }

    // Add numbers
    for (char c = '0'; c <= '9'; c++)
    {
        charset[idx++] = c;
    }

    cudaMemcpyToSymbol(d_charset, charset, CHARSET_SIZE);
}

// Device-side test function
__device__ void generate_string(char *buffer, size_t length, uint64_t index)
{
    for (size_t i = 0; i < length; i++)
    {
        buffer[i] = d_charset[index % CHARSET_SIZE];
        index /= CHARSET_SIZE;
    }
    buffer[length] = '\0';
}

__device__ bool compare_hash(unsigned char *hash1, const unsigned char *hash2)
{
    for (int i = 0; i < 24; i++)
    {
        if (hash1[i] != hash2[i])
            return false;
    }
    return true;
}

// Kernel for bruteforce
__global__ void bruteforce_kernel(size_t length, uint64_t start_index, volatile bool *found,
                                  char *result_string, uint64_t *attempts)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * blockDim.x;
    uint64_t current_index = start_index + tid;

    GPU_TIGER_CTX context;
    char test_string[32]; // Max length we'll test
    unsigned char hash[24];

    while (!(*found))
    {
        generate_string(test_string, length, current_index);

        // Compute hash
        TIGERInit_gpu(&context);
        TIGERUpdate_gpu(&context, (unsigned char *)test_string, length);
        TIGER192Final_gpu(hash, &context);

        atomicAdd((unsigned long long *)attempts, 1ULL);

        if (compare_hash(hash, d_target))
        {
            *found = true;
            memcpy(result_string, test_string, length + 1);
            break;
        }

        current_index += stride;
    }
}

// Host-side bruteforce function
bool bruteforce_gpu(const unsigned char *target_hash, size_t length, double time_limit,
                    char *result, uint64_t *total_attempts)
{
    bool *d_found;
    char *d_result;
    uint64_t *d_attempts;
    bool h_found = false;
    char h_result[32];
    uint64_t h_attempts = 0;

    // Initialize CUDA memory
    cudaMemcpyToSymbol(d_target, target_hash, 24);
    cudaMalloc(&d_found, sizeof(bool));
    cudaMalloc(&d_result, 32);
    cudaMalloc(&d_attempts, sizeof(uint64_t));

    cudaMemset(d_found, 0, sizeof(bool));
    cudaMemset(d_attempts, 0, sizeof(uint64_t));

    uint64_t start_index = 0;
    clock_t start_time = clock();

    while ((double)(clock() - start_time) / CLOCKS_PER_SEC < time_limit && !h_found)
    {
        bruteforce_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
            length, start_index, d_found, d_result, d_attempts);

        cudaMemcpy(&h_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost);
        start_index += NUM_BLOCKS * THREADS_PER_BLOCK;
    }

    if (h_found)
    {
        cudaMemcpy(result, d_result, 32, cudaMemcpyDeviceToHost);
    }

    cudaMemcpy(total_attempts, d_attempts, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_found);
    cudaFree(d_result);
    cudaFree(d_attempts);

    return h_found;
}

int main()
{
    initialize_charset();

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
            target_string[i] = d_charset[rand() % CHARSET_SIZE];
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