#include <stdio.h>
#include <string.h>
#include <time.h>
#include "tiger_gpu.h"
#include <cuda_runtime.h>
#include "tiger.h"

// Constants for GPU bruteforce
#define THREADS_PER_BLOCK 256
#define NUM_BLOCKS 1024
#define CHARSET_SIZE 62 // a-z, A-Z, 0-9

__constant__ char d_charset[CHARSET_SIZE];
__constant__ unsigned char d_target[24];

// Helper function to generate test strings
__device__ void generate_string(char *buffer, size_t length, uint64_t index)
{
    const char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    const int charset_size = 62;

    for (size_t i = 0; i < length; i++)
    {
        buffer[i] = charset[index % charset_size];
        index /= charset_size;
    }
    buffer[length] = '\0';
}

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
    uint64_t h_attempts = 0;
    cudaError_t err;

    // Initialize CUDA memory
    err = cudaMemcpyToSymbol(d_target, target_hash, 24);
    checkCudaError(err, "Failed to copy target hash to constant memory");

    err = cudaMalloc(&d_found, sizeof(bool));
    checkCudaError(err, "Failed to allocate device memory for found flag");

    err = cudaMalloc(&d_result, 32); // Max string length + null terminator
    checkCudaError(err, "Failed to allocate device memory for result");

    err = cudaMalloc(&d_attempts, sizeof(uint64_t));
    checkCudaError(err, "Failed to allocate device memory for attempts counter");

    err = cudaMemset(d_found, 0, sizeof(bool));
    checkCudaError(err, "Failed to initialize found flag");

    err = cudaMemset(d_attempts, 0, sizeof(uint64_t));
    checkCudaError(err, "Failed to initialize attempts counter");

    uint64_t start_index = 0;
    clock_t start_time = clock();

    while ((double)(clock() - start_time) / CLOCKS_PER_SEC < time_limit && !h_found)
    {
        bruteforce_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
            length, start_index, d_found, d_result, d_attempts);

        err = cudaGetLastError();
        checkCudaError(err, "Failed to launch bruteforce kernel");

        err = cudaMemcpy(&h_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost);
        checkCudaError(err, "Failed to copy found flag from device");

        start_index += NUM_BLOCKS * THREADS_PER_BLOCK;

        // Periodically update attempts count
        if (start_index % (NUM_BLOCKS * THREADS_PER_BLOCK * 100) == 0)
        {
            err = cudaMemcpy(&h_attempts, d_attempts, sizeof(uint64_t), cudaMemcpyDeviceToHost);
            checkCudaError(err, "Failed to copy attempts counter from device");
            *total_attempts = h_attempts;
        }
    }

    // Get final attempt count
    err = cudaMemcpy(&h_attempts, d_attempts, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    checkCudaError(err, "Failed to copy final attempts counter from device");
    *total_attempts = h_attempts;

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

// Bruteforce kernel
__global__ void bruteforce_kernel(size_t length, uint64_t start_index, bool *found,
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
        // Generate test string from current_index
        generate_string(test_string, length, current_index);

        // Compute hash
        TIGERInit_gpu(&context);
        TIGERUpdate_gpu(&context, (const unsigned char *)test_string, length);
        TIGER192Final_gpu(hash, &context);

        // Increment attempts counter
        atomicAdd(attempts, 1ULL);

        // Compare hash with target
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
            // Copy the found string to result
            for (size_t i = 0; i <= length; i++)
            {
                result_string[i] = test_string[i];
            }
            return;
        }

        current_index += stride;
    }
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