#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include "tiger.h"

// Function declarations for CUDA implementation
void TIGERInit_gpu(GPU_TIGER_CTX *context);
void TIGERUpdate_gpu(GPU_TIGER_CTX *context, const unsigned char *input, size_t len);
void TIGER192Final_gpu(unsigned char digest[24], GPU_TIGER_CTX *context);

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

    TIGERInit_gpu(&gpu_ctx);
    TIGERUpdate_gpu(&gpu_ctx, (const unsigned char *)input, strlen(input));
    TIGER192Final_gpu(gpu_digest, &gpu_ctx);

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