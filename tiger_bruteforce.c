// tiger_bruteforce.c
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include "tiger.h"

// Function to generate a random string of given length
void generate_random_string(char *str, size_t length)
{
    const char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    for (size_t i = 0; i < length; i++)
    {
        int key = rand() % (sizeof(charset) - 1);
        str[i] = charset[key];
    }
    str[length] = '\0';
}

// Function to print hash in hexadecimal
void print_hash(unsigned char *hash, size_t length)
{
    for (size_t i = 0; i < length; i++)
    {
        printf("%02x", hash[i]);
    }
}

// Function to create a random target hash from a string of given length
void create_target_hash(unsigned char *target_hash, size_t length)
{
    TIGER_CTX context;
    char *original_string = (char *)malloc(length + 1);

    generate_random_string(original_string, length);

    TIGERInit(&context);
    TIGERUpdate(&context, (const unsigned char *)original_string, length);
    TIGER192Final(target_hash, &context);

    printf("Created target hash from string: %s\n", original_string);
    printf("Target hash: ");
    print_hash(target_hash, 24);
    printf("\n\n");

    free(original_string);
}

// Bruteforce function with time limit
int bruteforce_length(size_t length, double time_limit)
{
    TIGER_CTX context;
    unsigned char target_hash[24];
    unsigned char current_hash[24];
    char *test_string;
    clock_t start_time, current_time;
    double time_spent;
    size_t attempts = 0;
    int success = 0;

    printf("Testing length %zu:\n", length);

    // Create a target hash from a random string of the current length
    create_target_hash(target_hash, length);

    test_string = (char *)malloc(length + 1);
    start_time = clock();

    while (1)
    {
        // Generate and test random string
        generate_random_string(test_string, length);
        TIGERInit(&context);
        TIGERUpdate(&context, (const unsigned char *)test_string, length);
        TIGER192Final(current_hash, &context);

        attempts++;

        // Check if we found a match
        if (memcmp(current_hash, target_hash, 24) == 0)
        {
            current_time = clock();
            time_spent = (double)(current_time - start_time) / CLOCKS_PER_SEC;

            printf("Found match!\n");
            printf("String: %s\n", test_string);
            printf("Attempts: %zu\n", attempts);
            printf("Time taken: %.2f seconds\n", time_spent);
            printf("Speed: %.2f million hashes/second\n\n", (attempts / time_spent) / 1000000.0);

            success = 1;
            break;
        }

        // Check if we've exceeded the time limit
        if (attempts % 100000 == 0)
        {
            current_time = clock();
            time_spent = (double)(current_time - start_time) / CLOCKS_PER_SEC;

            if (time_spent >= time_limit)
            {
                printf("Time limit reached!\n");
                printf("Attempts: %zu\n", attempts);
                printf("Time taken: %.2f seconds\n", time_spent);
                printf("Speed: %.2f million hashes/second\n", (attempts / time_spent) / 1000000.0);
                printf("Could not find match within time limit\n\n");
                break;
            }

            // Print progress
            printf("\rAttempts: %zu | Time: %.2f s | Speed: %.2f MH/s",
                   attempts, time_spent, (attempts / time_spent) / 1000000.0);
            fflush(stdout);
        }
    }

    free(test_string);
    return success;
}

int main()
{
    srand(time(NULL));
    size_t current_length = 1;
    double time_limit = 10.0; // 10 seconds

    printf("Starting progressive bruteforce test\n");
    printf("Time limit per length: %.1f seconds\n\n", time_limit);

    while (1)
    {
        int success = bruteforce_length(current_length, time_limit);

        // If we couldn't find a match within the time limit, stop
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