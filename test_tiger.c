// test_tiger.c
#include <stdio.h>
#include <string.h>
#include <stdint.h>

// Define PHP specific macros
#define PHP_HASH_API
#define ZEND_ATTRIBUTE_UNUSED
#define HashTable void

// Tiger context structure
typedef struct
{
    uint64_t state[3];
    uint64_t passed;
    unsigned char buffer[64];
    uint32_t length;
    unsigned int passes : 1;
} PHP_TIGER_CTX;

// Function declarations
void PHP_3TIGERInit(PHP_TIGER_CTX *context, HashTable *args);
void PHP_TIGERUpdate(PHP_TIGER_CTX *context, const unsigned char *input, size_t len);
void PHP_TIGER192Final(unsigned char digest[24], PHP_TIGER_CTX *context);

void print_hash(unsigned char *hash, int len)
{
    for (int i = 0; i < len; i++)
    {
        printf("%02x", hash[i]);
    }
    printf("\n");
}

int main()
{
    PHP_TIGER_CTX context;
    unsigned char digest[24]; // For Tiger192 (24 bytes = 192 bits)
    const char *test = "abc";

    // Initialize Tiger hash with 3 passes
    PHP_3TIGERInit(&context, NULL);

    // Update with test string
    PHP_TIGERUpdate(&context, (const unsigned char *)test, strlen(test));

    // Finalize the hash
    PHP_TIGER192Final(digest, &context);

    // Print the result
    printf("Tiger hash of '%s': ", test);
    print_hash(digest, 24);

    return 0;
}
