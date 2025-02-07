#include <stdio.h>
#include <string.h>
#include "tiger.h"

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
    TIGER_CTX context;
    unsigned char digest[24];
    const char *test = "abc";

    TIGERInit(&context);
    TIGERUpdate(&context, (const unsigned char *)test, strlen(test));
    TIGER192Final(digest, &context);

    printf("Tiger hash of '%s': ", test);
    print_hash(digest, 24);

    return 0;
}