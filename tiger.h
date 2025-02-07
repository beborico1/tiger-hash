// tiger.h

#ifndef TIGER_H
#define TIGER_H

#include <stdint.h>
#include <string.h>

// Define L64 macro for 64-bit constants
#define L64(x) x##ULL

// Replace ZEND_SECURE_ZERO with memset
#define ZEND_SECURE_ZERO(p, n) memset((p), 0, (n))

// Tiger context structure
typedef struct
{
    uint64_t state[3];
    uint64_t passed;
    unsigned char buffer[64];
    uint32_t length;
    unsigned int passes : 1;
} TIGER_CTX;

// void TIGERInit(TIGER_CTX *context);
// void TIGERUpdate(TIGER_CTX *context, const unsigned char *input, size_t len);
// void TIGER192Final(unsigned char digest[24], TIGER_CTX *context);

#endif