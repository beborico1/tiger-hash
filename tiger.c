#include "tiger.h"

// Tiger tables from the header file
extern const uint64_t table[4 * 256];

// Internal macros for Tiger hash computation
#define save_abc \
    aa = a;      \
    bb = b;      \
    cc = c;

#define round(a, b, c, x, mul)                                          \
    c ^= x;                                                             \
    a -= t1[(unsigned char)(c)] ^                                       \
         t2[(unsigned char)(((uint32_t)(c)) >> (2 * 8))] ^              \
         t3[(unsigned char)((c) >> (4 * 8))] ^                          \
         t4[(unsigned char)(((uint32_t)((c) >> (4 * 8))) >> (2 * 8))];  \
    b += t4[(unsigned char)(((uint32_t)(c)) >> (1 * 8))] ^              \
         t3[(unsigned char)(((uint32_t)(c)) >> (3 * 8))] ^              \
         t2[(unsigned char)(((uint32_t)((c) >> (4 * 8))) >> (1 * 8))] ^ \
         t1[(unsigned char)(((uint32_t)((c) >> (4 * 8))) >> (3 * 8))];  \
    b *= mul;

#define key_schedule                    \
    x0 -= x7 ^ L64(0xA5A5A5A5A5A5A5A5); \
    x1 ^= x0;                           \
    x2 += x1;                           \
    x3 -= x2 ^ ((~x1) << 19);           \
    x4 ^= x3;                           \
    x5 += x4;                           \
    x6 -= x5 ^ ((~x4) >> 23);           \
    x7 ^= x6;                           \
    x0 += x7;                           \
    x1 -= x0 ^ ((~x7) << 19);           \
    x2 ^= x1;                           \
    x3 += x2;                           \
    x4 -= x3 ^ ((~x2) >> 23);           \
    x5 ^= x4;                           \
    x6 += x5;                           \
    x7 -= x6 ^ L64(0x0123456789ABCDEF);

#define t1 (table)
#define t2 (table + 256)
#define t3 (table + 256 * 2)
#define t4 (table + 256 * 3)

void TIGERInit(TIGER_CTX *context)
{
    context->state[0] = L64(0x0123456789ABCDEF);
    context->state[1] = L64(0xFEDCBA9876543210);
    context->state[2] = L64(0xF096A5B4C3B2E187);
    context->passed = 0;
    context->length = 0;
    context->passes = 0;
    memset(context->buffer, 0, sizeof(context->buffer));
}

static void tiger_compress(const uint64_t *str, uint64_t state[3], int passes)
{
    uint64_t a, b, c, aa, bb, cc, x0, x1, x2, x3, x4, x5, x6, x7;
    int pass_no;

    a = state[0];
    b = state[1];
    c = state[2];

    // Load data block
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

    // Pass 1
    round(a, b, c, x0, 5);
    round(b, c, a, x1, 5);
    round(c, a, b, x2, 5);
    round(a, b, c, x3, 5);
    round(b, c, a, x4, 5);
    round(c, a, b, x5, 5);
    round(a, b, c, x6, 5);
    round(b, c, a, x7, 5);

    key_schedule;

    // Pass 2
    round(c, a, b, x0, 7);
    round(a, b, c, x1, 7);
    round(b, c, a, x2, 7);
    round(c, a, b, x3, 7);
    round(a, b, c, x4, 7);
    round(b, c, a, x5, 7);
    round(c, a, b, x6, 7);
    round(a, b, c, x7, 7);

    key_schedule;

    // Pass 3
    round(b, c, a, x0, 9);
    round(c, a, b, x1, 9);
    round(a, b, c, x2, 9);
    round(b, c, a, x3, 9);
    round(c, a, b, x4, 9);
    round(a, b, c, x5, 9);
    round(b, c, a, x6, 9);
    round(c, a, b, x7, 9);

    for (pass_no = 3; pass_no < passes; pass_no++)
    {
        key_schedule;

        round(a, b, c, x0, 9);
        round(b, c, a, x1, 9);
        round(c, a, b, x2, 9);
        round(a, b, c, x3, 9);
        round(b, c, a, x4, 9);
        round(c, a, b, x5, 9);
        round(a, b, c, x6, 9);
        round(b, c, a, x7, 9);

        // Rotate registers
        uint64_t tmp = a;
        a = c;
        c = b;
        b = tmp;
    }

    // Feedforward
    a ^= aa;
    b -= bb;
    c += cc;

    state[0] = a;
    state[1] = b;
    state[2] = c;
}

void TIGERUpdate(TIGER_CTX *context, const unsigned char *input, size_t len)
{
    if (context->length + len < 64)
    {
        memcpy(&context->buffer[context->length], input, len);
        context->length += len;
    }
    else
    {
        size_t i = 0, r = (context->length + len) % 64;

        if (context->length)
        {
            i = 64 - context->length;
            memcpy(&context->buffer[context->length], input, i);
            tiger_compress((const uint64_t *)context->buffer, context->state, 3);
            context->passed += 512;
        }

        for (; i + 64 <= len; i += 64)
        {
            memcpy(context->buffer, &input[i], 64);
            tiger_compress((const uint64_t *)context->buffer, context->state, 3);
            context->passed += 512;
        }

        memset(&context->buffer[r], 0, 64 - r);
        memcpy(context->buffer, &input[i], r);
        context->length = r;
    }
}

void TIGER192Final(unsigned char digest[24], TIGER_CTX *context)
{
    // Add padding
    context->buffer[context->length++] = 0x01;

    // If length is > 56 bytes, process this block and create a new one
    if (context->length > 56)
    {
        memset(&context->buffer[context->length], 0, 64 - context->length);
        tiger_compress((const uint64_t *)context->buffer, context->state, 3);
        context->length = 0;
    }

    // Pad with zeros up to 56 bytes
    memset(&context->buffer[context->length], 0, 56 - context->length);

    // Append length in bits
    uint64_t bits = context->passed + (context->length << 3);
    for (int i = 0; i < 8; i++)
    {
        context->buffer[56 + i] = (bits >> (i * 8)) & 0xFF;
    }

    // Process final block
    tiger_compress((const uint64_t *)context->buffer, context->state, 3);

    // Output hash
    for (int i = 0; i < 24; i++)
    {
        digest[i] = (context->state[i / 8] >> (8 * (i % 8))) & 0xFF;
    }

    // Clear sensitive data
    memset(context, 0, sizeof(*context));
}