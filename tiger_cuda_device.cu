#include "tiger_cuda.h"

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

__device__ void TIGERUpdate_gpu(GPU_TIGER_CTX *context, const unsigned char *input, size_t len)
{
    if (context->length + len < 64)
    {
        // If we don't have enough data for a full block
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
    // Add padding
    context->buffer[context->length++] = 0x01;

    // Zero fill rest of the block
    for (size_t i = context->length; i < 64; i++)
    {
        context->buffer[i] = 0;
    }

    // If we don't have enough space for the length
    if (context->length > 56)
    {
        tiger_compress_gpu((uint64_t *)context->buffer, context->state);
        for (int i = 0; i < 64; i++)
        {
            context->buffer[i] = 0;
        }
    }

    // Append length in bits
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

#define ROUND(a, b, c, x, mul)                                                     \
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

        // Load state
        a = state[0];
        b = state[1];
        c = state[2];

        // Load message block
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

        // Round 1
        ROUND(a, b, c, x0, 5)
        ROUND(b, c, a, x1, 5)
        ROUND(c, a, b, x2, 5)
        ROUND(a, b, c, x3, 5)
        ROUND(b, c, a, x4, 5)
        ROUND(c, a, b, x5, 5)
        ROUND(a, b, c, x6, 5)
        ROUND(b, c, a, x7, 5)

        // Key schedule 1
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

        // Key schedule 2
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

        // Feedforward
        a ^= aa;
        b -= bb;
        c += cc;

        // Save state
        state[0] = a;
        state[1] = b;
        state[2] = c;
    }

    a ^= aa;
    b -= bb;
    c += cc;

    state[0] = a;
    state[1] = b;
    state[2] = c;
}