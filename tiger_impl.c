#include "tiger.h"
#include "tiger_tables.h"

// Define the table pointers
#define t1 (table)
#define t2 (table + 256)
#define t3 (table + 512)
#define t4 (table + 768)

#if (defined(__APPLE__) || defined(__APPLE_CC__)) && (defined(__BIG_ENDIAN__) || defined(__LITTLE_ENDIAN__))
#if defined(__LITTLE_ENDIAN__)
#undef WORDS_BIGENDIAN
#else
#if defined(__BIG_ENDIAN__)
#define WORDS_BIGENDIAN
#endif
#endif
#endif

/* Tiger round macros */
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

#define pass(a, b, c, mul)                          \
    round(a, b, c, x0, mul)                         \
        round(b, c, a, x1, mul)                     \
            round(c, a, b, x2, mul)                 \
                round(a, b, c, x3, mul)             \
                    round(b, c, a, x4, mul)         \
                        round(c, a, b, x5, mul)     \
                            round(a, b, c, x6, mul) \
                                round(b, c, a, x7, mul)

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

#define feedforward \
    a ^= aa;        \
    b -= bb;        \
    c += cc;

#define compress(passes)                                                            \
    save_abc                                                                        \
    pass(a, b, c, 5)                                                                \
        key_schedule                                                                \
            pass(c, a, b, 7)                                                        \
                key_schedule                                                        \
                    pass(b, c, a, 9) for (pass_no = 0; pass_no < passes; pass_no++) \
    {                                                                               \
        key_schedule                                                                \
            pass(a, b, c, 9)                                                        \
                tmpa = a;                                                           \
        a = c;                                                                      \
        c = b;                                                                      \
        b = tmpa;                                                                   \
    }                                                                               \
    feedforward

#define split_ex(str) \
    x0 = str[0];      \
    x1 = str[1];      \
    x2 = str[2];      \
    x3 = str[3];      \
    x4 = str[4];      \
    x5 = str[5];      \
    x6 = str[6];      \
    x7 = str[7];

#ifdef WORDS_BIGENDIAN
#define split(str)                                                     \
    {                                                                  \
        int i;                                                         \
        uint64_t tmp[8];                                               \
                                                                       \
        for (i = 0; i < 64; ++i)                                       \
        {                                                              \
            ((unsigned char *)tmp)[i ^ 7] = ((unsigned char *)str)[i]; \
        }                                                              \
        split_ex(tmp);                                                 \
    }
#else
#define split split_ex
#endif

#define tiger_compress(passes, str, state)                               \
    {                                                                    \
        register uint64_t a, b, c, tmpa, x0, x1, x2, x3, x4, x5, x6, x7; \
        uint64_t aa, bb, cc;                                             \
        unsigned int pass_no;                                            \
                                                                         \
        a = state[0];                                                    \
        b = state[1];                                                    \
        c = state[2];                                                    \
                                                                         \
        split(str);                                                      \
                                                                         \
        compress(passes);                                                \
                                                                         \
        state[0] = a;                                                    \
        state[1] = b;                                                    \
        state[2] = c;                                                    \
    }

void TIGERInit(TIGER_CTX *context)
{
    context->state[0] = L64(0x0123456789ABCDEF);
    context->state[1] = L64(0xFEDCBA9876543210);
    context->state[2] = L64(0xF096A5B4C3B2E187);
    context->passed = 0;
    context->length = 0;
    context->passes = 3; // Standard Tiger uses 3 passes
    memset(context->buffer, 0, sizeof(context->buffer));
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
            tiger_compress(context->passes, ((const uint64_t *)context->buffer), context->state);
            context->passed += 512;
        }
        for (; i + 64 <= len; i += 64)
        {
            memcpy(context->buffer, &input[i], 64);
            tiger_compress(context->passes, ((const uint64_t *)context->buffer), context->state);
            context->passed += 512;
        }
        memset(&context->buffer[r], 0, 64 - r);
        memcpy(context->buffer, &input[i], r);
        context->length = r;
    }
}

void TIGER192Final(unsigned char digest[24], TIGER_CTX *context)
{
    context->buffer[context->length++] = 0x01;

    if (context->length > 56)
    {
        memset(&context->buffer[context->length], 0, 64 - context->length);
        tiger_compress(context->passes, ((const uint64_t *)context->buffer), context->state);
        context->length = 0;
    }
    memset(&context->buffer[context->length], 0, 56 - context->length);
    uint64_t bits = context->passed + (context->length << 3);
    for (int i = 0; i < 8; i++)
    {
        context->buffer[56 + i] = (bits >> (i * 8)) & 0xFF;
    }
    tiger_compress(context->passes, ((const uint64_t *)context->buffer), context->state);
    for (int i = 0; i < 24; i++)
    {
        digest[i] = (context->state[i / 8] >> (8 * (i % 8))) & 0xFF;
    }
    memset(context, 0, sizeof(*context));
}