#pragma once

#include <immintrin.h>
#include <algorithm>
#include "Timer.h"

namespace sl
{
namespace Helper
{
    template <class T>
    inline constexpr void Normalize(T *src, float *dst, size_t size)
    {
        float c = 1.0 / (static_cast<T>(~0));
        {
            TIME_SUPERVISED
#if 1
            auto constant = _mm256_set1_ps(c);

            for (int i = 0; i < size; i += 16, dst += 16)
            {
                auto u8h      = _mm_loadu_epi8(src + i);
                auto u8l      = _mm_shuffle_epi32(u8h, 0b1110);
                auto u8tou321 = _mm256_cvtepi8_epi32(u8h);
                auto u8tou322 = _mm256_cvtepi8_epi32(u8l);
                auto ps1      = _mm256_cvtepi32_ps(u8tou321);
                auto ps2      = _mm256_cvtepi32_ps(u8tou322);

                ps1 = _mm256_mul_ps(ps1, constant);
                ps2 = _mm256_mul_ps(ps1, constant);

                _mm256_storeu_ps(dst, ps1);
                _mm256_storeu_ps(dst + 8, ps2);
            }
#else
            for (size_t i = 0; i < size; i++)
            {
                dst[i] = static_cast<float>(src[i]) * c;
            }
#endif
        }
    }

    template <class T>
    inline constexpr void Clear(T *mem, size_t size, T value = 0)
    {
        memset(mem, value, sizeof(T) * size);
    }

    static inline std::string ToLower(std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), 
            [](unsigned char c)
            {
                return std::tolower(c);
            }
        );
        return s;
    }

    static inline size_t SafeBoundary(size_t size, size_t align = 32)
    {
        return size + align - (size & (align - 1));
    }

    static inline void AddBias(float *dst, float bias, size_t size)
    {
        for (size_t i = 0; i < size; i++)
        {
            dst[i] += bias;
        }
    }

    static inline void AddBiasAVX2(float *dst, float bias, size_t size)
    {
        auto BIAS = _mm256_set1_ps(bias);
        for (size_t i = 0; i < size; i += 8, dst += 8)
        {
            auto DST  = _mm256_load_ps(dst);
            auto RES  = _mm256_add_ps(DST, BIAS);
            _mm256_store_ps(dst, RES);
        }
    }
};

namespace BasicLinearAlgebraSubprograms
{
    inline void ScalarAlphaXPlusY(float *y, float *x, float alpha, int size, int INCX = 1, int INCY = 1)
    {
        for (int i = 0; i < size; i++)
        {
            y[i * INCY] += alpha * x[i * INCX];
        }
    }

    inline void ScalarAlphaXPlusYAVX2(float *y, float *x, float alpha, int size, int INCX = 1, int INCY = 1)
    {
        auto A = _mm256_set1_ps(alpha);
        for (int i = 0; i < size; i += 8)
        {
            auto Y = _mm256_loadu_ps(y + i * INCY);
            auto X = _mm256_loadu_ps(x + i * INCX);
            
            X = _mm256_mul_ps(X, A);
            Y = _mm256_add_ps(Y, X);

            _mm256_storeu_ps(y + i * INCY, Y);
        }
    }

    inline void Scale(float *x, float alpha, int size, int INC = 1)
    {
        for (int i = 0; i < size; i++)
        {
            x[i * INC] *= alpha;
        }
    }

    inline void ScaleAVX2(float *x, float alpha, int size, int INC = 1)
    {
        auto A = _mm256_set1_ps(alpha);
        for (int i = 0; i < size; i += 8)
        {
            auto X = _mm256_loadu_ps(x + i * INC);
            X = _mm256_mul_ps(X, A);
            _mm256_storeu_ps(x + i * INC, X);
        }
    }
};
};
