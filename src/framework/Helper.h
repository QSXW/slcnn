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

    static std::string ToLower(std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), 
            [](unsigned char c)
            {
                return std::tolower(c);
            }
        );
        return s;
    }
};
};
