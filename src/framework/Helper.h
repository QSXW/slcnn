#pragma once

#include <immintrin.h>

#include "Timer.h"

namespace sl
{
namespace Helper
{
    template <class T>
    inline constexpr void Normalize(T *src, float *dst, size_t size)
    {
        float c = 1.0 / (static_cast<unsigned char>(~0));
        {
            Timer t{ __FILE__, __LINE__, __func__ };
#if 0
            auto constant = _mm256_set1_ps(c);

            for (int i = 0; i < size; i += 16, dst += 16)
            {
                auto u8h = _mm_loadu_epi8(src + i);
                auto u8l = _mm_shuffle_epi32(u8h, 0b1110);
                auto u8tou321 = _mm256_cvtepi8_epi32(u8h);
                auto u8tou322 = _mm256_cvtepi8_epi32(u8l);
                auto ps1 = _mm256_cvtepi32_ps(u8tou321);
                auto ps2 = _mm256_cvtepi32_ps(u8tou322);

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
};
};
