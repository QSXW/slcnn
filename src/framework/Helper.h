#pragma once

#include <immintrin.h>
#include <algorithm>
#include "Timer.h"

namespace sl
{
namespace Helper
{
    template <class T>
    inline constexpr void DisplayMatrix(T * m, int row, int col, int flag, const char *tips)
    {
        printf("%s\n", tips);
        for (int i = 0; i < row * col; i++)
        {
            if (i && i % flag == 0)
            {
                putchar(10);
            }
            if constexpr (std::is_integral_v<T>)
            {
                printf("%4d", m[i]);
            }
            if constexpr (std::is_floating_point_v<T>)
            {
                printf("%g\t", m[i]);
            }
        }
        printf("\n\n");
    }

    template <class T>
    inline constexpr void Normalize(T *src, float *dst, size_t size)
    {
        float c = 1.0 / (static_cast<T>(~0));
        {
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

    static float IM2ColGetPixel(float *im, int width, int height, int channels, int col, int row, int channel, int pad)
    {
        col -= pad;
        row -= pad;

        if (row < 0 || col < 0 || row >= height || col >= width)
        {
            return 0;
        }
        return im[col + width * (row + height * channel)];
    }

    static void IM2Col(float *src, float *dst, int channels, int width,  int height, int ksize, int stride = 1, int pad = 0) 
    {
        int outHeight = (height + 2 * pad - ksize) / stride + 1;
        int outWidth  = (width + 2 * pad - ksize) / stride + 1;

        int channelsCol = channels * ksize * ksize;
        for (int i = 0; i < channelsCol; i++)
        {
            int widthOffset  = i % ksize;
            int heightOffset = (i / ksize) % ksize;
            int srcChannel   = i / ksize / ksize;

            for (int y = 0; y < outHeight; y++)
            {
                for (int x = 0; x < outWidth; x++)
                {
                    int srcX = widthOffset  + x * stride;
                    int srcY = heightOffset + y * stride;
                    dst[(i * outHeight + y) * outWidth + x] = IM2ColGetPixel(src, width, height, channels, srcX, srcY, srcChannel, pad);
                }
            }
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

    inline void Add(float *y, float *x, int size)
    {
        for (int i = 0; i < size; i++)
        {
            y[i] += x[i];
        }
    }

    inline void AddAVX2(float *y, float *x, int size)
    {
        for (int i = 0; i < size; i += 8)
        {
            auto Y   = _mm256_loadu_ps(y + i);
            auto X   = _mm256_loadu_ps(x + i);
            auto RES = _mm256_add_ps(Y, X);

            _mm256_storeu_ps(y + i, RES);
        }
    }

    static inline void GEMM(int M, int N, int K, float *A, int lda,  float *B, int ldb, float *C, int ldc)
    {
        #pragma omp parallel for
        for (int i = 0; i < M; i++)
        {
            int j = 0;
            int k = 0;
            for ( ; k < K; k++)
            {
                for (j = 0; j < N; j++)
                {
                    C[i * ldc + j] += A[i * lda + k] * B[k * ldb + j];
                }
            }
            B += k * j;
        }
    }

    static inline float Convolution(const float *left, const float *right, int size)
    {
        float sum = 0;
        for (int i = 0; i < size; i++)
        {
            sum += left[i] * right[i];
        }

        return sum;
    }

    static inline void PATCH(float *dst, float *patches, int patchSize, const float *kernel, int ksize)
    {
        #pragma omp parallel for
        for (int i = 0; i < patchSize; i++)
        {
            *dst++ = Convolution(patches + i * ksize, kernel, ksize);
        }
    }

    static inline void Mean(float *x, int depth, int spatial, float *mean)
    {
        float scale = 1.0f / spatial;
        #pragma omp parallel for
        for(int i = 0; i < depth; i++)
        {
            mean[i] = 0;
            for(int j = 0; j < spatial; j++)
            {
                mean[i] += x[i * spatial + j];
            }
            mean[i] *= scale;
        }
    }

    static inline void Variance(float *x, int depth, int spatial, float *mean, float *variance)
    {
        float scale = 1.0f / (spatial - 1);
        for(int i = 0; i < depth; i++)
        {
            variance[i] = 0;
            for(int j = 0; j < spatial; j++)
            {
                variance[i] += pow((x[i * spatial + j] - mean[i]), 2);
            }
            variance[i] *= scale;
        }
    }

    static inline void Normalize(float *x, int depth, int spatial, float *mean, float *variance)
    {
        for(int i = 0; i < depth; i++)
        {
            for(int j = 0; j < spatial; j++)
            {
                int index = i * spatial + j;
                x[index] = (x[index] - mean[i]) / (sqrt(variance[i]) + .000001f);
            }
        }
    }
};
};
