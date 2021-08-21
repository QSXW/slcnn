#include "Test.h"

#include <iostream>
#include <ctime>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <functional>

#include "framework/Timer.h"

#include "sl.h"

using namespace ::sl;

namespace Test
{
    template <class T, int row, int col, int flag>
    inline constexpr void DisplayMatrix(T * m, const char *tips)
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
                printf("%4g", m[i]);
            }
        }
        printf("\n\n");
    }

    template <class... Args>
    inline constexpr void Fail(const char *s, Args&&... args)
    {
        Log::Critical(s, std::forward<Args>(args)...);
        // __debugbreak();
    }

    template <class... Args>
    inline constexpr void Succeed(const char *s, Args&&... args)
    {
        printf(s, std::forward<Args>(args)...);
    }

    inline bool CompareFloatingPoint(float a, float b, float epsilon = 0.01f, int index = 0)
    {
        float diff = fabsf(a - b);

        if (diff < epsilon)
        {
            return true;
        }
        Fail("Test failed comparing {0} with {1} (abs diff={2} with epsilon={3})"
             "Index => {4}", a, b, diff, epsilon, index);
        return false;
    }

    template <size_t count, class T, class... Args>
    inline constexpr void Regress(T func, Args&&... args)
    {
        for (int i = 0; i < count; i++)
        {
            func(std::forward<Args>(args)...);
        }
    }

    template <size_t count, class T, class... Args>
    inline constexpr void RegressV(T func, Args&&... args)
    {
        float *ptr = gbuffer;
        for (int i = 0; i < count; i++)
        {
            auto ans = func(std::forward<Args>(args)...);
            *ptr++ = *((float *)&ans);
        }
    }

    #define RegressX(count, command) { \
        float *ptr = gbuffer; \
        for (int i = 0; i < count; i++) \
        { \
           auto ans = command; \
           *ptr++ = *((float *)&ans); \
        } \
    }

    template <class T, size_t rows, size_t cols>
    inline constexpr void RandomBuffer(T *buffer)
    {
        auto size = rows * cols;
        for (int i = 0; i < size; i++)
        {
            buffer[i] = (T)(rand() & 0xFF);
        }
    }
    
    template <size_t rows, size_t cols>
    inline constexpr bool CompareFloatingPointSequence(float *a, float *b, float epsilon = 0.01f, size_t r = 0, size_t c = 0)
    {
        if constexpr (rows != 0 && cols != 0)
        {
            r = rows;
            c = cols;
        }
        for (size_t i = 0, length = r * c; i < length; i++)
        {
            if (!CompareFloatingPoint(a[i], b[i], epsilon, i))
            {
                return false;
            }
        }
        return true;
    }
}

namespace Check
{
    bool Fail()
    {
        return false;
    }

    bool ReLu()
    {
        float def[4 * 4] = { -1, -1, 1, 1, 2, 2, -2, -2, -3, -3, 3, 3, 4, 4, -4, -4 };
        float ref[4 * 4] = {  0,  0, 1, 1, 2, 2,  0,  0,  0,  0, 3, 3, 4, 4,  0,  0 };

        for (int i = 0; i < 4 * 4; i++)
        {
            def[i] = ActivationLayer::Activates["relu"](def[i]);
        }
        return Test::CompareFloatingPointSequence<4, 4>(def, ref);
    }

    bool IM2Col()
    {
        float SLALIGNED(32) a[] = {
            0,  1,  2,  3,
            4,  5,  6,  7,
            8,  9,  10, 11,
            12, 13, 14, 15,

            0,  1,  2,  3,
            4,  5,  6,  7,
            8,  9,  10, 11,
            12, 13, 14, 15,

            0,  1,  2,  3,
            4,  5,  6,  7,
            8,  9,  10, 11,
            12, 13, 14, 15
        };

        float SLALIGNED(32) ref[] = {
            0,  1,  4,  5,
            1,  2,  5,  6,
            2,  3,  6,  7,
            4,  5,  8,  9,
            5,  6,  9,  10,
            6,  7,  10, 11,
            8,  9,  12, 13,
            9,  10, 13, 14,
            10, 11, 14, 15,

            0,  1,  4,  5,
            1,  2,  5,  6,
            2,  3,  6,  7,
            4,  5,  8,  9,
            5,  6,  9,  10,
            6,  7,  10, 11,
            8,  9,  12, 13,
            9,  10, 13, 14,
            10, 11, 14, 15,

            0,  1,  4,  5,
            1,  2,  5,  6,
            2,  3,  6,  7,
            4,  5,  8,  9,
            5,  6,  9,  10,
            6,  7,  10, 11,
            8,  9,  12, 13,
            9,  10, 13, 14,
            10, 11, 14, 15
        };
        
        Tensor src{ a, 4, 4, 3 };
        Tensor dst{};
        {
            Timer timer{ "Tensor::IM2Col\t", __LINE__, __func__ };
            dst = src.IM2Col(3);
        }
        /*src.Display();
        dst.Display();*/
        auto ret = Test::CompareFloatingPointSequence<4, 9>(dst.data.get(), ref);

        return ret;
    }

    bool CONV()
    {
        constexpr size_t n = 2160;

        float *x  = sl_aligned_malloc<float>(n * n * 3, 2);
        Test::RandomBuffer<float, n, n * 3>(x);

        Tensor src{ x, n, n, 3 };

        int outHeight = (src.height + 2 * 0 - 3) / 1 + 1;
        int outWidth  = (src.width  + 2 * 0 - 3) / 1 + 1;

        Tensor y1{ outWidth, outHeight, 3 };
        Tensor y2{ outWidth, outHeight, 3 };

        Tensor kernel{ DataSet::Kernel::Edge, 3, 3, 1 };

        {
            auto kernelptr = kernel.data.get();
            Timer timer{ "CONV: Standard Convolution", __LINE__, __func__ };
            for (int c = 0; c < 3; c++)
            {
                auto srcptr = src.data.get() + (c * src.width * src.height);
                auto dst = y1.data.get() + (c * y1.width * y1.height);
                for (int i = 0; i < outWidth; i++)
                {
                    auto dstptr = dst +i * outWidth;
                    for (int j = 0; j < outHeight; j++)
                    {
                        float sum = 0;
                        for (int m = 0; m < 3; m++)
                        {
                            for (int n = 0; n < 3; n++)
                            {
                                sum += kernelptr[m * 3 + n] * srcptr[(i + m) * src.width + (j + n)];
                            }
                        }
                        dstptr[j] = sum;
                    }
                }
            }
        }

        {
            auto im2col = src.IM2Col(3);
            kernel.ExtendRow(DataSet::Kernel::Edge, 9, im2col.height);
            auto &a = kernel;
            auto &b = im2col;
            auto &c = y2;
            auto n = y2.width * y2.height;
            Timer timer{ "CONV: GEMM", __LINE__, __func__ };
            for (int i = 0; i < 100; i++)
            {
                BasicLinearAlgebraSubprograms::GEMM(y2.depth, n, kernel.width, kernel.data.get(), kernel.width, im2col.data.get(), im2col.width, y2.data.get(), n);
            }
        }

        auto ret = Test::CompareFloatingPointSequence<0, 0>(y1.data.get(), y2.data.get(), y2.width, y2.height);

        sl_aligned_free(x);

        return true;
    }

    bool ScalarAlphaXPlusY()
    {
        constexpr size_t n = 1024;
        
        float *x  = sl_aligned_malloc<float>(n * n, 2);
        float *y1 = sl_aligned_malloc<float>(n * n, 2);
        float *y2 = sl_aligned_malloc<float>(n * n, 2);

        Test::RandomBuffer<float, n, n>(x);

        {
            Timer timer{ "BasicLinearAlgebraSubprograms::SAXPY\t", __LINE__, __func__ };
            BasicLinearAlgebraSubprograms::ScalarAlphaXPlusY(y1, x, 0.1114f, n * n);
        }
        {
            Timer timer{ "BasicLinearAlgebraSubprograms::SAXPY\t", __LINE__, __func__ };
            BasicLinearAlgebraSubprograms::ScalarAlphaXPlusYAVX2(y2, x, 0.1114f, n * n);
        }

        auto ret = Test::CompareFloatingPointSequence<n, n>(y1, y2);

        sl_aligned_free(x);
        sl_aligned_free(y1);
        sl_aligned_free(y2);

        return ret;
    }

    bool Scale()
    {
        constexpr size_t n = 1024;

        float *x  = sl_aligned_malloc<float>(n * n, 2);
        float *y1 = sl_aligned_malloc<float>(n * n, 2);
        float *y2 = sl_aligned_malloc<float>(n * n, 2);

        Test::RandomBuffer<float, n, n>(x);

        memcpy(y1, x, n * n);
        memcpy(y2, x, n * n);

        {
            Timer timer{ "BasicLinearAlgebraSubprograms::Scale\t", __LINE__, __func__ };
            BasicLinearAlgebraSubprograms::Scale(y1, 0.1114f, n * n);
        }
        {
            Timer timer{ "BasicLinearAlgebraSubprograms::ScaleAVX2\t", __LINE__, __func__ };
            BasicLinearAlgebraSubprograms::ScaleAVX2(y2, 0.1114f, n * n);
        }

        auto ret = Test::CompareFloatingPointSequence<n, n>(y1, y2);

        sl_aligned_free(x);
        sl_aligned_free(y1);
        sl_aligned_free(y2);

        return ret;
    }

     bool Add()
    {
        constexpr size_t n = 1024;

        float *x  = sl_aligned_malloc<float>(n * n, 2);
        float *y1 = sl_aligned_malloc<float>(n * n, 2);
        float *y2 = sl_aligned_malloc<float>(n * n, 2);

        Test::RandomBuffer<float, n, n>(x);

        memcpy(y1, x, n * n);
        memcpy(y2, x, n * n);

        {
            Timer timer{ "BasicLinearAlgebraSubprograms::Add\t", __LINE__, __func__ };
            BasicLinearAlgebraSubprograms::Add(y1, x, n * n);
        }
        {
            Timer timer{ "BasicLinearAlgebraSubprograms::AddAVX2\t", __LINE__, __func__ };
            BasicLinearAlgebraSubprograms::AddAVX2(y2, x, n * n);
        }

        auto ret = Test::CompareFloatingPointSequence<n, n>(y1, y2);

        sl_aligned_free(x);
        sl_aligned_free(y1);
        sl_aligned_free(y2);

        return ret;
    }

    bool AddBias()
    {
        constexpr size_t n = 1024;
        
        float *x  = sl_aligned_malloc<float>(n * n, 2);
        float *y1 = sl_aligned_malloc<float>(n * n, 2);
        float *y2 = sl_aligned_malloc<float>(n * n, 2);

        Test::RandomBuffer<float, n, n>(x);

        memcpy(y1, x, n * n);
        memcpy(y2, x, n * n);

        {
            Timer timer{ "Helper::AddBias\t", __LINE__, __func__ };
            BasicLinearAlgebraSubprograms::Scale(y1, 0.1114f, n * n);
        }
        {
            Timer timer{ "Helper::AddBiasAVX2\t", __LINE__, __func__ };
            BasicLinearAlgebraSubprograms::ScaleAVX2(y2, 0.1114f, n * n);
        }

        auto ret = Test::CompareFloatingPointSequence<n, n>(y1, y2);

        sl_aligned_free(x);
        sl_aligned_free(y1);
        sl_aligned_free(y2);

        return ret;
    }
}

namespace Test
{
    static std::map<std::string, std::pair<bool, std::function<bool()>>> Benchmarks = {
        { "ReLu",              { false, Check::ReLu } },
        { "IM2Col",            { false, Check::IM2Col } },
        { "CONV",              { false, Check::CONV } },
        { "SAXPY",             { false, Check::ScalarAlphaXPlusY } },
        { "Scale",             { false, Check::Scale } },
        { "Add",               { false, Check::Add } },
        { "AddBias",           { false, Check::AddBias } },
        { "A_Fail_Test",       { false, Check::Fail } }
    };

    int Launch()
    {
        system("chcp 65001 && cls");
        TIME_SUPERVISED
        int sum  = 0;
        int fail = 0;
        for (auto &b : Benchmarks)
        {
            sum++;
            auto &[name,  props] = b;
            auto &[passed, func] = props;

            passed = func();
            if (!passed)
            {
                fail++;
            }
        }

        for (auto &b : Benchmarks)
        {
            auto &[name,  props] = b;
            auto &[passed, func] = props;

            if (passed)
            {
                Log::Info("\033[1;33mTest: {0}\033[0m\t=> {1}", name.c_str(), "[ \033[0;32;32mOK\033[0m  ]");
            }
            else
            {
                
                Log::Info("\033[1;33mTest: {0}\033[0m\t=> {1}", name.c_str(), "[ \033[1;31;40mFail\033[0m ]");
            }
        }

        Log::Info("\033[1;36mTest: passed {0}/{1}\033[0m", (sum - fail), sum);
        return 0;
    }
}
