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

using namespace ::sl;

namespace Test
{
    template <class... Args>
    inline constexpr void Fail(const char *s, Args&&... args)
    {
        fprintf(stderr, s, std::forward<Args>(args)...);
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
        Fail("Test failed comparing %.10g with %.10g (abs diff=%.10g with epsilon=%.10g)\n"
             "Index => %d\n\n", a, b, diff, epsilon, index);
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
    inline constexpr bool CompareFloatingPointSequence(float *a, float *b, float epsilon = 0.01f)
    {
        for (size_t i = 0, length = rows * cols; i < length; i++)
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

    bool Image2Columns()
    {
        return true;
    }

    bool ScalarAlphaXPlusY()
    {
        constexpr size_t n = 1024;
        float *x = new float[n * n];

        Test::RandomBuffer<float, n, n>(x);

        float *y1 = new float[n * n];
        float *y2 = new float[n * n];

        {
            Timer timer{ "BasicLinearAlgebraSubprograms::ScalarAlphaXPlusY\t", __LINE__, __func__ };
            BasicLinearAlgebraSubprograms::ScalarAlphaXPlusY(y1, x, 0.1114f, 1024 * 1024);
        }
        {
            Timer timer{ "BasicLinearAlgebraSubprograms::ScalarAlphaXPlusYAVX2\t", __LINE__, __func__ };
            BasicLinearAlgebraSubprograms::ScalarAlphaXPlusYAVX2(y2, x, 0.1114f, 1024 * 1024);
        }

        return Test::CompareFloatingPointSequence<n, n>(y1, y2);
    }
}

namespace Test
{
    static std::map<std::string, std::pair<bool, std::function<bool()>>> Benchmarks = {
        { "ReLu",              { false, Check::ReLu } },
        { "im2col",            { false, Check::Image2Columns } },
        { "ScalarAlphaXPlusY", { false, Check::ScalarAlphaXPlusY } },
        { "Fail",              { false, Check::Fail } }
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
                fprintf(stdout, "\033[1;33mTest: %s\033[0m\t=> [  \033[0;32;32mOK\033[0m  ]\n", name.c_str());
            }
            else
            {
                
                fprintf(stdout, "\033[1;33mTest: %s\033[0m\t=> [ \033[1;31;40mFail\033[0m ]\n", name.c_str());
            }
        }

        fprintf(stdout, "...\n\033[1;36mTest: passed %d/%d\033[0m\n", (sum - fail), sum);
        return 0;
    }
}
