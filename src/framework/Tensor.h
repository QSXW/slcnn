#pragma once

#include <memory>

namespace sl
{
    class Tensor
    {
    public:
        Tensor() = default;

        Tensor(float *data, int x, int y, int z = 1);

        Tensor::Tensor(unsigned char *data, int x, int y, int z = 1);

        Tensor(int x, int y, int z = 1);

    public:
        int width{ 0 };
        int height{ 0 };
        int axis{ 3 };
        int rank{ 0 };
        std::unique_ptr<float> data;

        static Tensor Tensor::TestCase;
    };
}
