#pragma once

#include <memory>
#include <vector>

namespace sl
{
    class Tensor
    {
    public:
        Tensor() = default;

        Tensor(float *data, int x, int y, int z = 1);

        Tensor(const Tensor &);

        ~Tensor();

        template <class T>
        Tensor(T *data, int x, int y, int z = 1, bool normalize = true);

        Tensor(int x, int y, int z = 1);

        void Reshape(int x, int y, int z);

        auto &operator[](size_t index)
        {
            return data.get()[index];
        }

    public:
        int width{ 0 };
        int height{ 0 };
        int axis{ 3 };
        int depth{ 0 };
        size_t size{ 0 };

        std::shared_ptr<float> data;

        static Tensor Tensor::TestCase;
    };
}
