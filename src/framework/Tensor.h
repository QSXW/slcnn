#pragma once

#include <memory>
#include <vector>
#include <string>

#include "Helper.h"
#include "sl.h"

namespace sl
{
    class Tensor
    {
    public:
        struct Deleter
        {
            void operator()(float *p)
            {
                sl_aligned_free(static_cast<void*>(p));
            }
        };

        static inline int ALIGN_NUM{ 32 };

        static Tensor Tensor::TestCase; 

    public:
        Tensor() = default;

        Tensor(float *data, int x, int y, int z = 1);

        Tensor(const Tensor &);

        ~Tensor();

        template <class T>
        Tensor(const T *data, int x, int y, int z = 1, bool normalize = true);

        Tensor(int x, int y, int z = 1);

        void Reshape(int x, int y, int z);

        void ExtendRow(const float *data, int x, int y);

        void PushChannel(Tensor &channel, int index);

        Tensor IM2Col(int ksize, int stride = 1, int pad = 0);

        void GEMM(Tensor &a, Tensor &b);

        Tensor MaxPool(int poolSize, int sride = 1, int pad = 0);

        void Display(int channel = -1)
        {
            if (channel < 0)
            {
                channel = depth;
            }
            else
            {
                channel++;
            }
            for (int i = 0; i < channel; i++)
            {
                auto tips = std::string("channel ") + std::to_string(i);
                Helper::DisplayMatrix<float>(data.get() + i * width * height, width, height, width, tips.c_str());
            }
        }

        auto &operator[](size_t index)
        {
            return data.get()[index];
        }

        void ScalarAlphaXPlusY(Tensor &x, float alpha, int INCX = 1, int INCY = 1)
        {
            BasicLinearAlgebraSubprograms::ScalarAlphaXPlusYAVX2(this->data.get(), x.data.get(), alpha, this->size, INCX, INCY);
        }

        void AddBias(float bias)
        {
            Helper::AddBiasAVX2(this->data.get(), bias, this->size);
        }

        void Blend();

    public:
        int width{ 0 };
        int height{ 0 };
        int axis{ 3 };
        int depth{ 0 };
        size_t size{ 0 };
        std::shared_ptr<float> data;
    };

    template <class T>
    Tensor::Tensor(const T *data, int x, int y, int z, bool normalize)
    {
        size = static_cast<size_t>(x) * static_cast<size_t>(y) * static_cast<size_t>(z);
        this->data.reset(sl_aligned_malloc<float>(size, ALIGN_NUM), Deleter());

        if (normalize && std::is_integral_v<T>)
        {
            Helper::Normalize(data, this->data.get(), size);
        }
        else
        {
            memcpy(this->data.get(), data, size * sizeof(float));
        }

        width  = x;
        height = y;
        depth  = z;
    }
}
