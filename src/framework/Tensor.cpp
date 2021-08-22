#include "Tensor.h"

#include <memory>
#include <cassert>
#include <string>

#include "Map.h"
#include "Helper.h"
#include "sl.h"

namespace sl
{
    Tensor::Tensor(float *data, int x, int y, int z) :
        width{ x },
        height{ y },
        depth{ z }
    {
        // Log::Info("Construct tensor: x => {0}\ty => {1}\tz => {2}", x, y, z);
        size = static_cast<size_t>(x) * static_cast<size_t>(y) * static_cast<size_t>(z);
        this->data.reset(sl_aligned_malloc<float>(Helper::SafeBoundary(size), ALIGN_NUM), Deleter());

        memcpy(this->data.get(), data, size * sizeof(float));

        width  = x;
        height = y;
        depth  = z;
    }

    Tensor::Tensor(const Tensor &other) :
        width{ other.width },
        height{ other.height },
        axis{ other.axis },
        depth{ other.depth },
        size{ other.size },
        data{ other.data }
    {
        
    }

    Tensor::~Tensor()
    {
        // Log::Info("Destruct tensor: x => {0}\ty => {1}\tz => {2}", width, height, depth);
    }

    Tensor::Tensor(int x, int y, int z) :
        width{ x },
        height{ y },
        depth{ z }
    {
        // Log::Info("Construct tensor: x => {0}\ty => {1}\tz => {2}", x, y, z);
        size = static_cast<size_t>(x) * static_cast<size_t>(y) * static_cast<size_t>(z);
        this->data.reset(sl_aligned_malloc<float>(Helper::SafeBoundary(size), ALIGN_NUM), Deleter());
        Helper::Clear(data.get(), size);
    }

    void Tensor::Reshape(int x, int y, int z)
    {
        auto newSize = static_cast<size_t>(x) * static_cast<size_t>(y) * static_cast<size_t>(z);
        if (newSize > size)
        {
            size = newSize;
            this->data.reset(sl_aligned_malloc<float>(Helper::SafeBoundary(size), ALIGN_NUM), Deleter());
            Helper::Clear(this->data.get(), size);
        }
        width  = x;
        height = y;
        depth  = z;
    }

    void Tensor::ExtendRow(const float *data, int x, int y)
    {
        Tensor tensor{ x, y, 1 };
        for (int i = 0; i < y; i++)
        {
            memcpy(tensor.data.get() + x * i, data, x * sizeof(float));
        }
        
        *this = std::move(tensor);
    }

    void Tensor::PushChannel(Tensor &channel, int index)
    {
        assert(channel.size < this->size && channel.size == this->width * this->height * channel.depth &&
            "The size of channel must less than the current one and have a identical width and height");
        auto dst = this->data.get() + index * channel.width * channel.height;
        memcpy(dst, channel.data.get(), channel.size * sizeof(float));
    }

    Tensor Tensor::IM2Col(int ksize, int stride, int pad)
    {
        Tensor &src = *this;

        int outHeight = (src.height + 2 * pad - ksize) / stride + 1;
        int outWidth  = (src.width  + 2 * pad - ksize) / stride + 1;

        auto k2 = ksize * ksize;

        Tensor dst{ outWidth * outHeight,  k2, this->depth };

        auto dstptr = dst.data.get();
        auto srcptr = src.data.get();

        int depthCol = src.depth * k2;
        for (int i = 0; i < depthCol; i++)
        {
            int widthOffset  = i % ksize;
            int heightOffset = (i / ksize) % ksize;
            int srcChannel   = i / ksize / ksize;

            for (int y = 0; y < outHeight; y++)
            {
                for (int x = 0; x < outWidth; x++)
                {
                    int col   = widthOffset  + x * stride;
                    int row   = heightOffset + y * stride;
                    int index = (i * outHeight + y) * outWidth + x;
                    *dstptr++ = Helper::IM2ColGetPixel(srcptr,src.width, src.height, src.depth, col, row, srcChannel, pad);
                }
            }
        }

        return dst;
    }

    void Tensor::GEMM(Tensor &a, Tensor &b)
    {
        auto n = width * height;
        BasicLinearAlgebraSubprograms::GEMM(depth, n, a.width, a.data.get(), a.width, b.data.get(), b.width, data.get(), n);
    }

    Tensor Tensor::MaxPool(int poolSize, int sride, int pad)
    {
        Tensor *input = this;
        Tensor output{ input->width / poolSize, input->height / poolSize, input->depth };

        for (int d = 0; d < depth; d++)
        {
            auto src = input->data.get() + d * input->width * input->height;
            auto dst = output.data.get() + d * output.width * output.height;
            for (int i = 0; i < output.height; i++)
            {
                for (int j = 0; j < output.width; j++)
                {
                    float max = static_cast<float>(std::numeric_limits<float>().min());
                    for (int m = 0; m < poolSize; m++)
                    {
                        for (int n = 0; n < poolSize; n++)
                        {
                            auto iw = i * poolSize;
                            auto ih = j * poolSize;
                            auto value = src[std::max(0, iw) * std::min(input->width + ih, input->width)];
                            max = (value > max) ? value : max;
                        }
                    }
                    dst[i * output.width + j] = max;
                }
            }
        }

        return output;
    }

    void Tensor::Blend()
    {
        Tensor blend{ this->width, this->height, 1 };

        memcpy(blend.data.get(), this->data.get(), blend.size * sizeof(float));
        for (int i = 1; i < this->depth; i++)
        {
            auto src = this->data.get() + i * blend.size;
            BasicLinearAlgebraSubprograms::AddAVX2(blend.data.get(), src, blend.size);
        }

        *this = std::move(blend);
    }

    Tensor Tensor::TestCase {
        DataSet::Raw::BOAT, 64, 64, 3
    };
}
