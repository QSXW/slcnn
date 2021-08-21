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
        size = static_cast<size_t>(x) * static_cast<size_t>(y) * static_cast<size_t>(z);
        this->data.reset(sl_aligned_malloc<float>(size, ALIGN_NUM), Deleter());

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

    }

    Tensor::Tensor(int x, int y, int z) :
        width{ x },
        height{ y },
        depth{ z }
    {
        size = static_cast<size_t>(x) * static_cast<size_t>(y) * static_cast<size_t>(z);
        this->data.reset(sl_aligned_malloc<float>(size, ALIGN_NUM), Deleter());
        Helper::Clear(data.get(), size);
    }

    void Tensor::Reshape(int x, int y, int z)
    {
        auto newSize = static_cast<size_t>(x) * static_cast<size_t>(y) * static_cast<size_t>(z);
        if (newSize > size)
        {
            size = newSize;
            this->data.reset(sl_aligned_malloc<float>(size, ALIGN_NUM), Deleter());
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
        /*assert(channel.size < this->size && channel.size == this->width * this->height * channel.depth &&
            "The size of channel must less than the current one and have a identical width and height");*/
        auto dst = this->data.get() + index * this->width * this->height;
        memcpy(dst, channel.data.get(), channel.size);
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
        auto offsetC = this->width * this->height;
        auto offsetB = b.width * b.height;
        for (int i = 0; i < this->depth; i++)
        {
            // BasicLinearAlgebraSubprograms::GEMM(0, 0, a.height, b.width, a.width, 1, a.data.get(), a.width, b.data.get() + offsetB, b.width, 1, this->data.get() + offsetC, b.width);
        }
    }

    void Tensor::Blend()
    {
        Tensor blend{ this->width, this->height, 1 };

        memcpy(blend.data.get(), this->data.get(), blend.size * sizeof(float));
        for (int i = 1; i < this->depth; i++)
        {
            auto pass = this->data.get() + i * blend.size;
            BasicLinearAlgebraSubprograms::AddAVX2(blend.data.get(), pass, blend.size);
        }

        *this = std::move(blend);
    }

    Tensor Tensor::TestCase {
        DataSet::Raw::BOAT, 64, 64, 3
    };
}
