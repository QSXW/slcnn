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
        assert((static_cast<size_t>(x) * static_cast<size_t>(y) * static_cast<size_t>(z)) > size &&
                "Reshape to one dimentions but out of range");
        size = static_cast<size_t>(x) * static_cast<size_t>(y) * static_cast<size_t>(z);
        width  = x;
        height = y;
        depth  = z;
    }

    Tensor Tensor::IM2Col(int ksize, int stride, int pad)
    {
        Tensor &src = *this;

        int outHeight = (src.height + 2 * pad - ksize) / stride + 1;
        int outWidth  = (src.width  + 2 * pad - ksize) / stride + 1;

        Tensor dst{ outWidth * outHeight, ksize * ksize, this->depth };

        auto dstptr = dst.data.get();
        auto srcptr = src.data.get();

        int depthCol = src.depth * ksize * ksize;
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
                    dstptr[index] = Helper::IM2ColGetPixel(srcptr,src.width, src.height, src.depth, col, row, srcChannel, pad);
                }
            }
        }

        return dst;
    }

    void Tensor::GEMM(Tensor &kernel)
    {
        
    }

    Tensor Tensor::TestCase {
        DataSet::Raw::BOAT, 64, 64, 3
    };
}
