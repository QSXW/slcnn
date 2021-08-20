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

    template <class T>
    Tensor::Tensor(T *data, int x, int y, int z, bool normalize)
    {
        size = static_cast<size_t>(x) * static_cast<size_t>(y) * static_cast<size_t>(z);
        this->data.reset(sl_aligned_malloc<float>(size, ALIGN_NUM), Deleter());

        if (normalize && std::is_integral_v<T>)
        {
            Helper::Normalize(data, this->data.get(), size);
        }

        width  = x;
        height = y;
        depth  = z;
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

        /*
         * @brief In memory, all data are constinuous. Not need to change.
         * 
         */
        size = static_cast<size_t>(x) * static_cast<size_t>(y) * static_cast<size_t>(z);
        width  = x;
        height = y;
        depth  = z;
    }

    Tensor Tensor::IM2Col(int ksize, int stride, int pad)
    {
        Tensor &src = *this;
        Tensor dst{ src.width, src.width * ksize, this->depth };

        int outHeight = (src.height + 2 * pad - ksize) / stride + 1;
        int outWidth  = (src.width  + 2 * pad - ksize) / stride + 1;

        auto dstptr = dst.data.get();
        auto srcptr = src.data.get();

        int channelsCol = src.depth * ksize * ksize;
        for (int i = 0; i < channelsCol; i++)
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
                    dstptr[index] = Helper::IM2ColGetPixel(srcptr, src.width, src.height,
                                                           src.depth, col, row, srcChannel, pad);
                }
            }
        }

        return dst;
    }

    Tensor Tensor::TestCase {
        DataSet::Raw::BOAT, 64, 64, 3
    };
}
