#include "Tensor.h"

#include <memory>
#include <cassert>
#include "Map.h"
#include "Helper.h"

#include "sl.h"

namespace sl
{
    Tensor::Tensor(float *data, int x, int y, int z)
    {
        this->data.reset(data);

        size = static_cast<size_t>(x) * static_cast<size_t>(y) * static_cast<size_t>(z);
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

    Tensor Tensor::TestCase {
        DataSet::Raw::BOAT, 64, 64, 3
    };
}
