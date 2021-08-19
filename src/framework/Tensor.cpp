#include "Tensor.h"

#include <memory>
#include <cassert>
#include "Map.h"
#include "Helper.h"

namespace sl
{
    Tensor::Tensor(float *data, int x, int y, int z)
    {
        this->data.reset(data);

        width  = x;
        height = y;
        rank   = z;
    }

    Tensor::Tensor(const Tensor &other) :
        width{ other.width },
        height{ other.height },
        axis{ other.axis },
        rank{ other.rank },
        data{ other.data }
    {
        
    }

    Tensor::~Tensor()
    {

    }

    template <class T>
    Tensor::Tensor(T *data, int x, int y, int z, bool normalize)
    {
        static_assert(std::is_arithmetic_v<T> && "Only support arithmetic type");

        auto size = x * y * z;
        this->data.reset(new float[size]);

        if (normalize)
        {
            Helper::Normalize(data, this->data.get(), size);
        }

        width  = x;
        height = y;
        rank   = z;
    }

    Tensor::Tensor(int x, int y, int z) :
        width{ x },
        height{ y },
        rank{ z }
    {
        auto size = x * y * z;
        data.reset(new float[size]);
        Helper::Clear(data.get(), size);
    }

    void Tensor::Reshape(int x, int y, int z)
    {
        assert((x * y * z) > (width * height * rank) && "Reshape to one dimentions but out of range");

        /*
         * @brief In memory, all data are constinuous. Not need to change.
         * 
         */
        width  = x;
        height = y;
        rank   = z;
    }

    Tensor Tensor::TestCase {
        DataSet::Raw::BOAT, 64, 64, 3
    };
}
