#include "Tensor.h"

#include <cassert>
#include "Map.h"
#include "Helper.h"

namespace sl
{
    Tensor::Tensor(float *data, int x, int y, int z)
    {
        auto size = x * y * z;
        this->data.reset(data);
        width  = x;
        height = y;
        rank   = z;
    }

    Tensor::Tensor(unsigned char *data, int x, int y, int z)
    {
        auto size = x * y * z;
        this->data.reset(new float[size]);

        Helper::Normalize(data, this->data.get(), size);

        width  = x;
        height = y;
        rank   = z;
    }

    Tensor::Tensor(int x, int y, int z) :
        width{ x },
        height{ y },
        rank{ z }
    {
        data = std::make_unique<float>(x * y * z);
    }

    Tensor Tensor::TestCase {
        SL_BOAT_RAW_DATA, 64, 64, 3
    };
}
