#include "Batch.h"

namespace sl
{
    Batch::Batch(const float *kernel, int ksize, int count)
    {
        pool.reserve(count);
        for (int i = 0; i < count; i++)
        {
            pool.emplace_back(Tensor{ kernel, ksize, ksize });
        }
    }

    Batch::Batch(std::initializer_list<Tensor> &&tensors) :
        pool{ std::move(tensors) }
    {

    }

    Batch::~Batch()
    {

    }

    void Batch::Mean()
    {

    }

    void Batch::Variance()
    {

    }

    void Batch::Normalized()
    {

    }

    void Batch::Scale()
    {

    }

}
