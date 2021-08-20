#include "Batch.h"

namespace sl
{
    Batch::Batch(std::initializer_list<Tensor> &&tensors) :
        pool{ std::move(tensors) }
    {

    }

    Batch::~Batch()
    {

    }
}
