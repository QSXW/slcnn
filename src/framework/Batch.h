#pragma once

#include <memory>
#include <vector>

#include "Tensor.h"

namespace sl
{
    class Batch
    {
    public:
        using iterator = std::vector<Tensor>::iterator;

    public:
        Batch() = default;

        Batch(std::initializer_list<Tensor> &&tensors);

        ~Batch();

        iterator begin()
        {
            return pool.begin();
        }

        iterator end()
        {
            return pool.end();
        }

        Tensor &operator[](size_t index)
        {
            return pool[index];
        }

        auto empty()
        {
            return pool.empty();
        }

        auto size()
        {
            return pool.size();
        }

    public:
        std::vector<Tensor> pool;
    };
}
