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

        Batch(const float *kernel, int ksize, int count);

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

        bool empty()
        {
            return pool.empty();
        }

        size_t size()
        {
            return pool.size();
        }

        void push_back(Tensor &value)
        {
            pool.push_back(value);
        }

        void emplace_back(Tensor &&value)
        {
            pool.emplace_back(value);
        }

    public:
        std::vector<Tensor> pool;

        int pad{ 0 };

        int stride{ 1 };
    };
}
