#pragma once

#include <vector>
#include <memory>
#include <cassert>

#include "Timer.h"
#include "Tensor.h"
#include "Layer.h"

namespace sl
{
    class Net
    {
    public:
        using LayerList = std::vector<std::shared_ptr<Layer>>;
        static std::shared_ptr<Net> CreateNet();

        static float RKernel[];
        static float Gkernel[];
        static float BKernel[];

    public:
        Net(const std::initializer_list<Layer::Description> &&descs);

        ~Net();

        void Set(Tensor::Batch &&batch);

        void Train();

    /*
     * @brief inline function => never extern
     *
     */
    private:
        void Forward();

        void Backward();

    private:
        LayerList layers;

        Tensor::Batch input;
        Tensor::Batch output;
    };
}
