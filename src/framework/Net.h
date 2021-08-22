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

    public:
        Net(const std::initializer_list<Layer::Description> &&descs);

        ~Net();

        void Set(Batch &&batch);

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

        Batch input;
        Batch output;

        Layer::Type lastType{ Layer::Type::None };
    };
}
