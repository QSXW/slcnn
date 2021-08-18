#pragma once

#include <vector>
#include <memory>
#include "layer.h"

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

        void Forward();
        void Backward();

    private:
        LayerList layers;
    };
}
