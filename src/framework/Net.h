#pragma once

#include <vector>
#include <memory>
#include <cassert>

#include "Layer.h"
#include "Timer.h"
#include "Tensor.h"

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

        void Train();

    void Forward()
    {
        for (auto &layer : layers)
        {
            layer->Forward();
        }
    }

    void Net::Backward()
    {
        for (int i = layers.size() - 1; i >= 0; --i)
        {
            layers[i]->Forward();
        }
    }

    private:
        LayerList layers;

        Tensor input;
        Tensor output;
    };
}
