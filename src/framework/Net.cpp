#include "Net.h"

#include <cassert>

#include "layer/ConvolutionLayer.h"
#include "layer/BatchNormLayer.h"
#include "layer/ReLuLayer.h"
#include "Timer.h"

namespace sl
{
    std::shared_ptr<Net> Net::CreateNet()
    {
        return std::shared_ptr<Net>();
    }

    Net::Net(const std::initializer_list<Layer::Description> &&descs)
    {
        for (auto &desc : descs)
        {
            if (desc.Type == Layer::Type::Convolution)
            {
                layers.emplace_back(Layer::Create<ConvolutionLayer>(desc));
            }
            if (desc.Type == Layer::Type::BatchNormalize)
            {
                layers.emplace_back(Layer::Create<BatchNormLayer>(desc));
            }
            if (desc.Type == Layer::Type::ReLu)
            {
                layers.emplace_back(Layer::Create<ReLuLayer>(desc));
            }
        }
    }

    Net::~Net()
    {
        
    }

    void Net::Forward()
    {
        assert(!layers.empty() && "There is no available Layer");
        
        {
            Timer timer;
            for (auto &layer : layers)
            {
                layer->Forward();
            }
        }
    }
    void Net::Backward()
    {
    }
};
