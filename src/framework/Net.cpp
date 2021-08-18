#include "Net.h"

#include <cassert>

#include "layer/BatchNormLayer.h"

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
            if (desc.Type == Layer::Type::BatchNormalize)
            {
                layerStack.emplace_back(Layer::Create<BatchNormLayer>(desc));
            }
        }
    }

    Net::~Net()
    {
        
    }

    void Net::Train()
    {
        assert(!layerStack.empty() && "There is no available Layer");
    }
};
