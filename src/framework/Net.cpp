#include "Net.h"

#include <cassert>

#include "layer/ConvolutionalLayer.h"
#include "layer/BatchNormLayer.h"
#include "layer/ActivationLayer.h"
#include "layer/MaxPoolLayer.h"
#include "layer/SoftmaxLayer.h"

#include "Timer.h"
#include "Log.h"

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
            if (desc.Type == Layer::Type::Convolutional)
            {
                layers.emplace_back(Layer::Create<ConvolutionalLayer>(desc));
            }
            if (desc.Type == Layer::Type::BatchNormalize)
            {
                auto back = dynamic_cast<ConvolutionalLayer *>(layers.back().get());
                if (back->type == Layer::Type::Convolutional)
                {
                    back->Normalized = true;
                }
                layers.emplace_back(Layer::Create<BatchNormLayer>(desc));
            }
            if (desc.Type == Layer::Type::Activation)
            {
                layers.emplace_back(Layer::Create<ActivationLayer>(desc));
            }
            if (desc.Type == Layer::Type::MaxPool)
            {
                layers.emplace_back(Layer::Create<MaxPoolLayer>(desc));
            }
            if (desc.Type == Layer::Type::Softmax)
            {
                layers.emplace_back(Layer::Create<SoftmaxLayer>(desc));
            }
            lastType = desc.Type;
            Log::Info("Push Layer to Net => {0}", Layer::Stringify(desc.Type));
        }
        Log::Info("Net: Layer size => {0}", layers.size());
    }

    Net::~Net()
    {
        
    }

    inline void Net::Forward()
    {
        for (auto &layer : layers)
        {
            layer->Forward(input, output);
            output[0].Display(0);
            input = std::move(output);
            output = Batch{};
        }
    }

    inline void Net::Backward()
    {
        for (int i = layers.size() - 1; i >= 0; i--)
        {
            layers[i]->Backward(input, output);
            input = std::move(output);
            output = Batch{};
        }
    }

    void Net::Set(Batch &&batch)
    {
        assert(!batch.empty() && "dataset could not be none!");
        input = std::move(batch);
        Log::Info("Set dataset for Network: width => {0}, height => {1}, channel => {2}, batch size = {3}",
                input[0].width, input[0].height, input[0].depth, input.size());
    }

    void Net::Train()
    {
        TIME_SUPERVISED
        assert(!layers.empty() && "There is no available Layer");
        Forward();
        Backward();
    }
};
