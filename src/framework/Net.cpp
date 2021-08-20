#include "Net.h"

#include <cassert>

#include "layer/ConvolutionalLayer.h"
#include "layer/BatchNormLayer.h"
#include "layer/ActivationLayer.h"
#include "layer/MaxPoolLayer.h"
#include "Timer.h"

namespace sl
{
    float Net::RKernel[] = {
        -0.30, -0.21,  0.07, -0.19,  0.10, -0.01, -0.04, -0.02,  0.08,
         0.14, -0.03,  0.31,  0.14,  0.11,  0.12,  0.21, -0.31, -0.23,
        -0.03,  0.24, -0.05,  0.01, -0.02,  0.07,  0.30,  0.38,  0.19,
         
    };

    float Net::Gkernel[] = {
        -0.20, -0.13, -0.16, -0.09,  0.18,  0.20, -0.22, -0.01, -0.04,
        -0.04, -0.26, -0.12,  0.21, -0.02,  0.12, -0.13,  0.04, -0.14,
    };

    float Net::BKernel[] = {
        0.00, -0.12, -0.08,  0.13, 0.19,  0.10, -0.09,  0.13,  0.06,
        0.32, -0.14,  0.08, -0.04, 0.16,  0.12,  0.14, -0.22, -0.07
    };

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
        }
    }

    Net::~Net()
    {
        
    }

    inline void Net::Forward()
    {
        for (auto &layer : layers)
        {
            layer->Forward(this->input, this->output);
        }
    }

    inline void Net::Backward()
    {
        for (int i = layers.size() - 1; i >= 0; --i)
        {
            layers[i]->Backward(this->input, this->output);
        }
    }

    void Net::Set(Batch &&batch)
    {
        assert(!batch.empty() && "dataset could not be none!");
        input = std::move(batch);
        fprintf(stdout, "Set dataset for Network: width => %d, height => %d, channel => %d, batch size = %lld\n",
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
