#include "BatchNormLayer.h"

#include <iostream>

namespace sl
{
    void BatchNormLayer::Forward(Batch &input, Batch &output)
    {
        Log::Info("Forwarding: Layer => {0}", Layer::Stringify(type));
        
        output = std::move(input);
        output.Mean();
        output.Variance();
        output.Normalized();
        output.Scale(1.0f);
    }

    void BatchNormLayer::Backward(Batch &input, Batch &output)
    {
        Log::Info("Backwarding: Layer => {0}", Layer::Stringify(type));
    }
}
