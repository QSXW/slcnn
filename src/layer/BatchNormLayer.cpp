#include "BatchNormLayer.h"

#include <iostream>

namespace sl
{
    void BatchNormLayer::Forward(Batch &input, Batch &output)
    {
        Log::Info("Forwarding: Layer => {0}", Layer::Stringify(type));
    }

    void BatchNormLayer::Backward(Batch &input, Batch &output)
    {

    }
}
