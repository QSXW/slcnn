#include "MaxPoolLayer.h"

namespace sl
{
    void MaxPoolLayer::Forward(Batch &input, Batch &output)
    {
        Log::Info("Forwarding: Layer => {0}", Layer::Stringify(type));
    }

    void MaxPoolLayer::Backward(Batch &input, Batch &output)
    {

    }
}
