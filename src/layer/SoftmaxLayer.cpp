#include "SoftmaxLayer.h"

#include "framework/Helper.h"

namespace sl
{
    void SoftmaxLayer::Forward(Batch &input, Batch &output)
    {
        Log::Info("Forwarding: Layer => {0}", Layer::Stringify(type));
    }

    void SoftmaxLayer::Backward(Batch &input, Batch &output)
    {
        for (auto in = input.begin(), out = output.end(); in != input.end() && out != output.end(); in++, out++)
        {
            out->ScalarAlphaXPlusY(*in, 1.0f);
        }
    }
}
