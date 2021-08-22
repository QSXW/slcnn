#include "MaxPoolLayer.h"

namespace sl
{
    void MaxPoolLayer::Forward(Batch &input, Batch &output)
    {
        Log::Info("Forwarding: Layer => {0}", Layer::Stringify(type));

        for (int i = 0; i < input.size(); i++)
        {
            Log::Info("Processing Batch => {0} ", i);
            output.emplace_back(std::move(input[i].MaxPool(size)));
            Log::Info("Original Image: width => {0}\theight => {1}\tdepth => {2}",
                input[i].width, input[i].height, input[i].depth);
            Log::Info(" MaxPool Image: width => {0}\theight => {1}\tdepth => {2}",
                output[i].width, output[i].height, output[i].depth);
        }
    }

    void MaxPoolLayer::Backward(Batch &input, Batch &output)
    {
        Log::Info("Backwarding: Layer => {0}", Layer::Stringify(type));
        for (int i = 0; i < input.size(); i++)
        {

        }
    }
}
