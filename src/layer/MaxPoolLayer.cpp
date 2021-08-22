#include "MaxPoolLayer.h"

namespace sl
{
    void MaxPoolLayer::Forward(Batch &input, Batch &output)
    {
        Log::Info("Forwarding: Layer => {0}", Layer::Stringify(type));
        output = std::move(input);

        /*
        int offsetWidth   = input.pad / 2;
        int offsetHheight = input.pad / 2;

        for (int i = 0; i < input.size(); i++)
        {
            Log::Info("Processing Batch => {0} ", i);
            Tensor group{ outWidth, outHeight, static_cast<int>(bias.size()) };

            for (int j = 0; j < bias.size(); j++)
            {
                Log::Info("Processing {0} Group(s)", j);
                Tensor intermediate{ outWidth, outHeight, input[i].depth };
                intermediate.Blend();
                group.PushChannel(intermediate, j);

                Log::Info("Original Image: width => {0}\theight => {1}\tdepth => {2}",
                    input[i].width, input[i].height, input[i].depth);
                Log::Info("    Conv Image: width => {0}\theight => {1}\tdepth => {2}",
                   intermediate.width, intermediate.height, intermediate.depth);
            }
            Log::Info(" Grouped Image: width => {0}\theight => {1}\tdepth => {2}",
                   group.width, group.height, group.depth);
            output.emplace_back(std::move(group));
        }*/
    }

    void MaxPoolLayer::Backward(Batch &input, Batch &output)
    {

    }
}
