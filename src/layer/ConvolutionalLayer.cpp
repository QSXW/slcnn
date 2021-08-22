#include "ConvolutionalLayer.h"

#include <iostream>
#include <cassert>

namespace sl
{
    ConvolutionalLayer::ConvolutionalLayer(const Description &desc) : 
        Layer{ Type::Convolutional },
        description{ desc }
    {
        bias.reserve(desc.bias.size());

        for (auto &b : desc.bias)
        {
            bias.push_back(b.second);
        }

        assert((!description.kernels.empty()) && "Illegel empty kernel");
        ksize = description.kernels[0].width;

        assert((ksize * description.bias.size() == description.kernels.size()) &&
               (ksize >= 1 && ksize <= 19) && "Illegal kernel size");
    }

    void ConvolutionalLayer::Forward(Batch &input, Batch &output)
    {
        auto &kernels = description.kernels;

        assert(!input.empty() && "Cannot input a empty Batch");

        Log::Info("Forwarding: Layer => {0}", Layer::Stringify(type));
        Log::Info("Batches => {0}", input.size());
        Log::Info("Groups => {0}", bias.size());

        auto outWidth  = (input[0].width + 2 * pad - ksize) / stride + 1;
        auto outHeight = (input[0].height + 2 * pad - ksize) / stride + 1;

        // Use for every batches, Only for test
        Tensor kernel{};
        kernel.ExtendRow(DataSet::Kernel::Edge, ksize * ksize, outWidth * outHeight);

        for (int i = 0; i < input.size(); i++)
        {
            Log::Info("Processing Batch => {0} ", i);
            Tensor group{ outWidth, outHeight, static_cast<int>(bias.size()) };

            for (int j = 0; j < bias.size(); j++)
            {
                Log::Info("Processing {0} Group(s)", j);
                Tensor intermediate{ outWidth, outHeight, input[i].depth };
                if (ksize != 1)
                {
                    auto im2col = input[i].IM2Col(ksize);
                    intermediate.GEMM(kernel, im2col);
                }
                else
                {
                    intermediate.GEMM(kernel, input[i]);
                }
                intermediate.Blend();
                if (!Normalized)
                {
                    intermediate.AddBias(bias[i]);
                }
                group.PushChannel(intermediate, j);

                Log::Info("Original Image: width => {0}\theight => {1}\tdepth => {2}",
                    input[i].width, input[i].height, input[i].depth);
                Log::Info("    Conv Image: width => {0}\theight => {1}\tdepth => {2}",
                   intermediate.width, intermediate.height, intermediate.depth);
            }
            Log::Info(" Grouped Image: width => {0}\theight => {1}\tdepth => {2}",
                   group.width, group.height, group.depth);
            output.emplace_back(std::move(group));
        }
    }

    void ConvolutionalLayer::Backward(Batch &input, Batch &output)
    {
        Log::Info("Backwarding: Layer => {0}", Layer::Stringify(type));
    }
}
