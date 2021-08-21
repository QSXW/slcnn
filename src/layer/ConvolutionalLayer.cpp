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

        Log::Info("Forwarding: Layer => {0}", Layer::Stringify(type));
        Log::Info("Batches => {0}", input.size());
        Log::Info("Groups => {0}", bias.size());

        for (int i = 0; i < input.size(); i++)
        {
            Log::Info("Processing Batch => {0} ", i);
            Tensor group{};
            group.Reshape((input[i].width + 2 * pad - ksize) / stride + 1,
                (input[i].height + 2 * pad - ksize) / stride + 1,
                bias.size());
                          
            for (int j = 0; j < bias.size(); j++)
            {
                Log::Info("Processing {0} Group(s)", j);
                Tensor intermediate{ input[i].width, input[i].height, input[i].depth };
                if (ksize != 1)
                {
                    auto im2col = input[i].IM2Col(ksize);
                    intermediate.GEMM(im2col, kernels[ksize * j]);
                }
                else
                {
                    intermediate.GEMM(input[i], kernels[ksize * j]);
                }
                intermediate.Blend();
                intermediate.AddBias(bias[i]);
                group.PushChannel(intermediate, j);

                Log::Info("Original Image: width => {0}\theight => {1}\tdepth => {2}",
                    input[i].width, input[i].height, input[i].depth);
                Log::Info("Conv & Grouped: width => {0}\theight => {1}\tdepth => {2}",
                   intermediate.width, intermediate.height, intermediate.depth);
            }
        }
    }

    void ConvolutionalLayer::Backward(Batch &input, Batch &output)
    {

    }
}
