#include "ConvolutionalLayer.h"

#include <iostream>
#include <cassert>

namespace sl
{
    ConvolutionalLayer::ConvolutionalLayer(const Description &desc) : 
        Layer{ Type::Convolutional },
        description{ desc }
    {
        assert((!description.kernels.empty()) && "Illegel empty kernel");
        ksize = description.kernels[0].width;

        assert((ksize * description.bias.size() == description.kernels.size()) &&
               (ksize >= 1 && ksize <= 19) && "Illegal kernel size");
    }

    void ConvolutionalLayer::Forward(Batch &input, Batch &output)
    {
        Log::Info("Forwarding: Layer => {0}", Layer::Stringify(type));
        auto &bias   = description.bias;
        auto &kernels = description.kernels;

        for (int i = 0; i < input.size(); i++)
        {
            Log::Info("Convoluting {0} Batch", i);
            for (int j = 0; j < bias.size(); j++)
            {
                Tensor intermediate = input[i];
                if (ksize != 1)
                {
                    auto im2col = input[i].IM2Col(ksize);
                    intermediate = im2col;
                }
               intermediate.GEMM(kernels[i * j]);
               intermediate.AddBias(1.0);
               output.push_back(intermediate);
            }
        }
    }

    void ConvolutionalLayer::Backward(Batch &input, Batch &output)
    {

    }
}
