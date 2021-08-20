#include "ConvolutionalLayer.h"

#include <iostream>
#include <cassert>

namespace sl
{
    ConvolutionalLayer::ConvolutionalLayer(const Description &desc) : Layer{ Type::Convolutional }
    {
        auto it = desc.params.find("KernelSize");
        if (it != desc.params.end())
        {
            ksize = std::stoi(std::get<1>(*it));
            assert((ksize >= 1 && ksize <= 19) && "Illegal kernel size");
        }
    }

    void ConvolutionalLayer::Forward(Batch &input, Batch &output)
    {
        for (auto &batch : input)
        {
            Tensor &intemediate = batch;
            if (ksize != 1)
            {
                auto im2col = batch.IM2Col(ksize);
                intemediate = im2col;
            }
            output.push_back(intemediate.GEMM());
        }
    }

    void ConvolutionalLayer::Backward(Batch &input, Batch &output)
    {

    }
}
