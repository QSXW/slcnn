#pragma once

#include "framework/Layer.h"

namespace sl
{
    class ConvolutionLayer : public Layer
    {
    public:
        ConvolutionLayer(const Description &desc) : Layer{ Type::Convolution }
        {

        }

        virtual void Forward(Tensor::Batch &input, Tensor::Batch &output) override;

        virtual void Backward(Tensor::Batch &input, Tensor::Batch &output) override;

    private:
        
    };
}
