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

        virtual void Forward(Batch &input, Batch &output) override;

        virtual void Backward(Batch &input, Batch &output) override;

    private:
        
    };
}
