#pragma once

#include "framework/Layer.h"

namespace sl
{
    class ConvolutionalLayer : public Layer
    {
    public:
        ConvolutionalLayer(const Description &desc) : Layer{ Type::Convolutional }
        {

        }

        virtual void Forward(Batch &input, Batch &output) override;

        virtual void Backward(Batch &input, Batch &output) override;

    private:
        
    };
}
