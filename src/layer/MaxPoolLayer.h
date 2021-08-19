#pragma once

#include "framework/Layer.h"
#include "framework/Net.h"

namespace sl
{
    class MaxPoolLayer : public Layer
    {
    public:
        MaxPoolLayer(const Description &desc) : Layer{ Type::MaxPool }
        {

        }

        virtual void Forward(Tensor::Batch &input, Tensor::Batch &output) override;

        virtual void Backward(Tensor::Batch &input, Tensor::Batch &output) override;

    private:
        
    };
}
