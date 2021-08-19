#pragma once

#include "framework/Layer.h"

namespace sl
{
    class BatchNormLayer : public Layer
    {
    public:
        BatchNormLayer(const Description &desc) : Layer{ Type::BatchNormalize }
        {

        }

        virtual void Forward(Tensor::Batch &input, Tensor::Batch &output) override;

        virtual void Backward(Tensor::Batch &input, Tensor::Batch &output) override;

    private:
        
    };
}
