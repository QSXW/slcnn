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

        virtual void Forward(Batch &input, Batch &output) override;

        virtual void Backward(Batch &input, Batch &output) override;

    private:
        
    };
}
