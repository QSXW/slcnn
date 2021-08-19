#pragma once

#include "framework/Layer.h"

namespace sl
{
    class ReLuLayer : public Layer
    {
    public:
        ReLuLayer(const Description &desc) : Layer{ Type::ReLu }
        {

        }

        virtual void Forward(Tensor::Batch &input, Tensor::Batch &output) override;

        virtual void Backward(Tensor::Batch &input, Tensor::Batch &output) override;

    private:
        
    };
}
