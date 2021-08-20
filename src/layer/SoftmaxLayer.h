#pragma once

#include "framework/Layer.h"
#include "framework/Net.h"

namespace sl
{
    class SoftmaxLayer : public Layer
    {
    public:
        SoftmaxLayer(const Description &desc) : Layer{ Type::Softmax }
        {

        }

        virtual void Forward(Batch &input, Batch &output) override;

        virtual void Backward(Batch &input, Batch &output) override;

    private:
        
    };
}
