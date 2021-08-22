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

        virtual void Forward(Batch &input, Batch &output) override;

        virtual void Backward(Batch &input, Batch &output) override;

    private:
        int size{ 2 };
    };
}
