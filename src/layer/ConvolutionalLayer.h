#pragma once

#include <cassert>
#include <string>

#include "framework/Layer.h"

namespace sl
{
    class ConvolutionalLayer : public Layer
    {
    public:
        ConvolutionalLayer(const Description &desc);

        virtual void Forward(Batch &input, Batch &output) override;

        virtual void Backward(Batch &input, Batch &output) override;

    private:
        int ksize{ 0 };
    };
}
