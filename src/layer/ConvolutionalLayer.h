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
        Description description;
        int ksize{ 0 };
        std::vector<float> bias;
        int stride{ 1 };
        int pad{ 0 };
    };
}
