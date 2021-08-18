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

        virtual void Forward() override;

        virtual void Backward() override;

    private:
        
    };
}
