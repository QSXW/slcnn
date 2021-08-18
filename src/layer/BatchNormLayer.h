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

        virtual void Forward() override;

        virtual void Backward() override;

    private:
        
    };
}
