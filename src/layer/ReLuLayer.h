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

        virtual void Forward() override;

        virtual void Backward() override;

    private:
        
    };
}
