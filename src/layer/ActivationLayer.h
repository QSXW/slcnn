#pragma once

#include "framework/Layer.h"
#include "framework/Helper.h"

#include <map>

namespace sl
{
    namespace Activate
    {
        using Caller = float(*)(float x);
    }

    namespace Gradient
    {
        using Caller = float(*)(float x);
    }

    class ActivationLayer : public Layer
    {
    public:
        static std::map<std::string, Activate::Caller> Activates;
        static std::map<std::string, Gradient::Caller> Gradients;

    public:
        ActivationLayer(const Description &desc) : Layer{ Type::Activation }
        {
            auto pair = desc.params.find("Activation");
            if (pair == desc.params.end())
            {
                activate = Activates["linear"];
            }
            auto &key = Helper::ToLower(std::get<1>(*pair));
            activate = Activates[key];
            gradient = Gradients[key];
        }

        virtual void Forward(Batch &input, Batch &output) override;

        virtual void Backward(Batch &input, Batch &output) override;

    private:
        Activate::Caller activate{ nullptr };
        Gradient::Caller gradient{ nullptr };
    };
}
