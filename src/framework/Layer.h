#pragma once

#include <memory>
#include <map>
#include <string>
#include <vector>

#include "Map.h"
#include "Tensor.h"

namespace sl
{
    class Net;

    class Layer
    {
    public:
        enum class Type
        {
            None,
            Convolution,
            BatchNormalize,
            ReLu,
            MaxPool,
            Flatten,
            Softmax
        };

        struct Description
        {
            Description(Layer::Type t, const std::map<std::string, float> &b = DataSet::Bias::NONE, const std::map<std::string, std::string> &p = { }) :
                Type{ t },
                params{ p },
                bias{ b }
            {

            }

            Layer::Type Type;
            std::map<std::string, float> bias;
            const std::map<std::string, std::string> &params;
        };    

        template <class T>
        const T &Get() const
        {
            if constexpr (std::is_same_v<T, Type>)
            {
                return type;
            }
        }

        template <class T, class ... Args>
        inline constexpr static std::shared_ptr<Layer> Create(Args&& ... args)
        {
            return std::make_shared<T>(std::forward<Args>(args)...);
        }

    public:
        Layer(const Type &&t = Type::None) : type{ t } { }

        virtual ~Layer() { }

        virtual void Forward(Tensor::Batch &input, Tensor::Batch &output) { }

        virtual void Backward(Tensor::Batch &input, Tensor::Batch &output) { }

    public:
        Type type;
    };
}
