#pragma once

#include <memory>

namespace sl
{
    class Layer
    {
    public:
        enum class Type
        {
            None,
            Convolution,
            BatchNormalize,
            ReLu
        };

        struct Description
        {
            Description(Layer::Type t, int w, int s) :
                Type{ t },
                winograd{ w },
                sgemm{ s }
            {

            }

            Layer::Type Type;
            int winograd;
            int sgemm;
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

        virtual void OnAttach() { }

        virtual void OnDetach() { }

    public:
        Type type;
    };
}
