#include "ActivationLayer.h"

#include <iostream>
#include <cmath>
namespace sl
{
namespace Activate
{
    float LINEAR(float x)
    {
        return x;
    };

    float LOGSTIC(float x)
    {
        return 1.0f / (1.0f + exp(-x));
    };

    float LOGGY(float x)
    {
        return 2.0f / (1.0f + exp(-x)) - 1;
    };

    float RELU(float x)
    {
        return x * (x > 0);
    };

    float ELU(float x)
    { 
        return (x >= 0) * x + (x < 0) * (exp(x) - 1.0f);
    };

    float SELU(float x)
    { 
        return (x >= 0) * 1.0507f * x + (x < 0) * 1.0507f * 1.6732f * (exp(x) - 1.0f);
    };

    float RELIE(float x)
    {
        return (x > 0) ? x : .01f * x;
    };

    float RAMP(float x)
    {
        return x * (x > 0) + .1f * x;
    };

    float LEAKY(float x)
    {
        return (x > 0) ? x : .1f * x;
    };

    float TANH(float x)
    {
        return (exp(2.0f * x) - 1.0f) / (exp(2.0f * x) + 1.0f);
    };

    float STAIR(float x)
    {
        int n = floor(x);
        if (n % 2 == 0)
        {
            return floor(x / 2.0f);
        }
        else
        {
            return (x - static_cast<float>(n)) + floor(x / 2.0f);
        }
    };

    float HARDTAN(float x)
    {
        if (x < -1.0f)
        {
            return -1.0f;
        }
        if (x > 1.0f)
        {
            return 1.0f;
        }
        return x;
    };

    float PLSE(float x)
    {
        if (x < -4.0f)
        {
            return .01f * (x + 4.0f);
        }
        if (x > 4.0f)
        {
            return .01f * (x - 4.0f) + 1.0f;
        }
        return .125f * x + .5f;
    };

    float LHTAN(float x)
    {
        if (x < 0)
        {
            return .001f * x;
        }
        if (x > 1)
        {
            return .001f * (x - 1.0f) + 1.0f;
        }
        return x;
    };
}

namespace Gradient
{
    float LINEAR(float x)
    {
        return 1;
    };

    float LOGSTIC(float x)
    {
        return (1.0 - x) * x;
    };

    float LOGGY(float x)
    {
        float y = (x + 1.0f) / 2.0f;
        return 2.0f * (1.0f - y) * y;
    };

    float RELU(float x)
    {
       return (x > 0);
    };

    float ELU(float x)
    { 
        return (x >= 0) + (x < 0) * (x + 1.0f);
    };

    float SELU(float x)
    { 
        return (x >= 0) * 1.0507f + (x < 0) * (x + 1.0507f * 1.6732f);
    };

    float RELIE(float x)
    {
        return (x > 0 ) ? 1.0f : .01f;
    };

    float RAMP(float x)
    {
        return (x > 0) + .1f;
    };

    float LEAKY(float x)
    {
        return (x > 0) ? 1.0f : .1f;
    };

    float TANH(float x)
    {
        return 1.0f - x * x;
    };

    float STAIR(float x)
    {
        if (floor(x) == x)
        {
            return 0;
        }
        return 1;
    };

    float HARDTAN(float x)
    {
        if (x > -1 && x < 1)
        {
            return 1;
        }
        return 0;
    };

    float PLSE(float x)
    {
        return (x < 0 || x > 1.0f) ? .01f : .125f;
    };

    float LHTAN(float x)
    {
        if (x > 0 && x < 1)
        {
            return 1.0f;
        }
        return .001f;
    };
}

    std::map<std::string, Activate::Caller> ActivationLayer::Activates = {
        { "linear",   Activate::LINEAR  },
        { "logistic", Activate::LOGSTIC },
        { "loggy",    Activate::LOGGY   },      
        { "relu",     Activate::RELU    },      
        { "elu",      Activate::ELU     },      
        { "selu",     Activate::SELU    },      
        { "relie",    Activate::RELIE   },      
        { "ramp",     Activate::RAMP    },      
        { "leaky",    Activate::LEAKY   },      
        { "tanh",     Activate::TANH    },
        { "stair",    Activate::STAIR   },
        { "hardtan",  Activate::HARDTAN },
        { "plse",     Activate::PLSE     },
        { "lhtan",    Activate::LHTAN }
    };

     std::map<std::string, Gradient::Caller> ActivationLayer::Gradients = {
        { "linear",   Gradient::LINEAR  },
        { "logistic", Gradient::LOGSTIC },
        { "loggy",    Gradient::LOGGY   },      
        { "relu",     Gradient::RELU    },      
        { "elu",      Gradient::ELU     },      
        { "selu",     Gradient::SELU    },      
        { "relie",    Gradient::RELIE   },      
        { "ramp",     Gradient::RAMP    },      
        { "leaky",    Gradient::LEAKY   },      
        { "tanh",     Gradient::TANH    },
        { "stair",    Gradient::STAIR   },
        { "hardtan",  Gradient::HARDTAN },
        { "plse",     Gradient::PLSE    },
        { "lhtan",    Gradient::LHTAN   }
    };

    void ActivationLayer:: Forward(Batch &input, Batch &output)
    {
        Log::Info("Forwarding: Layer => {0}: {1}", Layer::Stringify(type), key);
        output = std::move(input);
        for (auto &batch : output)
        {
            for (int i = 0; i < batch.size; i++)
            {
                batch[i] = activate(batch[i]);
            }
        }
    }

    void ActivationLayer::Backward(Batch &input, Batch &output)
    {
        Log::Info("Backwarding: Layer => {0}: {1}", Layer::Stringify(type), key);
        output = std::move(input);
        for (auto &batch : output)
        {
            for (int i = 0; i < batch.size; i++)
            {
                batch[i] = gradient(batch[i]);
            }
        }
    }
}
