#include <iostream>
#include "slcnn.h"

using namespace ::sl;

int main()
{
    Net net{{
        { Layer::Type::Convolution, 10, 20 },
        { Layer::Type::BatchNormalize, 10, 20 },
        { Layer::Type::ReLu, 10, 20 }
    }};

    net.Forward();
    net.Backward();

    return 0;
}
