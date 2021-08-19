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

    unsigned char *c = new unsigned char[19200 * 10800];

    net.Train();
    return 0;
}
