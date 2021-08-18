#include <iostream>
#include "slcnn.h"

using namespace ::sl;

int main()
{
    Net net{{
        { Layer::Type::BatchNormalize, 10, 20 }
    }};

    net.Train();

    return 0;
}
