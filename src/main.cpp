#include <iostream>
#include "slcnn.h"

using namespace ::sl;

int main()
{
    Net net{ {
        { Layer::Type::Convolution, DataSet::Bias::CONV1_1 },
        { Layer::Type::BatchNormalize  },
        { Layer::Type::ReLu },
        { Layer::Type::Convolution, DataSet::Bias::CONV1_1 },
        { Layer::Type::BatchNormalize  },
        { Layer::Type::ReLu },
        { Layer::Type::MaxPool },
        { Layer::Type::Convolution, DataSet::Bias::CONV2_1 },
        { Layer::Type::BatchNormalize  },
        { Layer::Type::ReLu },
        { Layer::Type::Convolution, DataSet::Bias::CONV2_2 },
        { Layer::Type::BatchNormalize  },
        { Layer::Type::ReLu },
        { Layer::Type::MaxPool },
        { Layer::Type::Flatten },
        { Layer::Type::Softmax }
    } };

    net.Set(Tensor::Batch{ Tensor::TestCase });
    net.Train();
    return 0;
}
