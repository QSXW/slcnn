#include <iostream>
#include "slcnn.h"

using namespace ::sl;

#include "Test.h"

int main()
{
    Test::Launch();

    Net net{ {
        { Layer::Type::Convolutional, DataSet::Bias::CONV1_1 },
        { Layer::Type::BatchNormalize  },
        { Layer::Type::Activation, {}, {{ "Activation", "ReLu" }} },
        { Layer::Type::Convolutional, DataSet::Bias::CONV1_1 },
        { Layer::Type::BatchNormalize  },
        { Layer::Type::Activation, {}, {{ "Activation", "ReLu" }} },
        { Layer::Type::MaxPool },
        { Layer::Type::Convolutional, DataSet::Bias::CONV2_1 },
        { Layer::Type::BatchNormalize  },
        { Layer::Type::Activation, {}, {{ "Activation", "ReLu" }} },
        { Layer::Type::Convolutional, DataSet::Bias::CONV2_2 },
        { Layer::Type::BatchNormalize  },
        { Layer::Type::Activation, {}, {{ "Activation", "ReLu" }} },
        { Layer::Type::MaxPool },
        { Layer::Type::Flatten },
        { Layer::Type::Softmax }
    } };

    net.Set(Batch{ Tensor::TestCase, Tensor::TestCase, Tensor::TestCase });
    net.Train();
    return 0;
}
