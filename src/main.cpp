#include <iostream>
#include "slcnn.h"

using namespace ::sl;

#include "Test.h"

int main()
{
    Test::Launch();

    Net net{ {
        { Layer::Type::Convolutional, DataSet::Bias::CONV1_1, {{ "KernelSize", "3" }} },
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

    auto t = Tensor::TestCase.IM2Col(3);

    net.Train();
    return 0;
}
