#include <iostream>
#include "slcnn.h"

using namespace ::sl;

#include "Test.h"

#include "spdlog/spdlog.h"

int main()
{
    Log::Launch();
    Test::Launch();

    Batch kernels{ DataSet::Kernel::Gaussian, 3, 30 };

    Net net{ {
        { Layer::Type::Convolutional, {}, DataSet::Bias::CONV1_1, kernels },
        { Layer::Type::BatchNormalize  },
        { Layer::Type::Activation, {{ "Activation", "ReLu" }} },
        { Layer::Type::Convolutional, {}, DataSet::Bias::CONV1_1, kernels },
        { Layer::Type::BatchNormalize  },
        { Layer::Type::Activation, {{ "Activation", "ReLu" }} },
        { Layer::Type::MaxPool },
        { Layer::Type::Convolutional, {}, DataSet::Bias::CONV2_1, kernels },
        { Layer::Type::BatchNormalize  },
        { Layer::Type::Activation, {{ "Activation", "ReLu" }} },
        { Layer::Type::Convolutional, {}, DataSet::Bias::CONV2_2, kernels },
        { Layer::Type::BatchNormalize  },
        { Layer::Type::Activation, {{ "Activation", "ReLu" }} },
        { Layer::Type::MaxPool },
        { Layer::Type::Flatten },
        { Layer::Type::Softmax }
    } };

    net.Set(Batch{ Tensor::TestCase, Tensor::TestCase, Tensor::TestCase });

    auto t = Tensor::TestCase.IM2Col(3);

    net.Train();
    return 0;
}
