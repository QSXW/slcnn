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

    Tensor t1 = Tensor::TestCase;
    Tensor t2 = Tensor::TestCase;
    Tensor t3 = Tensor::TestCase;
    
    t1.Reshape(16, 16, 1);
    t2.Reshape(32, 32, 1);

    net.Set(Batch{ t1, t2, t3 });

    net.Train();
    return 0;
}
