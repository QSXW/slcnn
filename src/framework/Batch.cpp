#include "Batch.h"

namespace sl
{
    Batch::Batch(const float *kernel, int ksize, int count)
    {
        pool.reserve(count);
        for (int i = 0; i < count; i++)
        {
            pool.emplace_back(Tensor{ kernel, ksize, ksize });
        }
    }

    Batch::Batch(std::initializer_list<Tensor> &&tensors) :
        pool{ std::move(tensors) }
    {

    }

    Batch::~Batch()
    {

    }

    void Batch::Mean()
    {
        mean.Reshape(pool[0].depth, 1, 1);

        float *tmp = new float[mean.size];

        for (int i = 0; i < pool.size(); i++)
        {
            auto &p = pool[i];
            BasicLinearAlgebraSubprograms::Mean(p.data.get(), p.depth, p.width * p.height, tmp);
            BasicLinearAlgebraSubprograms::Add(mean.data.get(), tmp, mean.size);
        }
        for (int i = 0; i < mean.size; i++)
        {
            mean[i] /= pool.size();
        }

        delete []tmp;
    }

    void Batch::Variance()
    {
        variance.Reshape(pool[0].depth, 1, 1);

        float *tmp = new float[variance.size];

        for (int i = 0; i < pool.size(); i++)
        {
            auto &p = pool[i];
            BasicLinearAlgebraSubprograms::Variance(p.data.get(), p.depth, p.width * p.height, mean.data.get(), tmp);
            BasicLinearAlgebraSubprograms::Add(variance.data.get(), tmp, variance.size);
        }
        for (int i = 0; i < variance.size; i++)
        {
            variance[i] /= pool.size();
        }

        delete []tmp;
    }

    void Batch::Normalized()
    {
        for (int i = 0; i < pool.size(); i++)
        {
            auto &p = pool[i];
            BasicLinearAlgebraSubprograms::Normalize(p.data.get(), p.depth, p.width * p.height, mean.data.get(), variance.data.get());
        }
    }

    void Batch::Scale(float *scale)
    {
        for (auto &b : pool)
        {
            BasicLinearAlgebraSubprograms::ScaleBiasAVX2(b.data.get(), scale, b.size);
        }
    }

    void Batch::Scale(float scale)
    {
        for (auto &b : pool)
        {
            BasicLinearAlgebraSubprograms::ScaleAVX2(b.data.get(), scale, b.size);
        }
    }

}
