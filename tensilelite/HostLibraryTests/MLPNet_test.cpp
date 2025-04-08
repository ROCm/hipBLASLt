/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <stdexcept>
#include <random>

#include <gtest/gtest.h>
#include <Tensile/MLPClassification.hpp>

constexpr double abs_error = 10. * std::numeric_limits<TensileLite::MLPClassification::dtype>::epsilon();

TEST(MLPNet, DenseLayer)
{
    using namespace TensileLite;
    using namespace MLPClassification;
    
    DenseLayer test_dense{
        /* weights */ 
        std::vector({0.6634151484170232f,  0.788165180871102f,   0.31248166526753884f,
                     0.23942935302736823f, 0.6809768405365064f,  0.1367808736375885f,
                     0.5374190113071796f,  0.9243177539724999f,  0.2626090032418886f,
                     0.25681768989410403f, 0.9874451518147117f,  0.42539241956479557f}),
        /* bias    */
        std::vector({0.5428595043381658f, 0.17016816526861123f, 
                     0.848431351596801f, 0.8236885843811014f})
    };

    auto Fout = test_dense({0.43858885174024276f, 0.7889586579958023f, 0.6683846141676605f});

    std::vector<dtype> Ftrue({1.6645136731628662f, 0.9038640159734725f,
                              1.9889096507140147f, 1.9997051101391978f});

    for (std::size_t i=0; i<Fout.size(); i++)
        EXPECT_NEAR(Fout[i], Ftrue[i], abs_error);
}

template <typename T>
std::vector<T> normal_random_vector(std::size_t n) {
    std::default_random_engine gen;
    std::normal_distribution<T> dist(0., 1.0);
    auto generator = std::bind(dist, gen);
    std::vector<T> v(n);
    std::generate(v.begin(), v.end(), generator);
    return v;
}

TensileLite::MLPClassification::DenseLayer
random_dense_layer(std::size_t n_in, std::size_t n_out) {
    using namespace TensileLite;
    using namespace MLPClassification;
    return DenseLayer(normal_random_vector<dtype>(n_in * n_out),
                      normal_random_vector<dtype>(n_out));
}

TensileLite::MLPClassification::ResBlock
random_resblock(std::size_t n_in, std::size_t hidden, std::size_t n_out) {
    using namespace TensileLite;
    using namespace MLPClassification;
    ResBlock r;
    r.linear1 = random_dense_layer(n_in, hidden);
    r.linear2 = random_dense_layer(hidden, n_out);
    r.res = random_dense_layer(n_in, n_out);
    return r;
}

TEST(MLPNet, DenseLayerFixed)
{
    using namespace TensileLite;
    using namespace MLPClassification;

    /* DenseLayer has some sizes hardcoded for optimization, test these sizes */
    int n_in = 16, n_out = 3;

    auto weights = normal_random_vector<dtype>(n_out*n_in);
    auto bias = normal_random_vector<dtype>(n_out);
    DenseLayer test_dense{weights, bias};
    // EXPECT_FALSE(std::string(typeid(test_dense.W.get()).name()).find("WeightMatrixFixed") == std::string::npos);

    auto Fin = normal_random_vector<dtype>(n_in);
    auto Fout = test_dense(Fin);
    std::vector<dtype> Ftrue(n_out);
    for (int i=0; i<n_out; i++) {
        dtype ftrue = bias[i];
        for (int j=0; j<n_in; j++)
            ftrue += weights[i*n_in+j] * Fin[j];
        EXPECT_NEAR(Fout[i], ftrue, abs_error);
    }
}

TEST(MLPNet, DenseLayerDimFail)
{
    using namespace TensileLite;
    using namespace MLPClassification;

    /* weights dimension is not a multiple of bias dimension */
    EXPECT_THROW(
        (DenseLayer{std::vector({1.f, 2.f, 3.f}), std::vector({1.f, 2.f})}),
        std::runtime_error);
}

TEST(MLPNet, StandardScaler)
{
    using namespace TensileLite;
    using namespace MLPClassification;

    StandardScaler test_scaler{
        /* mean  */ std::vector{0.4525329262019901f, 0.8647806535129754f},
        /* scale */ std::vector{0.05201354125426511f, 0.06123320047178044f}
    };
    
    std::vector Fin{3.991355396203433e-04f, 6.381927186481492e-01f};
    auto F = Fin;
    test_scaler(F);

    for (std::size_t i=0; i<F.size(); i++)
        EXPECT_NEAR(F[i], (Fin[i] - test_scaler.mean[i]) / test_scaler.scale[i], abs_error);

    std::vector<dtype> Ftrue({-8.692616956267996f, -3.7004097959774316f});
    for (std::size_t i=0; i<F.size(); i++)
        EXPECT_NEAR(F[i], Ftrue[i], abs_error);
}

TEST(MLPNet, ResBlock)
{
    using namespace TensileLite;
    using namespace MLPClassification;

    int n_in = 3, h = 6, n_out = 5;

    ResBlock b = random_resblock(n_in, h, n_out);
    auto Fin = normal_random_vector<dtype>(n_in);
    auto Fout = b(Fin);

    auto Ftmp = b.linear1(Fin);
    for (auto& f : Ftmp) 
        f = f > 0 ? f : 0.;
    auto Ftrue = b.linear2(Ftmp);
    auto Fres = b.res(Fin);
    for (std::size_t i=0; i<Fres.size(); i++)
        Ftrue[i] += Fres[i];
    for (auto& f : Ftrue) 
        f = f > 0 ? f : 0.;

    for (std::size_t i=0; i<Ftrue.size(); i++)
        EXPECT_NEAR(Ftrue[i], Fout[i], abs_error);
}

TEST(MLPNet, MLPNet)
{
    using namespace TensileLite;
    using namespace MLPClassification;
    
    std::size_t n_solutions = 9, n_features = MLPNet::n_features;
    std::size_t h1 = 3, h2 = 5, h3 = 4, h4 = 7;

    MLPNet net;
    net.res_blocks.push_back(random_resblock(n_features, h1, h2));
    net.res_blocks.push_back(random_resblock(h2, h3, h4));
    net.dense = random_dense_layer(h4, n_solutions);
    net.scaler.mean = std::vector<dtype>(n_features, .7);
    net.scaler.scale = std::vector<dtype>(n_features, 3.6);

    EXPECT_TRUE(net.valid());

    std::vector<float> probkey = normal_random_vector<float>(4);
    auto Fout = net.predict(probkey);
    EXPECT_TRUE(Fout.size() == n_solutions);

    for (auto fi : Fout) {
        EXPECT_TRUE(std::isfinite(fi));
        EXPECT_FALSE(std::isnan(fi));
    }
}
