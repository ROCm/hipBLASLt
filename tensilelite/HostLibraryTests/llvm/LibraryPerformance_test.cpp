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

#include <gtest/gtest.h>

#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/PerformanceMetricTypes.hpp>
#include <Tensile/llvm/YAML.hpp>
#include <TestUtils.hpp>

#include "TestData.hpp"

using namespace TensileLite;

/**
 * LibraryPerformanceTest:
 *
 * This suite contains micro-benchmarks for pieces of the runtime library.  It does not
 * exercise any of the Hip-specific code.
 *
 * There are no performance-based assertions or checks.  The timing results are provided by
 * googletest.
 *
 * Most of these tests depend on a library being loaded from a DAT/YAML file.  The library objects
 * are cached so that the deserialization time is not a part of the actual test (outside of the
 * LoadLibrary test). PopulateCache is an empty test whose purpose is to ensure the cache is
 * populated for the actual tests.
 */
struct LibraryPerformanceTest : public ::testing::TestWithParam<
                                    std::tuple<AMDGPU, std::string, bool, bool, PerformanceMetric>>
{
    AMDGPU                                                   hardware;
    std::string                                              filename;
    bool                                                     hasNavi, solutionRequired;
    PerformanceMetric                                        perfMetric;
    std::shared_ptr<SolutionLibrary<ContractionProblemGemm>> library;

    static std::map<std::string, std::shared_ptr<SolutionLibrary<ContractionProblemGemm>>>
        libraryCache;

    void SetUp() override
    {
        std::tie(hardware, filename, hasNavi, solutionRequired, perfMetric) = GetParam();

        if(hardware.processor == AMDGPU::Processor::gfx1010 && !hasNavi)
            GTEST_SKIP();

        library = loadLibrary();

        if(library == nullptr)
        {
            std::cout << libraryPath().native() << std::endl;
            if(!boost::filesystem::is_regular_file(libraryPath()))
                GTEST_SKIP();
            else
                ASSERT_NE(library, nullptr);
        }
    }

    boost::filesystem::path libraryPath()
    {
        return TestData::Instance().file(filename);
    }

    std::shared_ptr<SolutionLibrary<ContractionProblemGemm>> loadLibrary(bool cache = true)
    {
        if(!cache)
            return loadLibraryNoCache();

        auto pathStr = libraryPath().native();

        auto iter = libraryCache.find(pathStr);
        if(iter != libraryCache.end())
            return iter->second;

        return libraryCache[pathStr] = loadLibraryNoCache();
    }

    std::shared_ptr<SolutionLibrary<ContractionProblemGemm>> loadLibraryNoCache()
    {
        auto path = libraryPath();

        if(boost::filesystem::is_regular_file(path))
            return LoadLibraryFile<ContractionProblemGemm>(path.native());

        return nullptr;
    }
};

std::map<std::string, std::shared_ptr<SolutionLibrary<ContractionProblemGemm>>>
    LibraryPerformanceTest::libraryCache;

TEST_P(LibraryPerformanceTest, PopulateCache)
{
    // Empty test to ensure cache is populated by the SetUp() function.
    // See comment at top of this file.
}

TEST_P(LibraryPerformanceTest, LoadLibrary)
{
    auto library = loadLibrary(false);
}

TEST_P(LibraryPerformanceTest, CreateProblem)
{
    for(int i = 0; i < 10; i++)
        RandomGEMM();
}

TEST_P(LibraryPerformanceTest, FindSolution)
{
    for(int i = 0; i < 10; i++)
    {
        auto problem = RandomGEMM();
        problem.setPerformanceMetric(perfMetric);
        auto solution = library->findTopSolutions(problem, hardware, 1);

        if(solutionRequired)
            EXPECT_EQ(solution.size(), 1) << i << problem;
    }
}

TEST_P(LibraryPerformanceTest, FindCachedSolution)
{
    for(int i = 0; i < 10; i++)
    {
        auto problem = RandomGEMM();
        problem.setPerformanceMetric(perfMetric);
        auto solution = library->findTopSolutions(problem, hardware, 1);

        if(solutionRequired)
            EXPECT_EQ(solution.size(), 1) << i << problem;
    }

    auto problem = RandomGEMM();
    problem.setPerformanceMetric(perfMetric);
    for(int i = 0; i < 10; i++)
    {
        auto solution = library->findTopSolutions(problem, hardware, 1);

        if(solutionRequired)
            EXPECT_EQ(solution.size(), 1) << i << problem;
    }
}

TEST_P(LibraryPerformanceTest, FindAllSolutions)
{
    auto problem = RandomGEMM();
    problem.setPerformanceMetric(perfMetric);
    auto solution = library->findAllSolutions(problem, hardware);

    if(solutionRequired)
        EXPECT_GE(solution.size(), 1)
            << problem << ", " << problem.transA() << ", " << problem.transB();
}

TEST_P(LibraryPerformanceTest, SpecificSizes)
{
    // N	N	256	12	1024	1	256	1024	0	256
    auto problem = ContractionProblemGemm::GEMM_Strides(false, // transA
                                                        false, // transB
                                                        rocisa::DataType::Float, // aType
                                                        rocisa::DataType::Float, // bType
                                                        rocisa::DataType::Float, // cType
                                                        rocisa::DataType::Float, // dType
                                                        256, // m
                                                        12, // n
                                                        1024, // k
                                                        1, // batchSize
                                                        256, // lda
                                                        1024, // aStride
                                                        1024, // ldb
                                                        12, // bStride
                                                        256, // ldc
                                                        12, // cStride
                                                        256, // ldd
                                                        12, // dStride
                                                        2.0); // beta
    problem.setPerformanceMetric(perfMetric);
    auto solution = library->findTopSolutions(problem, hardware, 1);
    EXPECT_EQ(solution.size(), 1) << problem << ", " << problem.transA() << ", "
                                  << problem.transB();
}

TEST_P(LibraryPerformanceTest, GetSolutionByIndex)
{
    if(perfMetric == PerformanceMetric::ExperimentalMLP)
    {
        auto problem = ContractionProblemGemm::GEMM_Strides(true,
                                                            false,
                                                            DataType::Float,
                                                            DataType::Float,
                                                            DataType::Float,
                                                            DataType::Float,
                                                            256,
                                                            12,
                                                            1024,
                                                            1,
                                                            1024,
                                                            256,
                                                            1024,
                                                            12,
                                                            256,
                                                            12,
                                                            256,
                                                            12,
                                                            2.0);
        problem.setPerformanceMetric(perfMetric);
        auto solution = library->getSolutionByIndex(problem, hardware, 11);
        EXPECT_NE(solution, nullptr)
            << problem << ", " << problem.transA() << ", " << problem.transB();

        problem = ContractionProblemGemm::GEMM_Strides(false,
                                                       false,
                                                       DataType::Float,
                                                       DataType::Float,
                                                       DataType::Float,
                                                       DataType::Float,
                                                       256,
                                                       12,
                                                       1024,
                                                       1,
                                                       256,
                                                       1024,
                                                       1024,
                                                       12,
                                                       256,
                                                       12,
                                                       256,
                                                       12,
                                                       2.0);
        problem.setPerformanceMetric(perfMetric);
        solution = library->getSolutionByIndex(problem, hardware, 20);
        EXPECT_NE(solution, nullptr)
            << problem << ", " << problem.transA() << ", " << problem.transB();
    }
}
std::vector<LibraryPerformanceTest::ParamType> GetLibraries(std::string const& ext)
{
    std::vector<LibraryPerformanceTest::ParamType> rv;
    rv.push_back(std::make_tuple(AMDGPU(AMDGPU::Processor::gfx942, 304, "Aquavanjaram"),
                                 "Kernels." + ext,
                                 false,
                                 true,
                                 PerformanceMetric::DeviceEfficiency));
    rv.push_back(std::make_tuple(AMDGPU(AMDGPU::Processor::gfx942, 304, "Aquavanjaram"),
                                 "Mlp_Kernels." + ext,
                                 false,
                                 true,
                                 PerformanceMetric::ExperimentalMLP));
    return rv;
}

std::vector<LibraryPerformanceTest::ParamType> GetParams()
{
    std::vector<LibraryPerformanceTest::ParamType> rv;

#ifdef TENSILE_YAML
    auto yamlParams = GetLibraries("yaml");
    rv.insert(rv.end(), yamlParams.begin(), yamlParams.end());
#endif

#ifdef TENSILE_MSGPACK
    auto datParams = GetLibraries("dat");
    rv.insert(rv.end(), datParams.begin(), datParams.end());
#endif

    return rv;
}

INSTANTIATE_TEST_SUITE_P(LLVM, LibraryPerformanceTest, ::testing::ValuesIn(GetParams()));
