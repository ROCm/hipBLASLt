/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionProblem_Detail.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/Utils.hpp>

#include <cctype>
#include <cstddef>
#include <set>

namespace Tensile
{
    ContractionProblem::ContractionProblem(size_t size, size_t workspaceSize)
        : m_workspaceSize(workspaceSize)
    {
        m_tensors.resize(size);
        m_names.resize(size);
    }

    ContractionProblemGemm ContractionProblemGemm::GEMM_Strides(bool     transA,
                                                                bool     transB,
                                                                DataType aType,
                                                                DataType bType,
                                                                DataType cType,
                                                                DataType dType,
                                                                size_t   m,
                                                                size_t   n,
                                                                size_t   k,
                                                                size_t   batchSize,
                                                                size_t   lda,
                                                                size_t   aStride,
                                                                size_t   ldb,
                                                                size_t   bStride,
                                                                size_t   ldc,
                                                                size_t   cStride,
                                                                size_t   ldd,
                                                                size_t   dStride,
                                                                double   beta)
    {
        Tensile::ContractionProblemGemm::FreeIndices  free(2);
        Tensile::ContractionProblemGemm::BoundIndices bound(1);
        Tensile::ContractionProblemGemm::BatchIndices batch(1);

        free[0].isA = true;
        free[0].i = free[0].c = free[0].d = 0;
        free[1].isA                       = false;
        free[1].i = free[1].c = free[1].d = 1;

        batch[0].a = batch[0].b = batch[0].c = batch[0].d = 2;

        TensorDescriptor a, b, c, d;

        if(transA)
        {
            a          = TensorDescriptor("a", aType, {k, m, batchSize}, {1, lda, aStride});
            free[0].i  = 1;
            bound[0].a = 0;
        }
        else
        {
            a          = TensorDescriptor("a", aType, {m, k, batchSize}, {1, lda, aStride});
            free[0].i  = 0;
            bound[0].a = 1;
        }

        if(transB)
        {
            b          = TensorDescriptor("b", bType, {n, k, batchSize}, {1, ldb, bStride});
            free[1].i  = 0;
            bound[0].b = 1;
        }
        else
        {
            b          = TensorDescriptor("b", bType, {k, n, batchSize}, {1, ldb, bStride});
            free[1].i  = 1;
            bound[0].b = 0;
        }

        c = TensorDescriptor("c", cType, {m, n, batchSize}, {1, ldc, cStride});
        d = TensorDescriptor("d", dType, {m, n, batchSize}, {1, ldd, dStride});

        TensorDescriptor e("e");
        TensorDescriptor bias("bias");
        TensorDescriptor scaleD("scaleD");

        ContractionProblemGemm problem(a, b, c, d, e, bias, scaleD, free, batch, bound, beta);

        return problem;
    }

    ContractionProblemGemm ContractionProblemGemm::GEMM(bool   transA,
                                                        bool   transB,
                                                        size_t m,
                                                        size_t n,
                                                        size_t k,
                                                        size_t lda,
                                                        size_t ldb,
                                                        size_t ldc,
                                                        double beta,
                                                        bool   colMajor,
                                                        size_t batchCount)
    {
        if(colMajor)
            throw std::runtime_error("Column major not yet implemented.");

        return GEMM_Strides(transA,
                            transB,
                            DataType::Float,
                            DataType::Float,
                            DataType::Float,
                            DataType::Float,
                            m,
                            n,
                            k,
                            batchCount,
                            lda,
                            -1,
                            ldb,
                            -1,
                            ldc,
                            -1,
                            ldc,
                            -1,
                            beta);
    }

    ContractionProblemGemm ContractionProblemGemm::GEMM(bool                    transA,
                                                        bool                    transB,
                                                        TensorDescriptor const& a,
                                                        TensorDescriptor const& b,
                                                        TensorDescriptor const& c,
                                                        TensorDescriptor const& d,
                                                        double                  beta)
    {
        Tensile::ContractionProblemGemm::FreeIndices free(2);
        BoundIndex                                   bound;

        free[0].isA = true;
        free[0].i = free[0].c = free[0].d = 0;
        free[1].isA                       = false;
        free[1].i = free[1].c = free[1].d = 1;

        if(transA)
        {
            free[0].i = 1;
            bound.a   = 0;
        }
        else
        {
            free[0].i = 0;
            bound.a   = 1;
        }

        if(transB)
        {
            free[1].i = 0;
            bound.b   = 1;
        }
        else
        {
            free[1].i = 1;
            bound.b   = 0;
        }

        FreeIndices  freeIndices{free};
        BatchIndices batchIndices;
        BoundIndices boundIndices{bound};

        batchIndices.push_back({2, 2, 2, 2});

        TensorDescriptor e("e");
        TensorDescriptor bias("bias");
        TensorDescriptor scaleD("scaleD");

        return ContractionProblemGemm(
            a, b, c, d, e, bias, scaleD, freeIndices, batchIndices, boundIndices, beta);
    }

    void ContractionProblemGemm::IdentifierToIndices(std::string const& identifier,
                                                     FreeIndices&       freeIndices,
                                                     BatchIndices&      batchIndices,
                                                     BoundIndices&      boundIndices,
                                                     std::vector<bool>& isComplex)
    {
        FreeIndices  free;
        BatchIndices batch;
        BoundIndices bound;
        bool         aIsComplex = false;
        bool         bIsComplex = false;
        bool         cIsComplex = false;
        bool         dIsComplex = false;

        std::string prefix = "Contraction_";
        if(identifier.find(prefix) != 0)
            throw std::runtime_error(concatenate(
                "Contraction identifier (", identifier, ") must start with '", prefix, "'."));

        size_t begin     = prefix.size();
        size_t end       = identifier.find("_", begin);
        size_t nextBegin = end + 1;

        std::string boundStr = identifier.substr(begin, end - begin);

        begin     = nextBegin;
        end       = identifier.find("_", begin);
        nextBegin = end + 1;

        if(identifier.at(begin) != 'A')
            throw std::runtime_error(concatenate(
                "Contraction identifier (", identifier, ")must match 'Contraction_s_A_B_C_D'"));

        if(identifier.at(end - 1) == 'C')
        {
            aIsComplex = true;
            end--;
        }

        begin++;
        std::string a = identifier.substr(begin, end - begin);

        begin     = nextBegin;
        end       = identifier.find("_", begin);
        nextBegin = end + 1;

        if(identifier.at(begin) != 'B')
            throw std::runtime_error(concatenate(
                "Contraction identifier (", identifier, ")must match 'Contraction_s_A_B_C_D'"));

        if(identifier.at(end - 1) == 'C')
        {
            bIsComplex = true;
            end--;
        }

        begin++;
        std::string b = identifier.substr(begin, end - begin);

        begin     = nextBegin;
        end       = identifier.find("_", begin);
        nextBegin = end + 1;

        if(identifier.at(begin) != 'C')
            throw std::runtime_error(concatenate(
                "Contraction identifier (", identifier, ")must match 'Contraction_s_A_B_C_D'"));

        if(identifier.at(end - 1) == 'C')
        {
            cIsComplex = true;
            end--;
        }

        begin++;
        std::string c = identifier.substr(begin, end - begin);

        begin = nextBegin;
        end   = identifier.find("_", begin);

        if(identifier.at(begin) != 'D')
            throw std::runtime_error(concatenate(
                "Contraction identifier (", identifier, ")must match 'Contraction_s_A_B_C_D'"));

        if(end != std::string::npos)
            throw std::runtime_error(concatenate(
                "Contraction identifier (", identifier, ")must match 'Contraction_s_A_B_C_D'"));

        end = identifier.size();

        if(identifier.at(end - 1) == 'C')
        {
            dIsComplex = true;
            end--;
        }

        begin++;
        std::string d = identifier.substr(begin, end - begin);

        auto caseInsensitiveCmp = [](char a, char b) { return std::tolower(a) < std::tolower(b); };
        std::set<char, decltype(caseInsensitiveCmp)> allIndices(
            a.begin(), a.end(), caseInsensitiveCmp);
        allIndices.insert(b.begin(), b.end());
        allIndices.insert(c.begin(), c.end());
        allIndices.insert(d.begin(), d.end());

        for(char index : allIndices)
        {
            char   caseInsensitiveIndex[] = {char(std::tolower(index)), char(std::toupper(index))};
            size_t aIndex                 = a.find_first_of(caseInsensitiveIndex, 0, 2);
            size_t bIndex                 = b.find_first_of(caseInsensitiveIndex, 0, 2);
            size_t cIndex                 = c.find(index);
            size_t dIndex                 = d.find(index);

            if(aIndex != std::string::npos && bIndex != std::string::npos
               && cIndex != std::string::npos && dIndex != std::string::npos)
            {
                batch.push_back(BatchIndex{aIndex, bIndex, cIndex, dIndex});
            }
            else if(aIndex != std::string::npos && bIndex != std::string::npos
                    && cIndex == std::string::npos && dIndex == std::string::npos)
            {
                bool aMirror = std::isupper(a[aIndex]);
                bool bMirror = std::isupper(b[bIndex]);
                bound.push_back(BoundIndex{aIndex, bIndex, aMirror, bMirror});
            }
            else if(aIndex != std::string::npos && bIndex == std::string::npos
                    && cIndex != std::string::npos && dIndex != std::string::npos)
            {
                free.resize(free.size() + 1);

                free.back().isA = true;
                free.back().i   = aIndex;
                free.back().c   = cIndex;
                free.back().d   = dIndex;
            }
            else if(aIndex == std::string::npos && bIndex != std::string::npos
                    && cIndex != std::string::npos && dIndex != std::string::npos)
            {
                free.resize(free.size() + 1);

                free.back().isA = false;
                free.back().i   = bIndex;
                free.back().c   = cIndex;
                free.back().d   = dIndex;
            }
        }
        freeIndices  = std::move(free);
        batchIndices = std::move(batch);
        boundIndices = std::move(bound);
        isComplex    = {aIsComplex, bIsComplex, cIsComplex, dIsComplex};
    }

    ContractionProblemGemm
        ContractionProblemGemm::FromIndexSizes(std::string const&         operationIdentifier,
                                               std::vector<size_t> const& indexSizes,
                                               DataType                   aType,
                                               std::vector<size_t> const& aStrides,
                                               DataType                   bType,
                                               std::vector<size_t> const& bStrides,
                                               DataType                   cType,
                                               std::vector<size_t> const& cStrides,
                                               DataType                   dType,
                                               std::vector<size_t> const& dStrides,
                                               double                     beta)
    {
        FreeIndices       freeIndices;
        BatchIndices      batchIndices;
        BoundIndices      boundIndices;
        std::vector<bool> isComplex;

        IdentifierToIndices(
            operationIdentifier, freeIndices, batchIndices, boundIndices, isComplex);

        for(size_t i = 0; i < isComplex.size(); i++)
        {
            if(isComplex[i])
            {
                std::runtime_error("Complex is not supported.");
            }
        }

        return FromIndexSizes(freeIndices,
                              batchIndices,
                              boundIndices,
                              indexSizes,
                              aType,
                              aStrides,
                              bType,
                              bStrides,
                              cType,
                              cStrides,
                              dType,
                              dStrides,
                              beta);
    }

    ContractionProblemGemm
        ContractionProblemGemm::FromIndexSizes(FreeIndices const&         freeIndices,
                                               BatchIndices const&        batchIndices,
                                               BoundIndices const&        boundIndices,
                                               std::vector<size_t> const& indexSizes,
                                               DataType                   aType,
                                               std::vector<size_t> const& aStrides,
                                               DataType                   bType,
                                               std::vector<size_t> const& bStrides,
                                               DataType                   cType,
                                               std::vector<size_t> const& cStrides,
                                               DataType                   dType,
                                               std::vector<size_t> const& dStrides,
                                               double                     beta)
    {
        size_t maxA = 0;
        size_t maxB = 0;
        size_t maxC = 0;
        size_t maxD = 0;

        // Determine number of dimension for each tensor.

        for(auto const& free : freeIndices)
        {
            if(free.isA)
                maxA = std::max(maxA, free.i);
            else
                maxB = std::max(maxB, free.i);
            maxC = std::max(maxC, free.c);
            maxD = std::max(maxD, free.d);
        }

        for(auto const& batch : batchIndices)
        {
            maxA = std::max(maxA, batch.a);
            maxB = std::max(maxB, batch.b);
            maxC = std::max(maxC, batch.c);
            maxD = std::max(maxD, batch.d);
        }

        for(auto const& bound : boundIndices)
        {
            maxA = std::max(maxA, bound.a);
            maxB = std::max(maxB, bound.b);
        }

        std::vector<size_t> aSizes(maxA + 1), bSizes(maxB + 1), cSizes(maxC + 1), dSizes(maxD + 1);

        for(auto const& free : freeIndices)
        {
            size_t indexSize = indexSizes.at(free.d);
            if(free.isA)
                aSizes[free.i] = indexSize;
            else
                bSizes[free.i] = indexSize;

            cSizes[free.c] = indexSize;
            dSizes[free.d] = indexSize;
        }

        for(auto const& batch : batchIndices)
        {
            size_t indexSize = indexSizes.at(batch.d);

            aSizes[batch.a] = indexSize;
            bSizes[batch.b] = indexSize;
            cSizes[batch.c] = indexSize;
            dSizes[batch.d] = indexSize;
        }

        size_t indexIdx = dSizes.size();
        for(auto const& bound : boundIndices)
        {
            size_t indexSize = indexSizes.at(indexIdx);

            aSizes[bound.a] = indexSize;
            bSizes[bound.b] = indexSize;

            indexIdx++;
        }

        TensorDescriptor a(
            "a", aType, aSizes.begin(), aSizes.end(), aStrides.begin(), aStrides.end());
        TensorDescriptor b(
            "b", bType, bSizes.begin(), bSizes.end(), bStrides.begin(), bStrides.end());
        TensorDescriptor c(
            "c", cType, cSizes.begin(), cSizes.end(), cStrides.begin(), cStrides.end());
        TensorDescriptor d(
            "d", dType, dSizes.begin(), dSizes.end(), dStrides.begin(), dStrides.end());

        TensorDescriptor e("e");
        TensorDescriptor bias("bias");
        TensorDescriptor scaleD("scaleD");

        return ContractionProblemGemm(
            a, b, c, d, e, bias, scaleD, freeIndices, batchIndices, boundIndices, beta);
    }

    ContractionProblemGemm ContractionProblemGemm::GetDummy()
    {
        ContractionProblemGemm gemm;
        gemm.m_tensors[ContractionProblemGemm::TENSOR::A]      = TensorDescriptor("a");
        gemm.m_tensors[ContractionProblemGemm::TENSOR::B]      = TensorDescriptor("b");
        gemm.m_tensors[ContractionProblemGemm::TENSOR::C]      = TensorDescriptor("c");
        gemm.m_tensors[ContractionProblemGemm::TENSOR::D]      = TensorDescriptor("d");
        gemm.m_tensors[ContractionProblemGemm::TENSOR::E]      = TensorDescriptor("e");
        gemm.m_tensors[ContractionProblemGemm::TENSOR::BIAS]   = TensorDescriptor("bias");
        gemm.m_tensors[ContractionProblemGemm::TENSOR::SCALED] = TensorDescriptor("scaleD");
        return gemm;
    }

    ContractionProblemGemm::ContractionProblemGemm(TensorDescriptor const& a,
                                                   TensorDescriptor const& b,
                                                   TensorDescriptor const& c,
                                                   TensorDescriptor const& d,
                                                   TensorDescriptor const& e,
                                                   TensorDescriptor const& bias,
                                                   TensorDescriptor const& scaleD,
                                                   FreeIndices const&      freeIndices,
                                                   BatchIndices const&     batchIndices,
                                                   BoundIndices const&     boundIndices,
                                                   double                  beta,
                                                   size_t                  workspaceSize)
        : ContractionProblem(ContractionProblemGemm::TENSOR::TENSOR_COUNT)
        , m_freeIndices(freeIndices)
        , m_batchIndices(batchIndices)
        , m_boundIndices(boundIndices)
        , m_beta(beta)
    {
        m_workspaceSize                                   = workspaceSize;
        m_tensors[ContractionProblemGemm::TENSOR::A]      = a;
        m_tensors[ContractionProblemGemm::TENSOR::B]      = b;
        m_tensors[ContractionProblemGemm::TENSOR::C]      = c;
        m_tensors[ContractionProblemGemm::TENSOR::D]      = d;
        m_tensors[ContractionProblemGemm::TENSOR::E]      = e;
        m_tensors[ContractionProblemGemm::TENSOR::BIAS]   = bias;
        m_tensors[ContractionProblemGemm::TENSOR::SCALED] = scaleD;
        m_tensors[ContractionProblemGemm::TENSOR::D].setAsOutput(true); // Set d as output
        m_betaRestriction = toScalarValueEnum(
            m_beta); // Set enum using beta to potentially allow for faster solutions
        consistencyCheck();
        normalize();
    }

    size_t ContractionProblemGemm::toAPos(size_t idx) const
    {
        if(idx >= d().dimensions())
            return boundIndices().at(idx - d().dimensions()).a;

        auto found = std::find_if(
            freeIndicesA().begin(),
            freeIndicesA().end(),
            [idx](const ContractionProblemGemm::FreeIndex& fi) { return fi.d == idx; });
        assert(found != freeIndicesA().end());
        assert(found->isA);

        return found->i;
    }

    size_t ContractionProblemGemm::toBPos(size_t idx) const
    {
        if(idx >= d().dimensions())
            return boundIndices().at(idx - d().dimensions()).b;

        auto found = std::find_if(
            freeIndicesB().begin(),
            freeIndicesB().end(),
            [idx](const ContractionProblemGemm::FreeIndex& fi) { return fi.d == idx; });
        assert(found != freeIndicesB().end());
        assert(!found->isA);

        return found->i;
    }

    void ContractionProblemGemm::checkPersistentKernelEligibility(
        ContractionSolution const& solution, Hardware const& hardware)
    {
        m_eligibleForPK = true;

        // Get the new WorkGroup numbers under the PK and CU value
        auto sizeMapping = solution.sizeMapping;
        if(sizeMapping.persistentKernel == 0)
            return;

        // Get the normal WorkGroup numbers by sizeMapping MacroTile
        dim3 numWG(1, 1, 1);
        for(size_t i = 0; i < m_freeIndicesA.size(); i++)
        {
            numWG.x *= m_freeSizesA.at(i);
        }
        for(size_t i = 0; i < m_freeIndicesB.size(); i++)
        {
            numWG.y *= m_freeSizesB.at(i);
        }
        for(size_t i = 0; i < m_batchIndices.size(); i++)
        {
            if(sizeMapping.packBatchDims & 0x1)
                numWG.x *= m_batchSizes[i];
            if(sizeMapping.packBatchDims & 0x2)
                numWG.y *= m_batchSizes[i];
            if(!sizeMapping.packBatchDims)
                numWG.z *= m_batchSizes[i];
        }

        numWG.x = CeilDivide(numWG.x, sizeMapping.macroTile.x);
        numWG.y = CeilDivide(numWG.y, sizeMapping.macroTile.y);
        numWG.y *= sizeMapping.globalSplitU;

        size_t problemTiles = numWG.x * numWG.y;
        if(sizeMapping.persistentKernelAlongBatch)
            problemTiles *= numWG.z;

        AMDGPU const* pAMDGPU = dynamic_cast<AMDGPU const*>(&hardware);
        assert(pAMDGPU != nullptr && pAMDGPU->computeUnitCount != 0);

        size_t cuCount      = pAMDGPU->computeUnitCount;
        size_t finalPKValue = sizeMapping.persistentKernel;
        if(finalPKValue == -1)
        {
            // 1. Get the largest pk value (ex.3)
            //    which can make the PK.G (ex.3*120=360) <= problemGroups (ex.433)
            // 2. Scale by 5/8 (can try 0.5~1, to control the tiles-per-workgroup = 1~2)
            finalPKValue = 5 * (problemTiles / cuCount) / 8;
            finalPKValue = std::max(finalPKValue, (size_t)1);
        }

        size_t persistentGroups = cuCount * finalPKValue;

        // If #PKWG (PK*CUs) >= #TotalTiles, the persistent kernel behaves just like non-PK
        m_eligibleForPK = persistentGroups < problemTiles;
    }

    void ContractionProblemGemm::updateProblem(FreeIndices const&  freeIndices,
                                               BatchIndices const& batchIndices,
                                               BoundIndices const& boundIndices,
                                               double              beta,
                                               size_t              workspaceSize)
    {
        m_freeIndices     = freeIndices;
        m_batchIndices    = batchIndices;
        m_boundIndices    = boundIndices;
        m_beta            = beta;
        m_workspaceSize   = workspaceSize;
        m_betaRestriction = toScalarValueEnum(
            m_beta); // Set enum using beta to potentially allow for faster solutions
        consistencyCheck();
        normalize();
    }

    void ContractionProblemGemm::normalize()
    {
        auto& aTensor    = m_tensors[ContractionProblemGemm::TENSOR::A];
        auto& bTensor    = m_tensors[ContractionProblemGemm::TENSOR::B];
        auto& cTensor    = m_tensors[ContractionProblemGemm::TENSOR::C];
        auto& dTensor    = m_tensors[ContractionProblemGemm::TENSOR::D];
        auto& aNames     = m_names[ContractionProblemGemm::TENSOR::A];
        auto& bNames     = m_names[ContractionProblemGemm::TENSOR::B];
        auto& cNames     = m_names[ContractionProblemGemm::TENSOR::C];
        auto& dNames     = m_names[ContractionProblemGemm::TENSOR::D];
        m_maxProblemSize = 0;

        m_batchSizes.resize(m_batchIndices.size());
        m_boundSizes.resize(m_boundIndices.size());

        m_freeSizesA.clear();
        m_freeSizesB.clear();
        m_freeSizesA.reserve(m_freeIndices.size());
        m_freeSizesB.reserve(m_freeIndices.size());

        m_freeIndicesA.clear();
        m_freeIndicesB.clear();
        m_freeIndicesA.reserve(m_freeIndices.size());
        m_freeIndicesB.reserve(m_freeIndices.size());

        for(int i = 0; i < m_freeIndices.size(); i++)
        {
            size_t mySize = dTensor.sizes()[m_freeIndices[i].d];
            if(m_freeIndices[i].isA)
            {
                m_freeIndicesA.push_back(m_freeIndices[i]);
                m_freeSizesA.push_back(mySize);
            }
            else
            {
                m_freeIndicesB.push_back(m_freeIndices[i]);
                m_freeSizesB.push_back(mySize);
            }

            m_maxProblemSize = std::max(m_maxProblemSize, mySize);
        }

        for(int i = 0; i < m_batchIndices.size(); i++)
        {
            m_batchSizes[i] = std::max({aTensor.sizes()[m_batchIndices[i].a],
                                        bTensor.sizes()[m_batchIndices[i].b],
                                        cTensor.empty() ? 0 : cTensor.sizes()[m_batchIndices[i].c],
                                        dTensor.sizes()[m_batchIndices[i].d]});
        }

        for(int i = 0; i < m_boundIndices.size(); i++)
        {
            m_boundSizes[i] = std::max(aTensor.sizes()[m_boundIndices[i].a],
                                       bTensor.sizes()[m_boundIndices[i].b]);

            m_maxProblemSize = std::max(m_maxProblemSize, m_boundSizes[i]);
        }

        getIndexNames(aNames, bNames, cNames, dNames, m_sumNames);

        m_operationIdentifier = getOperationIdentifier();

        m_problemSizes.resize(0);
        m_problemSizes.reserve(cTensor.dimensions() + m_boundSizes.size());

        m_problemSizes.insert(m_problemSizes.end(), cTensor.sizes().begin(), cTensor.sizes().end());
        m_problemSizes.insert(m_problemSizes.end(), m_boundSizes.begin(), m_boundSizes.end());

        m_problemStrides.resize(0);
        m_problemStrides.reserve(aTensor.dimensions() + bTensor.dimensions() + cTensor.dimensions()
                                 + dTensor.dimensions());
        m_problemStrides.insert(
            m_problemStrides.end(), aTensor.strides().begin(), aTensor.strides().end());
        m_problemStrides.insert(
            m_problemStrides.end(), bTensor.strides().begin(), bTensor.strides().end());
        m_problemStrides.insert(
            m_problemStrides.end(), cTensor.strides().begin(), cTensor.strides().end());
        m_problemStrides.insert(
            m_problemStrides.end(), dTensor.strides().begin(), dTensor.strides().end());

        m_allocatedElementsNonBatchA = 1;
        for(int idx = 0; idx < a().dimensions(); idx++)
        {
            bool isBatch = m_batchIndices.end()
                           != std::find_if(m_batchIndices.begin(),
                                           m_batchIndices.end(),
                                           [idx](const ContractionProblemGemm::BatchIndex& bi) {
                                               return bi.a == idx;
                                           });
            if(!isBatch)
                m_allocatedElementsNonBatchA += aTensor.strides()[idx] * (aTensor.sizes()[idx] - 1);
        }

        m_allocatedElementsNonBatchB = 1;
        for(int idx = 0; idx < b().dimensions(); idx++)
        {
            bool isBatch = m_batchIndices.end()
                           != std::find_if(m_batchIndices.begin(),
                                           m_batchIndices.end(),
                                           [idx](const ContractionProblemGemm::BatchIndex& bi) {
                                               return bi.b == idx;
                                           });
            if(!isBatch)
                m_allocatedElementsNonBatchB += bTensor.strides()[idx] * (bTensor.sizes()[idx] - 1);
        }

        // CD always contain index0.  if this is in the B free indices, then need to
        // transposing the output tensor.
        m_transposeC01 = freeIndicesB().end()
                         != std::find_if(freeIndicesB().begin(),
                                         freeIndicesB().end(),
                                         [](const ContractionProblemGemm::FreeIndex& fi) {
                                             return fi.c == 0 /*idx0*/;
                                         });
    }

    void ContractionProblemGemm::consistencyCheck() const
    {
        auto& aTensor = m_tensors[ContractionProblemGemm::TENSOR::A];
        auto& bTensor = m_tensors[ContractionProblemGemm::TENSOR::B];
        auto& cTensor = m_tensors[ContractionProblemGemm::TENSOR::C];
        auto& dTensor = m_tensors[ContractionProblemGemm::TENSOR::D];

        std::vector<int> aUseCount(aTensor.dimensions(), 0);
        std::vector<int> bUseCount(bTensor.dimensions(), 0);
        std::vector<int> cUseCount(cTensor.dimensions(), 0);
        std::vector<int> dUseCount(dTensor.dimensions(), 0);

        for(FreeIndex const& free : m_freeIndices)
        {
            if(free.isA)
            {
                aUseCount[free.i]++;
                TENSILE_ASSERT_EXC(free.i < aTensor.dimensions());
                TENSILE_ASSERT_EXC(aTensor.sizes()[free.i] == dTensor.sizes()[free.d]);
            }
            else
            {
                bUseCount[free.i]++;
                TENSILE_ASSERT_EXC(free.i < bTensor.dimensions());
                TENSILE_ASSERT_EXC(bTensor.sizes()[free.i] == dTensor.sizes()[free.d]);
            }

            TENSILE_ASSERT_EXC(free.d < dTensor.dimensions());
            dUseCount[free.d]++;

            if(!cTensor.empty())
            {
                TENSILE_ASSERT_EXC(free.c < cTensor.dimensions());

                cUseCount[free.c]++;

                if(free.isA)
                    TENSILE_ASSERT_EXC(aTensor.sizes()[free.i] == cTensor.sizes()[free.c]);
                else
                    TENSILE_ASSERT_EXC(bTensor.sizes()[free.i] == cTensor.sizes()[free.c]);
            }
        }

        for(BatchIndex const& batch : m_batchIndices)
        {
            TENSILE_ASSERT_EXC(batch.a < aTensor.dimensions());
            TENSILE_ASSERT_EXC(batch.b < bTensor.dimensions());
            TENSILE_ASSERT_EXC(batch.d < dTensor.dimensions());

            aUseCount[batch.a]++;
            bUseCount[batch.b]++;
            dUseCount[batch.d]++;

            size_t aSize = aTensor.sizes()[batch.a];
            size_t bSize = bTensor.sizes()[batch.b];
            size_t cSize = 1;
            size_t dSize = dTensor.sizes()[batch.d];

            if(!cTensor.empty())
            {
                TENSILE_ASSERT_EXC(batch.c < cTensor.dimensions());
                cUseCount[batch.c]++;

                cSize = cTensor.sizes()[batch.c];
            }

            size_t indexSize = std::max({aSize, bSize, cSize, dSize});

            TENSILE_ASSERT_EXC(aSize == 1 || aSize == indexSize);
            TENSILE_ASSERT_EXC(bSize == 1 || bSize == indexSize);
            TENSILE_ASSERT_EXC(cSize == 1 || cSize == indexSize);
            TENSILE_ASSERT_EXC(dSize == 1 || dSize == indexSize);
        }

        for(BoundIndex const& bound : m_boundIndices)
        {
            TENSILE_ASSERT_EXC(bound.a < aTensor.dimensions());
            TENSILE_ASSERT_EXC(bound.b < bTensor.dimensions());

            aUseCount[bound.a]++;
            bUseCount[bound.b]++;

            TENSILE_ASSERT_EXC(aTensor.sizes()[bound.a] == bTensor.sizes()[bound.b]);
        }

        for(int aUse : aUseCount)
            TENSILE_ASSERT_EXC(aUse == 1);
        for(int bUse : bUseCount)
            TENSILE_ASSERT_EXC(bUse == 1);
        for(int cUse : cUseCount)
            TENSILE_ASSERT_EXC(cUse == 1);
        for(int dUse : dUseCount)
            TENSILE_ASSERT_EXC(dUse == 1);
    }

    size_t ContractionProblemGemm::freeSizeA(size_t idx) const
    {
        return m_freeSizesA.at(idx);
    }

    size_t ContractionProblemGemm::freeSizeB(size_t idx) const
    {
        return m_freeSizesB.at(idx);
    }

    size_t ContractionProblemGemm::batchSize(size_t idx) const
    {
        return m_batchSizes[idx];
    }

    size_t ContractionProblemGemm::boundSize(size_t idx) const
    {
        return m_boundSizes[idx];
    }

    size_t ContractionProblemGemm::size(size_t idx) const
    {
        if(idx < c().sizes().size())
            return c().sizes()[idx];
        else
            return m_boundSizes.at(idx - c().sizes().size());
    }

    size_t ContractionProblemGemm::flopsPerMac() const
    {
        auto& aTensor = m_tensors[ContractionProblemGemm::TENSOR::A];
        return 2 * DataTypeInfo::Get(aTensor.dataType()).packing;
    }

    size_t ContractionProblemGemm::flopCount() const
    {
        size_t rv = flopsPerMac();

        for(auto size : m_freeSizesA)
            rv *= size;

        for(auto size : m_freeSizesB)
            rv *= size;

        for(auto size : m_batchSizes)
            rv *= size;

        for(auto size : m_boundSizes)
            rv *= size;

        return rv;
    }

    void ContractionProblemGemm::getIndexNames(std::string& aNames,
                                               std::string& bNames,
                                               std::string& cNames,
                                               std::string& dNames,
                                               std::string& sumNames) const
    {
        auto& aTensor = m_tensors[ContractionProblemGemm::TENSOR::A];
        auto& bTensor = m_tensors[ContractionProblemGemm::TENSOR::B];
        auto& cTensor = m_tensors[ContractionProblemGemm::TENSOR::C];
        auto& dTensor = m_tensors[ContractionProblemGemm::TENSOR::D];

        aNames.resize(aTensor.dimensions(), '_');
        bNames.resize(bTensor.dimensions(), '_');
        cNames.resize(cTensor.dimensions(), '_');
        dNames.resize(dTensor.dimensions(), '_');
        sumNames.resize(m_boundIndices.size(), '_');

        char name = 'i';

        for(char& ch : dNames)
        {
            ch = name;
            name++;
        }

        for(char& ch : sumNames)
        {
            ch = name;
            name++;
        }

        for(auto const& free : m_freeIndices)
        {
            if(free.isA)
                aNames[free.i] = dNames[free.d];
            else
                bNames[free.i] = dNames[free.d];
            if(!cTensor.empty())
            {
                cNames[free.c] = dNames[free.d];
            }
        }

        for(auto const& batch : m_batchIndices)
        {
            aNames[batch.a] = dNames[batch.d];
            bNames[batch.b] = dNames[batch.d];
            if(!cTensor.empty())
                cNames[batch.c] = dNames[batch.d];
        }

        for(ptrdiff_t i = 0; i < sumNames.size(); i++)
        {
            auto const& boundIndex = m_boundIndices[i];
            aNames[boundIndex.a]   = boundIndex.aMirror ? std::toupper(sumNames[i]) : sumNames[i];
            bNames[boundIndex.b]   = boundIndex.bMirror ? std::toupper(sumNames[i]) : sumNames[i];
        }

        if(cTensor.empty() || m_beta == 0.0)
            cNames = dNames;
    }

    std::string ContractionProblemGemm::getOperationDescription() const
    {
        auto& cTensor = m_tensors[ContractionProblemGemm::TENSOR::C];
        auto& aNames  = m_names[ContractionProblemGemm::TENSOR::A];
        auto& bNames  = m_names[ContractionProblemGemm::TENSOR::B];
        auto& cNames  = m_names[ContractionProblemGemm::TENSOR::C];
        auto& dNames  = m_names[ContractionProblemGemm::TENSOR::D];

        std::ostringstream rv;

        rv << "D[" << dNames << "] = alpha * (";

        if(!m_sumNames.empty())
            rv << "Sum[" << m_sumNames << "] ";

        rv << "A[" << aNames << "] * B[" << bNames << "])";

        if(!cTensor.empty() && m_beta != 0)
        {
            rv << " + ";
            if(m_beta != 1.0)
                rv << "beta * ";
            rv << "C[" << cNames << "]";
        }

        return rv.str();
    }

    std::string ContractionProblemGemm::getOperationIdentifier() const
    {
        auto& aTensor = m_tensors[ContractionProblemGemm::TENSOR::A];
        auto& bTensor = m_tensors[ContractionProblemGemm::TENSOR::B];
        auto& cTensor = m_tensors[ContractionProblemGemm::TENSOR::C];
        auto& dTensor = m_tensors[ContractionProblemGemm::TENSOR::D];
        auto& aNames  = m_names[ContractionProblemGemm::TENSOR::A];
        auto& bNames  = m_names[ContractionProblemGemm::TENSOR::B];
        auto& cNames  = m_names[ContractionProblemGemm::TENSOR::C];
        auto& dNames  = m_names[ContractionProblemGemm::TENSOR::D];

        std::string rv = "Contraction_";
        rv += m_sumNames;
        rv += "_A";
        rv += aNames;
        if(DataTypeInfo::Get(aTensor.dataType()).isComplex)
        {
            rv += "C";
        }

        rv += "_B";
        rv += bNames;
        if(DataTypeInfo::Get(bTensor.dataType()).isComplex)
        {
            rv += "C";
        }

        rv += "_C";
        rv += cNames;
        if(DataTypeInfo::Get(cTensor.dataType()).isComplex)
        {
            rv += "C";
        }

        rv += "_D";
        rv += dNames;
        if(DataTypeInfo::Get(dTensor.dataType()).isComplex)
        {
            rv += "C";
        }

        return rv;
    }

    std::string ContractionProblemGemm::description() const
    {
        auto&              aTensor = m_tensors[ContractionProblemGemm::TENSOR::A];
        auto&              bTensor = m_tensors[ContractionProblemGemm::TENSOR::B];
        auto&              cTensor = m_tensors[ContractionProblemGemm::TENSOR::C];
        auto&              dTensor = m_tensors[ContractionProblemGemm::TENSOR::D];
        std::ostringstream rv;

        rv << operationIdentifier() << ",\n"
           << "A: " << aTensor << ",\n"
           << "B: " << bTensor << ",\n"
           << "C: " << cTensor << ",\n"
           << "D: " << dTensor << "\n";

        return rv.str();
    }

    TENSILE_API std::ostream& operator<<(std::ostream&                 stream,
                                         ContractionProblemGemm const& contraction)
    {
        return stream << contraction.description();
    }

    std::ostream& operator<<(std::ostream& stream, ContractionProblemGemm::FreeIndex const& free)
    {
        return stream << "{isA=" << free.isA << " i=" << free.i << " c=" << free.c
                      << " d=" << free.d << "}";
    }
    std::ostream& operator<<(std::ostream& stream, ContractionProblemGemm::BatchIndex const& batch)
    {
        if(batch.a == batch.b && batch.a == batch.c && batch.a == batch.d)
            return stream << "{" << batch.a << "}";

        return stream << "{a=" << batch.a << " b=" << batch.b << " c=" << batch.c
                      << " d=" << batch.d << "}";
    }

    std::ostream& operator<<(std::ostream& stream, ContractionProblemGemm::BoundIndex const& bound)
    {
        return stream << "{a=" << bound.a << " b=" << bound.b << "}";
    }

    std::istream& operator>>(std::istream& stream, ContractionProblemGemm::FreeIndex& free)
    {
        StreamRead comma(",");
        return stream >> free.isA >> comma >> free.i >> comma >> free.c >> comma >> free.d;
    }

    std::istream& operator>>(std::istream& stream, ContractionProblemGemm::BatchIndex& batch)
    {
        StreamRead comma(",");
        return stream >> batch.a >> comma >> batch.b >> comma >> batch.c >> comma >> batch.d;
    }

    std::istream& operator>>(std::istream& stream, ContractionProblemGemm::BoundIndex& bound)
    {
        StreamRead comma(",");
        return stream >> bound.a >> comma >> bound.b;
    }

    TENSILE_API ProblemInputs::~ProblemInputs() = default;
    ContractionInputs::ContractionInputs()      = default;
    ContractionInputs::~ContractionInputs()     = default;

    ContractionInputs::ContractionInputs(void const*        _a,
                                         void const*        _b,
                                         void const*        _c,
                                         void*              _d,
                                         void const* const* _batchA,
                                         void const* const* _batchB,
                                         void const* const* _batchC,
                                         void* const*       _batchD,
                                         void const*        _bias,
                                         void const*        _scaleD,
                                         void*              _ws)
        : a(_a)
        , b(_b)
        , c(_c)
        , d(_d)
        , batchA(_batchA)
        , batchB(_batchB)
        , batchC(_batchC)
        , batchD(_batchD)
        , bias(_bias)
        , scaleD(_scaleD)
        , ws(_ws)
    {
    }
} // namespace Tensile
