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

#pragma once

#include <Tensile/Activation.hpp>
#include <Tensile/KernelLanguageTypes.hpp>
#include <Tensile/PerformanceMetricTypes.hpp>
#include <Tensile/ScalarValueTypes.hpp>
#include <Tensile/Tensile.hpp>

#include <Tensile/ContractionProblem_fwd.hpp>
#include <Tensile/ContractionSolution_fwd.hpp>

#include <Tensile/TensorDescriptor.hpp>
#include <Tensile/Utils.hpp>

namespace Tensile
{
    /**
 * \addtogroup Problem
 * @{
 */
    class TENSILE_API ContractionProblemGemm;

    struct ConstantDescriptor
    {
        std::string name;
        DataType    dataType;
    };

    class ContractionProblem : public Problem
    {
    public:
        ContractionProblem(size_t size, size_t workspaceSize = 0);

        /**
         * Return vector of TensorDescriptor.
         */
        std::vector<TensorDescriptor> const& tensors() const
        {
            return m_tensors;
        }

        /**
         * Return a TensorDescriptor.
         */
        virtual TensorDescriptor const& tensor(int idx) const
        {
            return m_tensors[idx];
        }

        virtual void resizeTensor(int                           idx,
                                  std::initializer_list<size_t> sizes,
                                  std::initializer_list<size_t> strides)
        {
            m_tensors[idx].resize(sizes, strides);
        }

        /**
         * Return vector of constant datatype.
         */
        virtual std::vector<ConstantDescriptor> const constants() const = 0;

        void setWorkspaceSize(size_t size)
        {
            m_workspaceSize = size;
        }

        size_t workspaceSize() const
        {
            return m_workspaceSize;
        }

    protected:
        friend class ContractionProblemGemm;
        std::vector<TensorDescriptor> m_tensors;
        std::vector<std::string>      m_names;

        size_t m_workspaceSize;
    };

    /**
 * Describes a tensor contraction in by using TensorDescriptor objects for
 * each input or output tensor as well as indices describing transposes,
 * summations, etc. This is decoupled from any particular pointers, which
 * are provided in ContractionInputs objects.
 */
    class ContractionProblemGemm : public ContractionProblem
    {
    public:
        enum TENSOR : int
        {
            A      = 0,
            B      = 1,
            C      = 2,
            D      = 3,
            E      = 4,
            BIAS   = 5,
            SCALED = 6,
            TENSOR_COUNT
        };

        enum CONST : int
        {
            ALPHA    = 0,
            BETA     = 1,
            ACTALPHA = 2,
            ACTBETA  = 3,
            CONST_COUNT
        };

        using Solution = ContractionSolution;
        using Inputs   = ContractionInputs;

        ContractionProblemGemm()
            : ContractionProblem(ContractionProblemGemm::TENSOR::TENSOR_COUNT){};

        /**
   * Represents a pair of free indices in a tensor contraction.
   */
        struct FreeIndex
        {
            bool   isA; //< True=index is in A; False=index is in B
            size_t i; //< Dimension in A or B (depending on isA)
            size_t c; //< Dimension of C which corresponds for this index
            size_t d; //< Dimension of D which corresponds for this index
        };
        using FreeIndices = std::vector<FreeIndex>;

        /**
   * Represents a batched index in a tensor contraction.
   */
        struct BatchIndex
        {
            size_t a, b, c, d;
        };
        using BatchIndices = std::vector<BatchIndex>;

        /**
   * Represents a bound (or summed) index in a tensor contraction.
   */
        struct BoundIndex
        {
            BoundIndex(size_t xa = 0, size_t xb = 0, bool aMirror = false, bool bMirror = false)
                : a(xa)
                , b(xb)
                , aMirror(aMirror)
                , bMirror(bMirror){};
            size_t a, b; //! positions in a or b tensor
            bool   aMirror, bMirror;
        };
        using BoundIndices = std::vector<BoundIndex>;

        virtual std::string description() const;

        /**
   * Create a ContractionProblemGemm representing a batched GEMM, specifying
   * strides between matrices.
   */
        static ContractionProblemGemm GEMM_Strides(bool     transA,
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
                                                   double   beta);

        /**
   * Create a ContractionProblemGemm representing a batched SGEMM, with
   * leading dimensions, but no strides.
   */
        static ContractionProblemGemm GEMM(bool   transA,
                                           bool   transB,
                                           size_t m,
                                           size_t n,
                                           size_t k,
                                           size_t lda,
                                           size_t ldb,
                                           size_t ldc,
                                           double beta,
                                           bool   unused,
                                           size_t batchCount);

        /**
   * Create a ContractionProblemGemm representing a batched SGEMM, with
   * leading dimensions, but no strides.
   */
        static ContractionProblemGemm GEMM(bool   transA,
                                           bool   transB,
                                           size_t m,
                                           size_t n,
                                           size_t k,
                                           size_t lda,
                                           size_t offsetA,
                                           size_t ldb,
                                           size_t offsetB,
                                           size_t ldc,
                                           size_t offsetC,
                                           double beta,
                                           bool   unused,
                                           size_t batchCount);

        /**
   * Create a ContractionProblemGemm representing a batched GEMM based on the
   * dimensions of each of the tensors.
   */
        static ContractionProblemGemm GEMM(bool                    transA,
                                           bool                    transB,
                                           TensorDescriptor const& a,
                                           TensorDescriptor const& b,
                                           TensorDescriptor const& c,
                                           TensorDescriptor const& d,
                                           double                  beta);

        /**
   * Converts an identifier such as `Contraction_l_AlikC_Bjlk_Cijk_Dijk`
   * into a set of indices and operations.
   */
        static void IdentifierToIndices(std::string const& identifier,
                                        FreeIndices&       freeIndices,
                                        BatchIndices&      batchIndices,
                                        BoundIndices&      boundIndices,
                                        std::vector<bool>& isComplex);

        /**
   * Create a ContractionProblemGemm from a definition of each index, the
   * size of each index, the strides of each tensor, and any operations.
   *
   * @param freeIndices  Free indices
   * @param batchIndices Batch indices
   * @param boundIndices Bound indices
   * @param indexSizes   Size of each index, in the order of appearance in
   *                     the D tensor.
   *
   * @param aType    Data type of A
   * @param aStrides Strides of A
   * @param aOps     Operations to apply to A as it is read
   *
   * @param bType    Data type of B
   * @param bStrides Strides of B
   * @param bOps     Operations to apply to B as it is read
   *
   * @param cType    Data type of C
   * @param cStrides Strides of C
   * @param cOps     Operations to apply to C as it is read
   *
   * @param dType    Data type of D
   * @param dStrides Strides of D
   * @param dOps     Operations to apply to D as it is read
   *
   * @param beta Representative value of beta. Is only used to possibly
   *             select a more efficient kernel if we know that
   *             `beta == 0` or `beta == 1`.
   */
        static ContractionProblemGemm FromIndexSizes(FreeIndices const&         freeIndices,
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
                                                     double                     beta);

        /**
   * Create a ContractionProblemGemm based on an operation identifier such as
   * `Contraction_l_AlikC_Bjlk_Cijk_Dijk` and individual index sizes.
   *
   * @param operationIdentifier String that represents this exact
   *                            operation in terms of transposes, data
   *                            types, and operations.
   * @param indexSizes   Size of each index, in the order of appearance in
   *                     the D tensor.
   *
   * @param aType    Data type of A
   * @param aStrides Strides of A
   *
   * @param bType    Data type of B
   * @param bStrides Strides of B
   *
   * @param cType    Data type of C
   * @param cStrides Strides of C
   *
   * @param dType    Data type of D
   * @param dStrides Strides of D
   *
   * @param beta Representative value of beta. Is only used to possibly
   *             select a more efficient kernel if we know that
   *             `beta == 0` or `beta == 1`.
   */
        static ContractionProblemGemm FromIndexSizes(std::string const&         operationIdentifier,
                                                     std::vector<size_t> const& indexSizes,
                                                     DataType                   aType,
                                                     std::vector<size_t> const& aStrides,
                                                     DataType                   bType,
                                                     std::vector<size_t> const& bStrides,
                                                     DataType                   cType,
                                                     std::vector<size_t> const& cStrides,
                                                     DataType                   dType,
                                                     std::vector<size_t> const& dStrides,
                                                     double                     beta);

        /**
         * Create a dummy ContractionProblemGemm to get the information of the problem type.
         */
        static ContractionProblemGemm GetDummy();

        ContractionProblemGemm(TensorDescriptor const& a,
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
                               size_t                  workspaceSize = 0);

        //! Returns size given original index assignment (in range
        //! 0..NumIndicesC+boundSizes)
        size_t size(size_t idx) const;

        size_t freeSizeA(size_t idx) const;
        size_t freeSizeB(size_t idx) const;

        size_t batchSize(size_t idx) const;
        size_t boundSize(size_t idx) const;

        size_t toAPos(size_t idx) const;
        size_t toBPos(size_t idx) const;

        // Translate specified index into a position of that index in the d tensor.
        // Since d tensor order is always index order this is 1:1 translation if the
        // index is in-bounds:
        size_t toDPos(size_t idx) const
        {
            if(idx < d().dimensions())
                return idx;
            else
                throw std::runtime_error("requested index not in D");
        }

        size_t toBoundsPos(size_t idx) const
        {
            if(idx < d().dimensions())
                throw std::runtime_error("invalid bounds index (is free or batch)");
            else if(idx > d().dimensions() + boundIndices().size())
                throw std::runtime_error("invalid bounds index (out-of-bounds)");
            else
                return idx - d().dimensions();
        }

        std::vector<size_t> const& problemSizes() const
        {
            return m_problemSizes;
        }

        std::vector<size_t> const& problemStrides() const
        {
            return m_problemStrides;
        }

        void setCEqualsD(bool cEqualsD)
        {
            m_cEqualsD = cEqualsD;
        }

        bool cEqualsD() const
        {
            return m_cEqualsD;
        }

        void setAlphaType(DataType type)
        {
            m_alphaType = type;
        }

        DataType alphaType() const
        {
            return m_alphaType;
        }

        void setAlphaRestriction(ScalarValue alpha)
        {
            m_alphaRestriction = alpha;
        }

        ScalarValue alphaRestriction() const
        {
            return m_alphaRestriction;
        }

        void setBetaType(DataType type)
        {
            m_betaType = type;
        }

        DataType betaType() const
        {
            return m_betaType;
        }

        void setUseE(bool useE)
        {
            m_useE = useE;
        }

        void setUseBias(bool useBias)
        {
            m_useBias = useBias;
        }

        void setUseScaleD(bool useScaleD)
        {
            m_useScaleD = useScaleD;
        }

        bool useE() const
        {
            return m_useE;
        }

        bool useBias() const
        {
            return m_useBias;
        }

        bool useScaleD() const
        {
            return m_useScaleD;
        }

        void setE(DataType                   type,
                  std::vector<size_t> const& sizes,
                  std::vector<size_t> const& strides,
                  bool                       isOutput = false)
        {
            if(type != DataType::None && m_useE)
            {
                // Currently only supports offset = 0
                m_tensors[ContractionProblemGemm::TENSOR::E]
                    = {"e", type, sizes.begin(), sizes.end(), strides.begin(), strides.end()};
                m_tensors[ContractionProblemGemm::TENSOR::E].setAsOutput(isOutput);
            }
        }

        void setBias(DataType type, size_t length, bool isOutput = false)
        {
            m_biasType = type;
            if(type != DataType::None && m_useBias)
            {
                m_tensors[ContractionProblemGemm::TENSOR::BIAS]
                    = {"bias", m_biasType, {length}, {1, length}};
                m_tensors[ContractionProblemGemm::TENSOR::BIAS].setAsOutput(isOutput);
            }
        }

        DataType biasType() const
        {
            return m_biasType;
        }

        void setScaleD(DataType type, size_t length)
        {
            m_scaleDType = type;
            if(type != DataType::None && m_useScaleD)
            {
                m_tensors[ContractionProblemGemm::TENSOR::SCALED]
                    = {"scaleD", m_scaleDType, {length}, {1, length}};
            }
        }

        void setBetaRestriction(ScalarValue beta)
        {
            m_betaRestriction = beta;
        }

        ScalarValue betaRestriction() const
        {
            return m_betaRestriction;
        }

        void setStridedBatched(bool value)
        {
            m_stridedBatched = value;
        }

        bool stridedBatched() const
        {
            return m_stridedBatched;
        }

        void setGroupedGemm(bool value)
        {
            m_groupedGemm = value;
        }

        bool groupedGemm() const
        {
            return m_groupedGemm;
        }

        void setHighPrecisionAccumulate(bool value)
        {
            m_highPrecisionAccumulate = value;
        }

        bool highPrecisionAccumulate() const
        {
            return m_highPrecisionAccumulate;
        }

        void setKernelLanguage(KernelLanguage value)
        {
            m_kernelLanguage = value;
        }
        KernelLanguage kernelLanguage() const
        {
            return m_kernelLanguage;
        }

        void setPerformanceMetric(PerformanceMetric value)
        {
            m_performanceMetric = value;
        }

        PerformanceMetric performanceMetric() const
        {
            const bool experimental = Debug::Instance().useExperimentalSelection();
            return experimental ? PerformanceMetric::Experimental : m_performanceMetric;
        }

        void setDeterministicMode(bool value)
        {
            m_deterministicMode = value;
        }
        bool deterministicMode() const
        {
            return m_deterministicMode;
        }

        void setFp16AltImpl(bool value)
        {
            m_fp16AltImpl = value;
        }

        bool fp16AltImpl() const
        {
            return m_fp16AltImpl;
        }

        void setUseGradient(bool value)
        {
            m_useGradient = value;
        }

        bool useGradient() const
        {
            return m_useGradient;
        }

        void setActivationType(ActivationType activationtype)
        {
            m_activationType = activationtype;
        }

        ActivationType activationType() const
        {
            return m_activationType;
        }

        void setActivationHPA(bool value)
        {
            m_activationHPA = value;
        }

        bool activationHPA() const
        {
            return m_activationHPA;
        }

        void setActivationGuard(bool value)
        {
            m_activationGuard = value;
        }

        bool activationGuard() const
        {
            return m_activationGuard;
        }

        void setActivationEnumArg(ActivationType activationEnumArg)
        {
            m_activationEnumArg = activationEnumArg;
        }

        ActivationType activationEnumArg() const
        {
            return m_activationEnumArg;
        }

        /// Largest of the free and bound indices.  Does not include batch size.
        size_t maxProblemSize() const
        {
            return m_maxProblemSize;
        }

        /// Allocated elements excluding batch dimensions
        /// Used in assembly kernels to determine buffer limits, if batch dimes not
        /// packed
        size_t allocatedElementsNonBatchA() const
        {
            return m_allocatedElementsNonBatchA;
        }
        size_t allocatedElementsNonBatchB() const
        {
            return m_allocatedElementsNonBatchB;
        }

        size_t flopsPerMac() const;
        size_t flopCount() const;

        TensorDescriptor const& a() const
        {
            return m_tensors[ContractionProblemGemm::TENSOR::A];
        }
        TensorDescriptor const& b() const
        {
            return m_tensors[ContractionProblemGemm::TENSOR::B];
        }
        TensorDescriptor const& c() const
        {
            return m_tensors[ContractionProblemGemm::TENSOR::C];
        }
        TensorDescriptor const& d() const
        {
            return m_tensors[ContractionProblemGemm::TENSOR::D];
        }

        FreeIndices const& freeIndicesA() const
        {
            return m_freeIndicesA;
        }
        FreeIndices const& freeIndicesB() const
        {
            return m_freeIndicesB;
        }
        FreeIndices const& freeIndices() const
        {
            return m_freeIndices;
        }
        BatchIndices const& batchIndices() const
        {
            return m_batchIndices;
        }
        BoundIndices const& boundIndices() const
        {
            return m_boundIndices;
        }

        bool transposeC01() const
        {
            return m_transposeC01;
        };

        double beta() const
        {
            return m_beta;
        }

        std::string const& aNames() const
        {
            return m_names[ContractionProblemGemm::TENSOR::A];
        }
        std::string const& bNames() const
        {
            return m_names[ContractionProblemGemm::TENSOR::B];
        }
        std::string const& cNames() const
        {
            return m_names[ContractionProblemGemm::TENSOR::C];
        }
        std::string const& dNames() const
        {
            return m_names[ContractionProblemGemm::TENSOR::D];
        }
        std::string const& sumNames() const
        {
            return m_sumNames;
        }

        bool transA() const
        {
            return m_names[ContractionProblemGemm::TENSOR::A] == "lik";
        }
        bool transB() const
        {
            return m_names[ContractionProblemGemm::TENSOR::B] == "jlk";
        }

        std::string        operationName() const;
        std::string const& operationIdentifier() const
        {
            return m_operationIdentifier;
        }
        std::string operationDescription() const
        {
            return getOperationDescription();
        }

        void checkPersistentKernelEligibility(ContractionSolution const& solution,
                                              Hardware const&            hardware);

        bool getPersistentKernelEligibility() const
        {
            return m_eligibleForPK;
        }

        virtual std::vector<ConstantDescriptor> const constants() const
        {
            std::vector<ConstantDescriptor> c = {{"alpha", m_alphaType}, {"beta", m_betaType}};

            size_t                   num    = getAdditionalArgNum(activationType());
            size_t                   numAll = getAdditionalArgNum(ActivationType::All);
            std::vector<std::string> s      = generateArgNameList(numAll, "activation");
            size_t                   i      = 0;
            if(m_activationHPA)
            {
                for(i = 0; i < num; i++)
                    c.push_back({s[i], m_betaType});
            }
            else
            {
                for(i = 0; i < num; i++)
                    c.push_back({s[i], d().dataType()});
            }
            // Push the rest of the args even unused.
            for(; i < numAll; i++)
                c.push_back({s[i], DataType::None});

            return c;
        }

    private:
        std::string m_sumNames;
        std::string m_operationIdentifier;

        bool              m_cEqualsD                = false;
        bool              m_stridedBatched          = true;
        bool              m_groupedGemm             = false;
        bool              m_highPrecisionAccumulate = false;
        bool              m_deterministicMode       = false;
        bool              m_eligibleForPK           = true;
        bool              m_fp16AltImpl             = false;
        bool              m_useGradient             = false;
        bool              m_useE                    = false;
        bool              m_useBias                 = false;
        bool              m_useScaleD               = false;
        ActivationType    m_activationType          = ActivationType::None;
        ActivationType    m_activationEnumArg       = ActivationType::None;
        bool              m_activationHPA           = false;
        bool              m_activationGuard         = false;
        KernelLanguage    m_kernelLanguage          = KernelLanguage::Any;
        PerformanceMetric m_performanceMetric       = PerformanceMetric::DeviceEfficiency;

        DataType m_alphaType  = DataType::None; // if not assigned, will follow d-type
        DataType m_betaType   = DataType::None; // for bwd-compatible
        DataType m_biasType   = DataType::None;
        DataType m_scaleDType = DataType::None; // if not assigned, will follow alpha-type

        ScalarValue m_alphaRestriction = ScalarValue::Any; // restrictions on the alpha value used
        ScalarValue m_betaRestriction  = ScalarValue::Any; // restrictions on the beta value used

        FreeIndices  m_freeIndicesA; //< in same order as IndexAssignmentsA
        FreeIndices  m_freeIndicesB; //< in same order as IndexAssignmentsB
        FreeIndices  m_freeIndices;
        BatchIndices m_batchIndices;
        BoundIndices m_boundIndices;

        std::vector<size_t> m_freeSizesA;
        std::vector<size_t> m_freeSizesB;
        std::vector<size_t> m_batchSizes;
        std::vector<size_t> m_boundSizes;

        std::vector<size_t> m_problemSizes;
        std::vector<size_t> m_problemStrides;

        bool   m_transposeC01;
        double m_beta;

        size_t m_maxProblemSize = 1;

        size_t m_allocatedElementsNonBatchA;
        size_t m_allocatedElementsNonBatchB;

        void normalize();
        void consistencyCheck() const;

        void getIndexNames(std::string& aNames,
                           std::string& bNames,
                           std::string& cNames,
                           std::string& dNames,
                           std::string& sumNames) const;

        std::string getOperationIdentifier() const;
        std::string getOperationDescription() const;
    };

    class ContractionProblemGroupedGemm : public ContractionProblem
    {
    public:
        ContractionProblemGroupedGemm()
            : ContractionProblem(0){};
        std::vector<ContractionProblemGemm> gemms;
        virtual std::string                 description() const
        {
            throw std::runtime_error("Get the information from gemms[idx].description() instead.");
        }
        virtual std::vector<ConstantDescriptor> const constants() const
        {
            throw std::runtime_error("Get the information from gemms[idx].constants() instead.");
        }
    };

    struct TENSILE_API ContractionInputs : public ProblemInputs
    {
        ContractionInputs();
        virtual ~ContractionInputs();

        ContractionInputs(void const*        _a,
                          void const*        _b,
                          void const*        _c,
                          void*              _d,
                          void const* const* _batchA,
                          void const* const* _batchB,
                          void const* const* _batchC,
                          void* const*       _batchD,
                          void const*        _bias,
                          void const*        _scaleD,
                          void*              _ws);

        // TODO: Remove this
        void const* a = nullptr;
        void const* b = nullptr;
        void const* c = nullptr;
        void*       d = nullptr;
        void*       e = nullptr;

        void const* const* batchA = nullptr;
        void const* const* batchB = nullptr;
        void const* const* batchC = nullptr;
        void* const*       batchD = nullptr;

        void const* bias   = nullptr;
        void const* scaleD = nullptr;

        // Constants
        ConstantVariant              alpha = static_cast<float>(0);
        ConstantVariant              beta  = static_cast<float>(0);
        std::vector<ConstantVariant> activationArgs;

        // Workspace
        void* ws = nullptr;

        std::vector<size_t> maxElements;
        size_t              workspaceSize;
        bool                gpu = false;
    };

    struct TENSILE_API ContractionGroupedInputs : public ProblemInputs
    {
        std::vector<ContractionInputs> grouped;
    };

    // Deprecated: Legacy support for hipBLASLt
    template <typename A     = float,
              typename B     = A,
              typename C     = A,
              typename D     = C,
              typename Alpha = D,
              typename Beta  = D>
    struct TypedContractionInputs : ContractionInputs
    {
    };

    TENSILE_API std::ostream& operator<<(std::ostream&                 stream,
                                         ContractionProblemGemm const& contraction);

    TENSILE_API std::ostream& operator<<(std::ostream&                            stream,
                                         ContractionProblemGemm::FreeIndex const& free);
    TENSILE_API std::ostream& operator<<(std::ostream&                             stream,
                                         ContractionProblemGemm::BatchIndex const& batch);
    TENSILE_API std::ostream& operator<<(std::ostream&                             stream,
                                         ContractionProblemGemm::BoundIndex const& bound);

    TENSILE_API std::istream& operator>>(std::istream&                      stream,
                                         ContractionProblemGemm::FreeIndex& free);
    TENSILE_API std::istream& operator>>(std::istream&                       stream,
                                         ContractionProblemGemm::BatchIndex& batch);
    TENSILE_API std::istream& operator>>(std::istream&                       stream,
                                         ContractionProblemGemm::BoundIndex& bound);

    /**
 * @}
 */
} // namespace Tensile
