/* ************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

// The implementation of the rocblaslt<->Tensile interface layer.

#include "rocblaslt.h"

/*****************************************************************************
 * This is the only file in rocblaslt which should #include Tensile headers    *
 * or reference Tensile identifiers. tensile_host.hpp defines the interface. *
 *****************************************************************************/

#include "rocblaslt-types.h"
#include "rocblaslt_mat_utils.hpp"
#include "tensile_host.hpp"

//#include <Tensile/AMDGPU.hpp>
#include <Tensile/Contractions.hpp>
#include <Tensile/EmbeddedLibrary.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/TensorDescriptor.hpp>
#include <Tensile/Utils.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/hip/HipUtils.hpp>
#include <atomic>
#include <complex>
#include <exception>
#include <iomanip>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

#include <glob.h>
#include <libgen.h>
#include <link.h>
#include <unistd.h>

#define HIPBLASLT_LIB_PATH "/opt/rocm/hipblaslt/lib"

#ifdef ENABLE_ROCTX
#include <roctracer/roctx.h>
#endif

#define INTERNAL_HIPHOSTMEM_SIZE 32768

namespace
{
    std::string getHipblasltSoPath()
    {
        return rocblaslt_internal_get_so_path("libhipblaslt");
    }
    /******************************************************
 * Map a rocblaslt type to a corresponding Tensile type *
 ******************************************************/
    template <typename T>
    struct rocblaslt_to_tensile_type
    {
        using tensile_type = T;
    };

    template <>
    struct rocblaslt_to_tensile_type<rocblaslt_half>
    {
        using tensile_type = Tensile::Half;
    };

    template <>
    struct rocblaslt_to_tensile_type<rocblaslt_bfloat16>
    {
        using tensile_type = Tensile::BFloat16;
    };

    template <>
    struct rocblaslt_to_tensile_type<rocblaslt_f8>
    {
        using tensile_type = Tensile::Float8;
    };

    template <>
    struct rocblaslt_to_tensile_type<rocblaslt_bf8>
    {
        using tensile_type = Tensile::BFloat8;
    };

    template <>
    struct rocblaslt_to_tensile_type<rocblasltInt8>
    {
        using tensile_type = Tensile::Int8;
    };
    /********************************************************************
 * Variable template to map a rocblaslt type into a Tensile::DataType *
 ********************************************************************/
    template <typename>
    constexpr auto tensile_datatype = nullptr;

    template <>
    constexpr auto tensile_datatype<rocblaslt_half> = Tensile::DataType::Half;

    template <>
    constexpr auto tensile_datatype<float> = Tensile::DataType::Float;

    template <>
    constexpr auto tensile_datatype<double> = Tensile::DataType::Double;

    template <>
    constexpr auto tensile_datatype<rocblaslt_bfloat16> = Tensile::DataType::BFloat16;

    template <>
    constexpr auto tensile_datatype<rocblaslt_f8> = Tensile::DataType::Float8;

    template <>
    constexpr auto tensile_datatype<rocblaslt_bf8> = Tensile::DataType::BFloat8;

    template <>
    constexpr auto tensile_datatype<rocblasltInt8> = Tensile::DataType::Int8;

    template <>
    constexpr auto tensile_datatype<int32_t> = Tensile::DataType::Int32;

    /*************************************************************************
 * Class for converting alpha and beta between rocblaslt and Tensile types *
 * By default, alpha and beta are the same type as Tc compute_type       *
 *************************************************************************/
    template <typename TiA, typename TiB = TiA, typename To = TiB, typename Tc = To>
    struct AlphaBeta
    {
        using tensile_type = typename rocblaslt_to_tensile_type<Tc>::tensile_type;
        static void copy(tensile_type* dst, const Tc* src)
        {
            static_assert(sizeof(*src) == sizeof(*dst),
                          "Tensile and rocblaslt types are not the same size");
            static_assert(std::is_standard_layout<tensile_type>{} && std::is_standard_layout<Tc>{},
                          "Tensile or rocblaslt types are not standard layout types");
            memcpy(dst, src, sizeof(*dst));
        }
    };

    inline Tensile::ActivationType getTensileActivationType(rocblaslt_epilogue epilogue)
    {
        switch(epilogue)
        {
        case ROCBLASLT_EPILOGUE_RELU:
        case ROCBLASLT_EPILOGUE_RELU_BIAS:
            return Tensile::ActivationType::Relu;
            break;
        case ROCBLASLT_EPILOGUE_GELU:
        case ROCBLASLT_EPILOGUE_GELU_BIAS:
        case ROCBLASLT_EPILOGUE_GELU_AUX:
        case ROCBLASLT_EPILOGUE_GELU_AUX_BIAS:
            return Tensile::ActivationType::Gelu;
            break;
        case ROCBLASLT_EPILOGUE_DGELU:
        case ROCBLASLT_EPILOGUE_DGELU_BGRAD:
            return Tensile::ActivationType::DGelu;
        case ROCBLASLT_EPILOGUE_BIAS:
        case ROCBLASLT_EPILOGUE_DEFAULT:
        case ROCBLASLT_EPILOGUE_BGRADA:
        case ROCBLASLT_EPILOGUE_BGRADB:
            break;
        }
        return Tensile::ActivationType::None;
    }

    inline Tensile::ContractionProblemGemm::TENSOR getBiasSrc(rocblaslt_epilogue epilogue)
    {
        switch(epilogue)
        {
        case ROCBLASLT_EPILOGUE_BGRADA:
            return Tensile::ContractionProblemGemm::TENSOR::A;
            break;
        case ROCBLASLT_EPILOGUE_BGRADB:
            return Tensile::ContractionProblemGemm::TENSOR::B;
            break;
        default:
            break;
        }
        return Tensile::ContractionProblemGemm::TENSOR::D;
    }

    Tensile::DataType hip2TensileType(hipblasltDatatype_t type)
    {
        switch(type)
        {
        case HIPBLASLT_R_32F:
            return Tensile::DataType::Float;
        case HIPBLASLT_R_16F:
            return Tensile::DataType::Half;
        case HIPBLASLT_R_64F:
            return Tensile::DataType::Double;
        case HIPBLASLT_R_16B:
            return Tensile::DataType::BFloat16;
        case HIPBLASLT_R_8F_E4M3:
            return Tensile::DataType::Float8;
        case HIPBLASLT_R_8F_E5M2:
            return Tensile::DataType::BFloat8;
        case HIPBLASLT_R_8I:
            return Tensile::DataType::Int8;
        case HIPBLASLT_R_32I:
            return Tensile::DataType::Int32;
        default:
            throw std::runtime_error("Unsupported type.");
        }
        return Tensile::DataType::None;
    }

    hipblasltDatatype_t tensile2HipType(Tensile::DataType type)
    {
        switch(type)
        {
        case Tensile::DataType::Float:
            return HIPBLASLT_R_32F;
        case Tensile::DataType::Half:
            return HIPBLASLT_R_16F;
        case Tensile::DataType::Double:
            return HIPBLASLT_R_64F;
        case Tensile::DataType::BFloat16:
            return HIPBLASLT_R_16B;
        case Tensile::DataType::Float8:
            return HIPBLASLT_R_8F_E4M3;
        case Tensile::DataType::BFloat8:
            return HIPBLASLT_R_8F_E5M2;
        case Tensile::DataType::Int8:
            return HIPBLASLT_R_8I;
        case Tensile::DataType::Int32:
            return HIPBLASLT_R_32I;
        default:
            throw std::runtime_error("Unsupported type.");
        }
        return HIPBLASLT_R_32F;
    }

    Tensile::DataType roc2TensileType(rocblaslt_compute_type type)
    {
        switch(type)
        {
        case rocblaslt_compute_f32:
            return Tensile::DataType::Float;
        case rocblaslt_compute_f32_fast_xf32:
            return Tensile::DataType::XFloat32;
        case rocblaslt_compute_f64:
            return Tensile::DataType::Double;
        case rocblaslt_compute_i32:
            return Tensile::DataType::Int32;
        case rocblaslt_compute_f32_fast_f16:
            return Tensile::DataType::Float;
        default:
            throw std::runtime_error("Unsupported type.");
        }
        return Tensile::DataType::None;
    }

    inline const Tensile::DataType
        roc2TensileComputeInputType(const Tensile::DataType&      typeA,
                                    const Tensile::DataType&      typeB,
                                    const rocblaslt_compute_type& typeCompute)
    {
        switch(typeCompute)
        {
        case rocblaslt_compute_f32_fast_f16:
            return Tensile::DataType::Half;
        default:;
        }

        if(typeA == Tensile::DataType::Float8 && typeB == Tensile::DataType::BFloat8)
        {
            return Tensile::DataType::Float8BFloat8;
        }
        else if(typeA == Tensile::DataType::BFloat8 && typeB == Tensile::DataType::Float8)
        {
            return Tensile::DataType::BFloat8Float8;
        }
        return Tensile::DataTypeInfo::Get(typeA).elementSize
                       <= Tensile::DataTypeInfo::Get(typeB).elementSize
                   ? typeA
                   : typeB;
    }

    inline auto CreateTensileProblem(hipblasOperation_t     opA,
                                     hipblasOperation_t     opB,
                                     hipblasltDatatype_t    typeA,
                                     hipblasltDatatype_t    typeB,
                                     hipblasltDatatype_t    typeC,
                                     hipblasltDatatype_t    typeD,
                                     rocblaslt_compute_type typeCompute,
                                     float                  alpha,
                                     float                  beta,
                                     bool                   isGroupedGemm,
                                     size_t                 maxWorkspaceBytes)
    {
        auto typeATensile = hip2TensileType(typeA);
        auto typeBTensile = hip2TensileType(typeB);
        return Tensile::ContractionProblemGemm::createDefaultProblem(
            (opA != HIPBLAS_OP_N),
            (opB != HIPBLAS_OP_N),
            typeATensile,
            typeBTensile,
            hip2TensileType(typeC),
            hip2TensileType(typeD),
            roc2TensileType(typeCompute),
            roc2TensileType(typeCompute),
            roc2TensileComputeInputType(typeATensile, typeBTensile, typeCompute),
            roc2TensileType(typeCompute),
            alpha,
            beta,
            isGroupedGemm,
            maxWorkspaceBytes);
    }

    /****************************************************************
 * Construct a Tensile Problem from a RocblasltContractionProblem *
 ****************************************************************/
    // Should remove this
    template <typename TiA, typename TiB, typename To, typename Tc>
    auto ConstructTensileProblem(const RocblasltContractionProblem<TiA, TiB, To, Tc>& prob)
    {
        // Tensile DataTypes corresponding to rocblaslt data types
        static constexpr Tensile::DataType Tensile_TiA = tensile_datatype<TiA>;
        static constexpr Tensile::DataType Tensile_TiB = tensile_datatype<TiB>;
        static constexpr Tensile::DataType Tensile_To  = tensile_datatype<To>;
        static constexpr Tensile::DataType Tensile_Tc  = tensile_datatype<Tc>;

        // Tensor descriptors for a, b
        Tensile::TensorDescriptor a, b;

        // Tensile Indices for contraction problem
        Tensile::ContractionProblemGemm::FreeIndices  freeIndex(2);
        Tensile::ContractionProblemGemm::BoundIndices boundIndex(1);
        Tensile::ContractionProblemGemm::BatchIndices batchIndex{{2, 2, 2, 2}};

        // Set up GEMM indices
        freeIndex[0].isA = true;
        freeIndex[1].isA = false;
        freeIndex[0].c = freeIndex[0].d = 0;
        freeIndex[1].c = freeIndex[1].d = 1;

        // We set K=0 when alpha==0.
        // This makes alpha==0 a change in the problem, and not just a change in the
        // inputs. It optimizes all problems with alpha==0 into K=0 and alpha=(don't
        // care)
        auto k = prob.k && *prob.alpha ? prob.k : 0;

        // clang-format off

        // If A is transposed, swap the free and bound dimensions and their ranks
        if(prob.trans_a != HIPBLAS_OP_N)
        {
            a = {
                    "a",
                    Tensile_TiA,
                    {k, prob.m, prob.batch_count},
                    {prob.row_stride_a, prob.col_stride_a, prob.batch_stride_a}
                };
            freeIndex[0].i  = 1;
            boundIndex[0].a = 0;
        }
        else
        {
            a = {
                    "a",
                    Tensile_TiA,
                    {prob.m, k, prob.batch_count},
                    {prob.row_stride_a, prob.col_stride_a, prob.batch_stride_a}
                };
            freeIndex[0].i  = 0;
            boundIndex[0].a = 1;
        }

        // If B is transposed, swap the free and bound dimensions and their ranks
        if(prob.trans_b != HIPBLAS_OP_N)
        {
            b = {
                    "b",
                    Tensile_TiB,
                    {prob.n, k, prob.batch_count},
                    {prob.row_stride_b, prob.col_stride_b, prob.batch_stride_b}
                };
            freeIndex[1].i  = 0;
            boundIndex[0].b = 1;
        }
        else
        {
            b = {
                    "b",
                    Tensile_TiB,
                    {k, prob.n, prob.batch_count},
                    {prob.row_stride_b, prob.col_stride_b, prob.batch_stride_b}
                };
            freeIndex[1].i  = 1;
            boundIndex[0].b = 0;
        }

        // clang-format on

        // Descriptor for input matrix C
        Tensile::TensorDescriptor c{"c",
                                    Tensile_To,
                                    {prob.m, prob.n, prob.batch_count},
                                    {prob.row_stride_c, prob.col_stride_c, prob.batch_stride_c}};

        // Descriptor for output matrix D
        Tensile::TensorDescriptor d{"d",
                                    Tensile_To,
                                    {prob.m, prob.n, prob.batch_count},
                                    {prob.row_stride_d, prob.col_stride_d, prob.batch_stride_d}};

        Tensile::TensorDescriptor e{"e"};
        Tensile::TensorDescriptor bias{"bias"};
        Tensile::TensorDescriptor scaleA{"scaleA"};
        Tensile::TensorDescriptor scaleB{"scaleB"};
        Tensile::TensorDescriptor scaleC{"scaleC"};
        Tensile::TensorDescriptor scaleD{"scaleD"};
        //remove once scaleD is removed from tensile
        Tensile::TensorDescriptor scaleDVec{"scaleDVec"};
        Tensile::TensorDescriptor scaleAlphaVec{"scaleAlphaVec"};

        // The ContractionProblemGemm
        Tensile::ContractionProblemGemm tensileProblem{a,
                                                       b,
                                                       c,
                                                       d,
                                                       e,
                                                       bias,
                                                       scaleA,
                                                       scaleB,
                                                       scaleC,
                                                       scaleD,
                                                       scaleDVec,
                                                       scaleAlphaVec,
                                                       freeIndex,
                                                       batchIndex,
                                                       boundIndex,
                                                       value_category(*prob.beta),
                                                       prob.workspaceSize};

        tensileProblem.setComputeInputType(
            roc2TensileComputeInputType(Tensile_TiA, Tensile_TiB, prob.compute_type));
        tensileProblem.setAlphaType(Tensile_Tc);
        tensileProblem.setBetaType(Tensile_Tc);

        // HPA is active iff sizeof(compute type) > sizeof(input type)
        tensileProblem.setHighPrecisionAccumulate(sizeof(Tc) > sizeof(TiA));

        // set batch mode
        tensileProblem.setStridedBatched(prob.strided_batch);
        tensileProblem.setGroupedGemm(prob.grouped_gemm);
        // FIXME: Hardcoded, fix prob.trans_b == HIPBLAS_OP_N when GSU is supported
        if(prob.grouped_gemm && prob.trans_b == HIPBLAS_OP_N)
            tensileProblem.setUseDeviceUserArguments(true);
        else
            tensileProblem.setUseDeviceUserArguments(false);

        // alpha and beta are stored by value in Tensile::TypedContractionInputs
        // alpha and beta are copied from host to Tensile::TypedContractionInputs
        // If k==0, we do not need to dereference prob.alpha and can set
        // tensileAlpha=0 Not positive if this is necessary here as well
        typename AlphaBeta<TiA, TiB, To, Tc>::tensile_type tensileAlpha;
        if(prob.k)
            AlphaBeta<TiA, TiB, To, Tc>::copy(&tensileAlpha, prob.alpha);
        else
            memset(&tensileAlpha, 0, sizeof(tensileAlpha));
        tensileProblem.setAlphaRestriction(Tensile::toScalarValueEnum(tensileAlpha));

        // Add problem predicates for CEqualsD
        tensileProblem.setCEqualsD(prob.C == prob.D);

        if(is_e_enabled(prob.epilogue))
        {
            bool isOutput = prob.gradient ? false : true;
            tensileProblem.setUseE(true);
            tensileProblem.setE(Tensile_Tc,
                                {prob.m, prob.n, prob.batch_count},
                                {prob.row_stride_e, prob.col_stride_e, prob.batch_stride_e},
                                isOutput);
        }

        // set bias mode
        auto biasSrc = getBiasSrc(prob.epilogue);
        auto biasSize
            = (biasSrc == Tensile::ContractionProblemGemm::TENSOR::B) ? d.sizes()[1] : d.sizes()[0];
        tensileProblem.setUseBias(true);
        tensileProblem.setBias(
            hipblasltDatatype_to_tensile_type(prob.bias_type), biasSize, 0, prob.gradient, biasSrc);

        // ScaleAB is only supported on F8/BF8
        if(Tensile_TiA == Tensile::DataType::Float8 || Tensile_TiA == Tensile::DataType::BFloat8
           || Tensile_TiB == Tensile::DataType::Float8 || Tensile_TiB == Tensile::DataType::BFloat8)
        {
            tensileProblem.setUseScaleAB(true);
            tensileProblem.setScaleA(Tensile_Tc);
            tensileProblem.setScaleB(Tensile_Tc);
            tensileProblem.setUseScaleAlphaVec(true);
            tensileProblem.setScaleAlphaVec(Tensile_Tc, d.sizes()[0]);
        }
        else
        {
            tensileProblem.setUseScaleAB(false);
            // set ScaleAlphaVec mode
            tensileProblem.setUseScaleAlphaVec(true);
            tensileProblem.setScaleAlphaVec(Tensile_Tc, d.sizes()[0]);
        }

        // set Actvation
        // only bias src A/B cannot enabled Act
        if(!is_biasSrc_AB(prob.epilogue))
        {
            tensileProblem.setActivationType(Tensile::ActivationType::All);
            tensileProblem.setActivationComputeType(Tensile_Tc);
            tensileProblem.setActivationEnumArg(getTensileActivationType(prob.epilogue));
        }
        else
        {
            tensileProblem.setActivationType(Tensile::ActivationType::None);
        }

        // set use gradient
        tensileProblem.setUseGradient(is_grad_enabled(prob.epilogue));

        if constexpr(std::is_same<TiA, float>{} && std::is_same<To, float>{}
                     && std::is_same<Tc, float>{})
            if(prob.compute_type == rocblaslt_compute_f32_fast_xf32)
                tensileProblem.setF32XdlMathOp(Tensile::DataType::XFloat32);

        return tensileProblem;
    }

    template <typename TiA, typename TiB, typename To, typename Tc>
    void updateTensileProblem(const bool                                           fallback,
                              const RocblasltContractionProblem<TiA, TiB, To, Tc>& prob,
                              Tensile::ContractionProblemGemm&                     tensileProblem)
    {
        // Tensile DataTypes corresponding to rocblaslt data types
        static constexpr Tensile::DataType Tensile_TiA = tensile_datatype<TiA>;
        static constexpr Tensile::DataType Tensile_TiB = tensile_datatype<TiB>;
        static constexpr Tensile::DataType Tensile_To  = tensile_datatype<To>;
        static constexpr Tensile::DataType Tensile_Tc  = tensile_datatype<Tc>;

        // Tensile Indices for contraction problem
        Tensile::ContractionProblemGemm::FreeIndices  freeIndex(2);
        Tensile::ContractionProblemGemm::BoundIndices boundIndex(1);
        Tensile::ContractionProblemGemm::BatchIndices batchIndex{{2, 2, 2, 2}};

        // Set up GEMM indices
        freeIndex[0].isA = true;
        freeIndex[1].isA = false;
        freeIndex[0].c = freeIndex[0].d = 0;
        freeIndex[1].c = freeIndex[1].d = 1;

        // We set K=0 when alpha==0.
        // This makes alpha==0 a change in the problem, and not just a change in the
        // inputs. It optimizes all problems with alpha==0 into K=0 and alpha=(don't
        // care)
        auto k = prob.k; // && *prob.alpha ? prob.k : 0;

        // clang-format off

        // If A is transposed, swap the free and bound dimensions and their ranks
        if(prob.trans_a != HIPBLAS_OP_N)
        {
            tensileProblem.resetTensor(Tensile::ContractionProblemGemm::TENSOR::A,
                    Tensile_TiA,
                    {k, prob.m, prob.batch_count},
                    {prob.row_stride_a, prob.col_stride_a, prob.batch_stride_a});
            freeIndex[0].i  = 1;
            boundIndex[0].a = 0;
        }
        else
        {
            tensileProblem.resetTensor(Tensile::ContractionProblemGemm::TENSOR::A,
                    Tensile_TiA,
                    {prob.m, k, prob.batch_count},
                    {prob.row_stride_a, prob.col_stride_a, prob.batch_stride_a});
            freeIndex[0].i  = 0;
            boundIndex[0].a = 1;
        }

        // If B is transposed, swap the free and bound dimensions and their ranks
        if(prob.trans_b != HIPBLAS_OP_N)
        {
            tensileProblem.resetTensor(Tensile::ContractionProblemGemm::TENSOR::B,
                    Tensile_TiB,
                    {prob.n, k, prob.batch_count},
                    {prob.row_stride_b, prob.col_stride_b, prob.batch_stride_b});
            freeIndex[1].i  = 0;
            boundIndex[0].b = 1;
        }
        else
        {
            tensileProblem.resetTensor(Tensile::ContractionProblemGemm::TENSOR::B,
                    Tensile_TiB,
                    {k, prob.n, prob.batch_count},
                    {prob.row_stride_b, prob.col_stride_b, prob.batch_stride_b});
            freeIndex[1].i  = 1;
            boundIndex[0].b = 0;
        }

        // clang-format on

        // Descriptor for input matrix C
        tensileProblem.resetTensor(Tensile::ContractionProblemGemm::TENSOR::C,
                                   Tensile_To,
                                   {prob.m, prob.n, prob.batch_count},
                                   {prob.row_stride_c, prob.col_stride_c, prob.batch_stride_c});

        // Descriptor for output matrix D
        tensileProblem.resetTensor(Tensile::ContractionProblemGemm::TENSOR::D,
                                   Tensile_To,
                                   {prob.m, prob.n, prob.batch_count},
                                   {prob.row_stride_d, prob.col_stride_d, prob.batch_stride_d});

        tensileProblem.updateProblem(
            freeIndex, batchIndex, boundIndex, (double)(*prob.beta), prob.workspaceSize);

        tensileProblem.setComputeInputType(
            roc2TensileComputeInputType(Tensile_TiA, Tensile_TiB, prob.compute_type));
        tensileProblem.setAlphaType(Tensile_Tc);
        tensileProblem.setBetaType(Tensile_Tc);

        // HPA is active iff sizeof(compute type) > sizeof(input type)
        tensileProblem.setHighPrecisionAccumulate(sizeof(Tc) > sizeof(TiA));

        // set batch mode
        tensileProblem.setStridedBatched(prob.strided_batch);
        tensileProblem.setGroupedGemm(prob.grouped_gemm);
        // FIXME: Hardcoded, fix prob.trans_b == HIPBLAS_OP_N when GSU is supported
        if(prob.grouped_gemm && prob.trans_b == HIPBLAS_OP_N)
            tensileProblem.setUseDeviceUserArguments(true);
        else
            tensileProblem.setUseDeviceUserArguments(false);

        // alpha and beta are stored by value in Tensile::TypedContractionInputs
        // alpha and beta are copied from host to Tensile::TypedContractionInputs
        // If k==0, we do not need to dereference prob.alpha and can set
        // tensileAlpha=0 Not positive if this is necessary here as well
        typename AlphaBeta<TiA, TiB, To, Tc>::tensile_type tensileAlpha;
        if(prob.k)
            AlphaBeta<TiA, TiB, To, Tc>::copy(&tensileAlpha, prob.alpha);
        else
            memset(&tensileAlpha, 0, sizeof(tensileAlpha));
        tensileProblem.setAlphaRestriction(Tensile::toScalarValueEnum(tensileAlpha));

        // Add problem predicates for CEqualsD
        tensileProblem.setCEqualsD(prob.C == prob.D);

        auto tensileAct = getTensileActivationType(prob.epilogue);

        if(fallback && prob.bias == nullptr && prob.scaleAlphaVec == nullptr && prob.E == nullptr
           && tensileAct == Tensile::ActivationType::None)
        {
            tensileProblem.setUseBias(false);
            tensileProblem.setActivationType(Tensile::ActivationType::None);
            tensileProblem.setUseScaleAlphaVec(false);
            tensileProblem.setUseE(false);
            tensileProblem.setUseGradient(false);
        }
        else
        {
            auto& d = tensileProblem.tensor(Tensile::ContractionProblemGemm::TENSOR::D);
            // set bias mode
            auto biasSrc  = getBiasSrc(prob.epilogue);
            auto biasSize = (biasSrc == Tensile::ContractionProblemGemm::TENSOR::B) ? d.sizes()[1]
                                                                                    : d.sizes()[0];
            tensileProblem.setUseBias(true);
            tensileProblem.setBias(hipblasltDatatype_to_tensile_type(prob.bias_type),
                                   biasSize,
                                   0,
                                   prob.gradient,
                                   biasSrc);

            // ScaleAB is only supported on F8/BF8
            if(Tensile_TiA == Tensile::DataType::Float8 || Tensile_TiA == Tensile::DataType::BFloat8
               || Tensile_TiB == Tensile::DataType::Float8
               || Tensile_TiB == Tensile::DataType::BFloat8)
            {
                tensileProblem.setUseScaleAB(true);
                tensileProblem.setUseScaleCD(false); // Currently disabled.
                tensileProblem.setScaleA(Tensile_Tc);
                tensileProblem.setScaleB(Tensile_Tc);
                tensileProblem.setUseScaleAlphaVec(true);
                tensileProblem.setScaleAlphaVec(Tensile_Tc, d.sizes()[0]);
            }
            else
            {
                tensileProblem.setUseScaleAB(false);
                tensileProblem.setUseScaleCD(false);
                // set ScaleAlphaVec mode
                tensileProblem.setUseScaleAlphaVec(true);
                tensileProblem.setScaleAlphaVec(Tensile_Tc, d.sizes()[0]);
            }

            // set Actvation
            // only bias src A/B cannot enabled Act
            if(!is_biasSrc_AB(prob.epilogue))
            {
                tensileProblem.setActivationType(Tensile::ActivationType::All);
                tensileProblem.setActivationComputeType(Tensile_Tc);
                tensileProblem.setActivationEnumArg(tensileAct);
            }
            else
            {
                tensileProblem.setActivationType(Tensile::ActivationType::None);
                tensileProblem.setActivationEnumArg(Tensile::ActivationType::None);
            }

            // set E
            if(is_e_enabled(prob.epilogue))
            {
                bool isOutput = prob.gradient ? false : true;
                tensileProblem.setUseE(true);
                tensileProblem.setE(Tensile_Tc,
                                    {prob.m, prob.n, prob.batch_count},
                                    {prob.row_stride_e, prob.col_stride_e, prob.batch_stride_e},
                                    isOutput);
            }

            // set gradient
            tensileProblem.setUseGradient(is_grad_enabled(prob.epilogue));
        }

        if constexpr(std::is_same<TiA, float>{} && std::is_same<To, float>{}
                     && std::is_same<Tc, float>{})
            if(prob.compute_type == rocblaslt_compute_f32_fast_xf32)
                tensileProblem.setF32XdlMathOp(Tensile::DataType::XFloat32);
    }

    /***************************************************************
 * Construct the inputs to a Tensile ContractionProblemGemm        *
 ***************************************************************/
    template <typename TiA, typename TiB, typename To, typename Tc>
    auto GetTensileInputs(const RocblasltContractionProblem<TiA, TiB, To, Tc>& prob)
    {
        // Tensile types corresponding to TiA, TiB, To, Tc
        using Tensile_TiA         = typename rocblaslt_to_tensile_type<TiA>::tensile_type;
        using Tensile_TiB         = typename rocblaslt_to_tensile_type<TiB>::tensile_type;
        using Tensile_To          = typename rocblaslt_to_tensile_type<To>::tensile_type;
        using Tensile_Talpha_beta = typename AlphaBeta<TiA, TiB, To, Tc>::tensile_type;

        // Make sure rocblaslt and Tensile types are compatible
        // (Even if Ti=rocblaslt_int8x4, Tensile_Ti=Int8x4, they are both 32-byte)
        static_assert(sizeof(Tensile_TiA) == sizeof(TiA) && sizeof(Tensile_To) == sizeof(To),
                      "Tensile and rocblaslt types are not the same size");

        static_assert(std::is_standard_layout<TiA>{} && std::is_standard_layout<Tensile_TiA>{}
                          && std::is_standard_layout<To>{} && std::is_standard_layout<Tensile_To>{},
                      "Tensile or rocblaslt types are not standard layout types");

        // Structure describing the inputs (A, B, C, D, alpha, beta)
        Tensile::ContractionInputs inputs;

        // Set the A, B, C, D matrices pointers in Tensile
        inputs.a = reinterpret_cast<const void*>(prob.A);
        inputs.b = reinterpret_cast<const void*>(prob.B);
        inputs.c = reinterpret_cast<const void*>(prob.C);
        inputs.d = reinterpret_cast<void*>(prob.D);
        inputs.e = reinterpret_cast<void*>(prob.E);

        inputs.batchA = reinterpret_cast<void const* const*>(prob.batch_A);
        inputs.batchB = reinterpret_cast<void const* const*>(prob.batch_B);
        inputs.batchC = reinterpret_cast<void const* const*>(prob.batch_C);
        inputs.batchD = reinterpret_cast<void* const*>(prob.batch_D);

        // Set the GSU workspace
        inputs.ws = prob.workspace;

        // set bias vector
        inputs.bias          = reinterpret_cast<const void*>(prob.bias);
        inputs.scaleA        = reinterpret_cast<const void*>(prob.scaleA);
        inputs.scaleB        = reinterpret_cast<const void*>(prob.scaleB);
        inputs.scaleC        = nullptr;
        inputs.scaleD        = nullptr;
        inputs.scaleAlphaVec = reinterpret_cast<const void*>(prob.scaleAlphaVec);

        // push 2 activation arguments
        inputs.activationArgs.push_back(static_cast<Tensile_Talpha_beta>(0.0f));
        inputs.activationArgs.push_back(static_cast<Tensile_Talpha_beta>(0.0f));

        // alpha and beta are stored by value in Tensile::TypedContractionInputs
        // alpha and beta are copied from host to Tensile::TypedContractionInputs
        // If k==0, we do not need to dereference prob.alpha and can set
        // inputs.alpha=0
        if(prob.k)
            inputs.alpha = static_cast<Tensile_Talpha_beta>((*prob.alpha));
        else
            inputs.alpha = static_cast<Tensile_Talpha_beta>(0);
        inputs.beta = static_cast<Tensile_Talpha_beta>((*prob.beta));

        return inputs;
    }

    /**************************************************
 * The TensileHost struct interfaces with Tensile *
 **************************************************/
    class TensileHost
    {
        // The library object
        std::shared_ptr<Tensile::MasterSolutionLibrary<Tensile::ContractionProblemGemm>> m_library;
        std::shared_ptr<hipDeviceProp_t> m_deviceProp;

        // The adapter object. mutable is used to allow adapters to be modified
        // even when they are stored in a const vector which is immutable in size
        struct adapter_s
        {
            mutable std::atomic<Tensile::hip::SolutionAdapter*> adapter{nullptr};
            mutable std::mutex                                  mutex;
        };

        // Each device contains an adapter
        std::vector<adapter_s> const m_adapters;

    public:
        TensileHost()
            : m_adapters(GetDeviceCount())
        {
            // We mark TensileHost as initialized. This is so that CI tests can
            // verify that the initialization occurs in the "multiheaded" tests
            rocblaslt_internal_tensile_is_initialized() = true;
        }

        // TensileHost is not copyable or assignable
        TensileHost(const TensileHost&)            = delete;
        TensileHost& operator=(const TensileHost&) = delete;

        // Get the number of devices
        static int GetDeviceCount()
        {
            int count;
            if(hipGetDeviceCount(&count) != hipSuccess)
            {
                std::cerr << "\nrocblaslt error: Could not initialize Tensile host: No "
                             "devices found"
                          << std::endl;
                // rocblaslt_abort();
            }
            return count;
        }

        ~TensileHost()
        {
            for(auto& a : m_adapters)
                delete a.adapter;
        }

        auto& get_library() const
        {
            return m_library;
        }

        auto& get_device_property() const
        {
            return m_deviceProp;
        }

        auto& get_adapters() const
        {
            return m_adapters;
        }

        /*******************************************************
   * Testpath() tests that a path exists and is readable *
   *******************************************************/
        static bool TestPath(const std::string& path)
        {
            return rocblaslt_internal_test_path(path);
        }

        /*********************************************************************
   * Initialize adapter and library according to environment variables *
   * and default paths based on librocblaslt.so location and GPU         *
   *********************************************************************/
        void initialize(Tensile::hip::SolutionAdapter& adapter, int32_t deviceId)
        {
            std::string path;
#ifndef WIN32
            path.reserve(PATH_MAX);
#endif

            // The name of the current GPU platform
            std::string processor = rocblaslt_internal_get_arch_name();

            const char* env = getenv("HIPBLASLT_TENSILE_LIBPATH");
            if(env)
            {
                path = env;
            }
            else
            {
                path = HIPBLASLT_LIB_PATH;

                // Find the location of librocblaslt.so
                // Fall back on hard-coded path if static library or not found

#ifndef HIPBLASLT_STATIC_LIB
                auto hipblaslt_so_path = getHipblasltSoPath();

                if(hipblaslt_so_path.size())
                    path = std::string{dirname(&hipblaslt_so_path[0])};
#endif // ifndef HIPBLASLT_STATIC_LIB

                // Find the location of the libraries
                if(TestPath(path + "/../Tensile/library"))
                    path += "/../Tensile/library";
                else if(TestPath(path + "library"))
                    path += "/library";
                else
                    path += "/hipblaslt/library";

                if(TestPath(path + "/" + processor))
                    path += "/" + processor;
            }

            // only load modules for the current architecture
            auto dir = path + "/*" + processor + "*co";

            bool no_match = false;
#ifdef WIN32
            std::replace(dir.begin(), dir.end(), '/', '\\');
            WIN32_FIND_DATAA finddata;
            HANDLE           hfine = FindFirstFileA(dir.c_str(), &finddata);
            if(hfine != INVALID_HANDLE_VALUE)
            {
                do
                {
                    std::string codeObjectFile = path + "\\" + finddata.cFileName;
                    static_cast<void>(adapter.loadCodeObjectFile(codeObjectFile.c_str()));
                } while(FindNextFileA(hfine, &finddata));
            }
            else
            {
                no_match = true;
            }
            FindClose(hfine);
#else
            glob_t glob_result{};
            int    g = glob(dir.c_str(), GLOB_NOSORT, nullptr, &glob_result);
            if(!g)
            {
                for(size_t i = 0; i < glob_result.gl_pathc; ++i)
                    static_cast<void>(adapter.loadCodeObjectFile(glob_result.gl_pathv[i]));
            }
            else if(g == GLOB_NOMATCH)
            {
                no_match = true;
            }
            else
            {
#if 0
                // clang-format off
                static std::ostream& once = std::cerr
                                    << "\nrocblaslt warning: glob(\"" << dir << "\", ...) returned "
                                    << (g == GLOB_ABORTED ? "GLOB_ABORTED"
                                                          : g == GLOB_NOSPACE ? "GLOB_NOSPACE"
                                                                              : "an unknown error")
                                    << "." << std::endl;
                // clang-format on
#endif
            }
            globfree(&glob_result);
#endif
            if(no_match)
            {
                // static rocblaslt_internal_ostream& once
                //    = rocblaslt_cerr
                std::cerr << "\nrocblaslt warning: No paths matched " << dir
                          << ". Make sure that HIPBLASLT_TENSILE_LIBPATH is set correctly."
                          << std::endl;
            }

            // We initialize a local static variable with a lambda function call to
            // avoid race conditions when multiple threads with different device IDs try
            // to initialize library. This ensures that only one thread initializes
            // library, and other threads trying to initialize library wait for it to
            // complete.
            static int once = [&] {
#ifdef TENSILE_YAML
                path += "/TensileLibrary.yaml";
#else
                path += "/TensileLibrary.dat";
#endif
                if(!TestPath(path))
                {
                    std::cerr << "\nrocblaslt error: Cannot read " << path << ": "
                              << strerror(errno) << std::endl;
                    // rocblaslt_abort();
                }

                auto lib = Tensile::LoadLibraryFile<Tensile::ContractionProblemGemm>(path);
                if(!lib)
                    std::cerr << "\nrocblaslt error: Could not load " << path << std::endl;
                else
                {
                    using MSL = Tensile::MasterSolutionLibrary<Tensile::ContractionProblemGemm>;
                    m_library = std::dynamic_pointer_cast<MSL>(lib);
                }
                return 0;
            }();

            if(!m_library && once != 0)
            {
                std::cerr << "\nrocblaslt error: Could not initialize Tensile library" << std::endl;
                // rocblaslt_abort();
            }

            hipDeviceProp_t prop;
            HIP_CHECK_EXC(hipGetDeviceProperties(&prop, deviceId));

            m_deviceProp = std::make_shared<hipDeviceProp_t>(prop);
        }
    };

    // Return the library and adapter for the current HIP device
    Tensile::hip::SolutionAdapter* get_library_and_adapter(
        std::shared_ptr<Tensile::MasterSolutionLibrary<Tensile::ContractionProblemGemm>>* library
        = nullptr,
        std::shared_ptr<hipDeviceProp_t>* deviceProp = nullptr,
        int                               device     = -1)
    try
    {
        // TensileHost is initialized on the first call
        static TensileHost host;

        if(device == -1)
            static_cast<void>(hipGetDevice(&device));

        // Adapter entry for the current HIP device ID
        auto& a       = host.get_adapters().at(device);
        auto* adapter = a.adapter.load(std::memory_order_acquire);

        // Once set, a.adapter contains the adapter for the current HIP device ID
        if(!adapter)
        {
            // Lock so that only one thread performs initialization of the adapter
            std::lock_guard<std::mutex> lock(a.mutex);

            adapter = a.adapter.load(std::memory_order_relaxed);
            if(!adapter)
            {
                // Allocate a new adapter using the current HIP device
                adapter = new Tensile::hip::SolutionAdapter;

                // Initialize the adapter and possibly the library
                host.initialize(*adapter, device);

                // Atomically change the adapter stored for this device ID
                a.adapter.store(adapter, std::memory_order_release);
            }
        }

        // If an adapter is found, it is assumed that the library is initialized
        if(library)
            *library = host.get_library();
        if(deviceProp)
            *deviceProp = host.get_device_property();

        return adapter;
    }
    catch(const std::exception& e)
    {
        std::cerr << "\nrocblaslt error: Could not initialize Tensile host:\n"
                  << e.what() << std::endl;
        return nullptr;
    }
    catch(...)
    {
        std::cerr << "\nrocblaslt error: Could not initialize Tensile host:\nUnknown "
                     "exception thrown"
                  << std::endl;
        return nullptr;
    }

#if 0
    /**************************************************************************
    * We normally print error messages only once, to avoid excessive logging *
    **************************************************************************/
    void print_once(const std::ostream& msg)
    {
        if(rocblaslt_suppress_tensile_error_messages())
            return;
        static constexpr char varname[] = "ROCBLASLT_VERBOSE_TENSILE_ERROR";
        static const char*    verbose   = getenv(varname);
        if(!verbose)
        {
            static auto& once = std::cerr
                                << msg
                                << "\nThis message will be only be displayed once, unless the "
                                << varname << " environment variable is set." << std::endl;
        }
        else
            std::cerr << msg << std::endl;
    }
#endif
} // namespace


struct TensileDataGemm
{
    bool                                   enableEpilogue = true;
    Tensile::ContractionProblemGemm        problem;
    Tensile::ContractionInputs             inputs;
    std::vector<Tensile::KernelInvocation> kernels;
    int                                    algoIndex = std::numeric_limits<int>::max();
};

struct TensileDataGroupedGemm
{
    bool                                   enableEpilogue = true;
    Tensile::ContractionProblemGroupedGemm problem;
    Tensile::ContractionGroupedInputs      inputs;
    std::vector<Tensile::KernelInvocation> kernels;
    int                                    algoIndex = std::numeric_limits<int>::max();
    std::shared_ptr<void>                  hipHostMemory;
    size_t                                 hipHostMemorySize;
    bool                                   useUserArgs = false;
};

void initTensileGemmData(rocblaslt_handle       handle,
                         rocblaslt::RocGemmType gemmType,
                         hipblasOperation_t     opA,
                         hipblasOperation_t     opB,
                         hipblasltDatatype_t    typeA,
                         hipblasltDatatype_t    typeB,
                         hipblasltDatatype_t    typeC,
                         hipblasltDatatype_t    typeD,
                         rocblaslt_compute_type typeCompute,
                         size_t                 maxWorkspaceBytes,
                         std::shared_ptr<void>& gemmData)
{
    float alpha = 1.0;
    float beta  = 1.0;
    if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GEMM)
    {
        TensileDataGemm data;
        data.problem = CreateTensileProblem(opA,
                                            opB,
                                            typeA,
                                            typeB,
                                            typeC,
                                            typeD,
                                            typeCompute,
                                            alpha,
                                            beta,
                                            false,
                                            maxWorkspaceBytes);
        gemmData     = std::static_pointer_cast<void>(std::make_shared<TensileDataGemm>(data));
        return;
    }
    else if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GROUPED_GEMM)
    {
        TensileDataGroupedGemm                  data;
        Tensile::ContractionProblemGroupedGemm& tensile_probs = data.problem;
        Tensile::ContractionGroupedInputs&      groupedInputs = data.inputs;

        tensile_probs.gemms.push_back(CreateTensileProblem(opA,
                                                           opB,
                                                           typeA,
                                                           typeB,
                                                           typeC,
                                                           typeD,
                                                           typeCompute,
                                                           alpha,
                                                           beta,
                                                           true,
                                                           maxWorkspaceBytes));
        groupedInputs.grouped.resize(1);

        void* tmp = nullptr;
        static_cast<void>(hipHostMalloc(&tmp, INTERNAL_HIPHOSTMEM_SIZE, 0));
        data.hipHostMemory
            = std::shared_ptr<void>(tmp, [](auto p) { static_cast<void>(hipFree(p)); });
        data.hipHostMemorySize = INTERNAL_HIPHOSTMEM_SIZE;

        gemmData = std::static_pointer_cast<void>(std::make_shared<TensileDataGroupedGemm>(data));
        return;
    }

    throw std::runtime_error("Gemm problem type initialization not implemented.");
}

/******************************************************************************
 * runContractionProblem calls Tensile to run a contraction problem described *
 * by RocblasltContractionProblem *
 ******************************************************************************/
template <typename TiA, typename TiB, typename To, typename Tc>
rocblaslt_status runContractionProblem(rocblaslt_handle                                     handle,
                                       const rocblaslt_matmul_algo*                         algo,
                                       const RocblasltContractionProblem<TiA, TiB, To, Tc>& prob,
                                       std::shared_ptr<void> gemmData)
{
    rocblaslt_status status = rocblaslt_status_internal_error;
    try
    {
        std::shared_ptr<Tensile::MasterSolutionLibrary<Tensile::ContractionProblemGemm>> library;
        std::shared_ptr<hipDeviceProp_t>                                                 deviceProp;
        std::shared_ptr<Tensile::Hardware>                                               hardware;

        auto adapter = get_library_and_adapter(&library, &deviceProp, handle->device);
        hardware     = Tensile::hip::GetDevice(*deviceProp);

        std::shared_ptr<TensileDataGemm> data = std::static_pointer_cast<TensileDataGemm>(gemmData);
        updateTensileProblem(algo->fallback, prob, data->problem);

        int* solutionIndex = (int*)algo->data;
        data->algoIndex    = *solutionIndex;

        auto solution = library->getSolutionByIndex(data->problem, *hardware, *solutionIndex);
        if(!solution)
        {
#if 0
            std::ostream msg;
            print_once(msg << "\nrocblaslt error: No Tensile solution found for " << prob);
#endif
            status = rocblaslt_status_not_implemented;
        }
        else
        {
            static_cast<void>(adapter->launchKernels(
                solution->solve(data->problem, GetTensileInputs(prob), *hardware),
                prob.stream,
                nullptr,
                nullptr));
            status = rocblaslt_status_success;
        }
    }
    catch(const std::exception& e)
    {
#if 0
        std::ostream msg;
        print_once(msg << "\nrocblaslt error: " << (solution ? "" : "No ")
                       << "Tensile solution found, but exception thrown for " << prob << e.what());
#endif
    }
    catch(...)
    {
#if 0
        std::ostream msg;
        print_once(msg << "\nrocblaslt error: " << (solution ? "" : "No ")
                       << "Tensile solution found, but unknown exception thrown for " << prob);
#endif
    }

    return status;
}

template <typename TiA, typename TiB, typename To, typename Tc>
rocblaslt_status gemmCreate(RocblasltContractionProblem<TiA, TiB, To, Tc> const& problem,
                            std::shared_ptr<void>&                               gemmData,
                            size_t&                                              gemmCount)
{
    rocblaslt_status status = rocblaslt_status_internal_error;
    try
    {
        // Check if pointer is valid
        if(problem.alpha == nullptr || problem.beta == nullptr || problem.A == nullptr
           || problem.B == nullptr || problem.C == nullptr || problem.D == nullptr)
        {
            log_error(__func__, "invalid data pointer");
            return rocblaslt_status_invalid_pointer;
        }
        gemmCount = 1;
        if(gemmData)
        {
            std::shared_ptr<TensileDataGemm> data
                = std::static_pointer_cast<TensileDataGemm>(gemmData);
            updateTensileProblem(false, problem, data->problem);
            data->inputs         = GetTensileInputs(problem);
            data->enableEpilogue = problem.epilogue == ROCBLASLT_EPILOGUE_DEFAULT ? false : true;
        }
        else
        {
            TensileDataGemm data;
            data.problem        = ConstructTensileProblem(problem);
            data.inputs         = GetTensileInputs(problem);
            data.enableEpilogue = problem.epilogue == ROCBLASLT_EPILOGUE_DEFAULT ? false : true;

            gemmData = std::static_pointer_cast<void>(std::make_shared<TensileDataGemm>(data));
        }

        status = rocblaslt_status_success;
    }
    catch(const std::exception& e)
    {
#if 0
        std::ostream msg;
        print_once(msg << "\nrocblaslt error: " << (solution ? "" : "No ")
                       << "Tensile solution found, but exception thrown for " << prob << e.what());
#endif
    }
    catch(...)
    {
#if 0
        std::ostream msg;
        print_once(msg << "\nrocblaslt error: " << (solution ? "" : "No ")
                       << "Tensile solution found, but unknown exception thrown for " << prob);
#endif
    }

    return status;
}

template <typename TiA, typename TiB, typename To, typename Tc>
rocblaslt_status
    groupedGemmCreate(std::vector<RocblasltContractionProblem<TiA, TiB, To, Tc>>& probs,
                      std::shared_ptr<void>&                                      gemmData,
                      size_t&                                                     gemmCount)
{
    gemmCount = probs.size();
    if(gemmCount == 0)
        return rocblaslt_status_success;
    rocblaslt_status status = rocblaslt_status_internal_error;
    try
    {
        bool enableEpilogue = false;
        if(gemmData)
        {
            // Need to check if is same type?
            std::shared_ptr<TensileDataGroupedGemm> data
                = std::static_pointer_cast<TensileDataGroupedGemm>(gemmData);
            Tensile::ContractionProblemGroupedGemm& tensile_probs = data->problem;
            Tensile::ContractionGroupedInputs&      groupedInputs = data->inputs;

            groupedInputs.grouped.clear();
            if(tensile_probs.gemms.size() != probs.size())
                tensile_probs.gemms.clear();

            for(int i = 0; i < probs.size(); i++)
            {
                // Check if pointer is valid for n != 0
                if(probs[i].n)
                {
                    if(probs[i].alpha == nullptr || probs[i].beta == nullptr
                       || probs[i].A == nullptr || probs[i].B == nullptr || probs[i].C == nullptr
                       || probs[i].D == nullptr)
                    {
                        log_error(__func__, "invalid data pointer");
                        return rocblaslt_status_invalid_pointer;
                    }
                }
                if(tensile_probs.gemms.size() != probs.size())
                    tensile_probs.gemms.push_back(ConstructTensileProblem(probs[i]));
                else
                    updateTensileProblem(false, probs[i], tensile_probs.gemms[i]);
                groupedInputs.grouped.push_back(GetTensileInputs(probs[i]));
                if(probs[i].epilogue != ROCBLASLT_EPILOGUE_DEFAULT)
                    enableEpilogue = true;
            }
            data->enableEpilogue = enableEpilogue;
        }
        else
        {
            TensileDataGroupedGemm                  data;
            Tensile::ContractionProblemGroupedGemm& tensile_probs = data.problem;
            Tensile::ContractionGroupedInputs&      groupedInputs = data.inputs;

            for(int i = 0; i < probs.size(); i++)
            {
                // Check if pointer is valid for n != 0
                if(probs[i].n)
                {
                    if(probs[i].alpha == nullptr || probs[i].beta == nullptr
                       || probs[i].A == nullptr || probs[i].B == nullptr || probs[i].C == nullptr
                       || probs[i].D == nullptr)
                    {
                        log_error(__func__, "invalid data pointer");
                        return rocblaslt_status_invalid_pointer;
                    }
                }
                tensile_probs.gemms.push_back(ConstructTensileProblem(probs[i]));
                groupedInputs.grouped.push_back(GetTensileInputs(probs[i]));
                if(probs[i].epilogue != ROCBLASLT_EPILOGUE_DEFAULT)
                    enableEpilogue = true;
            }
            data.enableEpilogue = enableEpilogue;

            gemmData
                = std::static_pointer_cast<void>(std::make_shared<TensileDataGroupedGemm>(data));
        }
        status = rocblaslt_status_success;
    }
    catch(const std::exception& e)
    {
#if 0
        std::ostream msg;
        print_once(msg << "\nrocblaslt error: " << (solution ? "" : "No ")
                       << "Tensile solution found, but exception thrown for " << prob << e.what());
#endif
    }
    catch(...)
    {
#if 0
        std::ostream msg;
        print_once(msg << "\nrocblaslt error: " << (solution ? "" : "No ")
                       << "Tensile solution found, but unknown exception thrown for " << prob);
#endif
    }

    return status;
}

rocblaslt_status makeArgument(rocblaslt_handle             handle,
                              const rocblaslt::RocGemmType gemmType,
                              const rocblaslt_matmul_algo& algo,
                              void*                        workspace,
                              bool                         useUserArgs,
                              hipStream_t                  stream,
                              std::shared_ptr<void>        gemmData)
{
    rocblaslt_status status = rocblaslt_status_internal_error;
    try
    {
        std::shared_ptr<Tensile::MasterSolutionLibrary<Tensile::ContractionProblemGemm>> library;
        std::shared_ptr<hipDeviceProp_t>                                                 deviceProp;
        std::shared_ptr<Tensile::Hardware>                                               hardware;

        auto adapter = get_library_and_adapter(&library, &deviceProp, handle->device);
        hardware     = Tensile::hip::GetDevice(*deviceProp);

        int* solutionIndex = (int*)algo.data;
        if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GEMM)
        {
            std::shared_ptr<TensileDataGemm> data
                = std::static_pointer_cast<TensileDataGemm>(gemmData);

            data->algoIndex = *solutionIndex;
            auto solution   = library->getSolutionByIndex(data->problem, *hardware, *solutionIndex);

            data->inputs.ws = workspace;

            // Backup and restore settings
            bool                    useBias          = data->problem.useBias();
            Tensile::ActivationType actType          = data->problem.activationType();
            bool                    useScaleAlphaVec = data->problem.useScaleAlphaVec();
            bool                    useE             = data->problem.useE();
            bool                    useGrad          = data->problem.useGradient();
            data->problem.setUseBias(solution->problemType.useBias);
            data->problem.setActivationType(solution->problemType.activationType);
            data->problem.setUseScaleAlphaVec(solution->problemType.useScaleAlphaVec);
            data->problem.setUseE(solution->problemType.useE);
            data->problem.setUseGradient(solution->problemType.useGradient);
            data->kernels = solution->solve(data->problem, data->inputs, *hardware);
            data->problem.setUseBias(useBias);
            data->problem.setActivationType(actType);
            data->problem.setUseScaleAlphaVec(useScaleAlphaVec);
            data->problem.setUseE(useE);
            data->problem.setUseGradient(useGrad);
        }
        else if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GROUPED_GEMM)
        {
            std::shared_ptr<TensileDataGroupedGemm> data
                = std::static_pointer_cast<TensileDataGroupedGemm>(gemmData);

            data->algoIndex = *solutionIndex;
            auto solution
                = library->getSolutionByIndex(data->problem.gemms[0], *hardware, *solutionIndex);

            for(int i = 0; i < data->inputs.grouped.size(); i++)
            {
                data->inputs.grouped[i].ws = workspace;
            }
            data->inputs.ws = workspace;

            data->useUserArgs = useUserArgs;
            if(useUserArgs)
            {
                data->kernels = solution->solveGroupedGemmGPU(
                    data->problem.gemms, data->inputs, nullptr, workspace, stream);
            }
            else
            {
                // fallback to normal gemm if is normal kernel
                std::vector<bool>                    useBias, actHPA, useScaleAlphaVec;
                std::vector<Tensile::ActivationType> actType;
                for(int i = 0; i < data->problem.gemms.size(); i++)
                {
                    useBias.push_back(data->problem.gemms[i].useBias());
                    actType.push_back(data->problem.gemms[i].activationType());
                    data->problem.gemms[i].setUseBias(solution->problemType.useBias);
                    data->problem.gemms[i].setActivationType(solution->problemType.activationType);
                    data->problem.gemms[i].setUseScaleAlphaVec(
                        solution->problemType.useScaleAlphaVec);
                }

                size_t requiedHostSize
                    = solution->requiredHostWorkspaceSizePerProblem * data->problem.gemms.size();
                if(requiedHostSize > data->hipHostMemorySize)
                {
                    void* tmp = nullptr;
                    static_cast<void>(hipHostMalloc(&tmp, requiedHostSize, 0));
                    data->hipHostMemory
                        = std::shared_ptr<void>(tmp, [](auto p) { static_cast<void>(hipFree(p)); });
                    data->hipHostMemorySize = requiedHostSize;
                }

                data->kernels = solution->solveGroupedGemm(data->problem.gemms,
                                                           data->inputs,
                                                           *hardware,
                                                           data->hipHostMemory.get(),
                                                           data->hipHostMemorySize,
                                                           stream);
                for(int i = 0; i < data->problem.gemms.size(); i++)
                {
                    data->problem.gemms[i].setUseBias(useBias[i]);
                    data->problem.gemms[i].setActivationType(actType[i]);
                    data->problem.gemms[i].setUseScaleAlphaVec(useScaleAlphaVec[i]);
                }
            }
        }
        status = rocblaslt_status_success;
    }
    catch(const std::exception& e)
    {
#if 0
        std::ostream msg;
        print_once(msg << "\nrocblaslt error: " << (solution ? "" : "No ")
                       << "Tensile solution found, but exception thrown for " << prob << e.what());
#endif
    }
    catch(...)
    {
#if 0
        std::ostream msg;
        print_once(msg << "\nrocblaslt error: " << (solution ? "" : "No ")
                       << "Tensile solution found, but unknown exception thrown for " << prob);
#endif
    }

    return status;
}

rocblaslt_status runKernelFromInvocation(rocblaslt_handle       handle,
                                         rocblaslt::RocGemmType gemmType,
                                         std::shared_ptr<void>  gemmData,
                                         hipStream_t            stream)
{
    rocblaslt_status status = rocblaslt_status_internal_error;
    try
    {
        std::shared_ptr<Tensile::MasterSolutionLibrary<Tensile::ContractionProblemGemm>> library;
        std::shared_ptr<hipDeviceProp_t>                                                 deviceProp;
        std::shared_ptr<Tensile::Hardware>                                               hardware;

        auto adapter = get_library_and_adapter(&library, &deviceProp, handle->device);

        if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GEMM)
        {
            std::shared_ptr<TensileDataGemm> data
                = std::static_pointer_cast<TensileDataGemm>(gemmData);
            static_cast<void>(adapter->launchKernels(data->kernels, stream, nullptr, nullptr));
        }
        else if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GROUPED_GEMM)
        {
            std::shared_ptr<TensileDataGroupedGemm> data
                = std::static_pointer_cast<TensileDataGroupedGemm>(gemmData);
            if(data->useUserArgs)
            {
                log_error(__func__,
                          "GG is initialized with useUserArgs = true, workspace has no arguments.");
                return rocblaslt_status_not_initialized;
            }
            static_cast<void>(adapter->launchKernels(data->kernels, stream, nullptr, nullptr));
        }
        else
        {
            return rocblaslt_status_invalid_value;
        }

        status = rocblaslt_status_success;
    }
    catch(const std::exception& e)
    {
#if 0
        std::ostream msg;
        print_once(msg << "\nrocblaslt error: " << (solution ? "" : "No ")
                       << "Tensile solution found, but exception thrown for " << prob << e.what());
#endif
    }
    catch(...)
    {
#if 0
        std::ostream msg;
        print_once(msg << "\nrocblaslt error: " << (solution ? "" : "No ")
                       << "Tensile solution found, but unknown exception thrown for " << prob);
#endif
    }

    return status;
}

rocblaslt_status getDeviceUserArgumentsValuesFromContractionProblem(rocblaslt_handle       handle,
                                                                    rocblaslt::RocGemmType gemmType,
                                                                    std::shared_ptr<void>  gemmData,
                                                                    void* hostDeviceUserArgs)
{
    rocblaslt_status status = rocblaslt_status_internal_error;
    try
    {
        std::shared_ptr<Tensile::MasterSolutionLibrary<Tensile::ContractionProblemGemm>> library;
        std::shared_ptr<hipDeviceProp_t>                                                 deviceProp;
        std::shared_ptr<Tensile::Hardware>                                               hardware;

        auto adapter = get_library_and_adapter(&library, &deviceProp, handle->device);

        if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GROUPED_GEMM)
        {
            std::shared_ptr<TensileDataGroupedGemm> data
                = std::static_pointer_cast<TensileDataGroupedGemm>(gemmData);
            auto  solution = library->getSolutionByIndex(data->algoIndex);
            auto& problem  = data->problem.gemms[0];
            if(problem.activationComputeType() == Tensile::DataType::Float)
            {
                setDeviceUserArgs(data->problem.gemms,
                                  data->inputs,
                                  (Tensile::DeviceUserArguments<float>*)hostDeviceUserArgs);
            }
            else
            {
                throw std::runtime_error("Currently only supports DeviceUserArguments<float>");
            }
        }
        else
        {
            return rocblaslt_status_not_implemented;
        }
        status = rocblaslt_status_success;
    }
    catch(const std::exception& e)
    {
#if 0
        std::ostream msg;
        print_once(msg << "\nrocblaslt error: "
                       << "Is hostDeviceUserArgs not match the size of the problem type? " << prob << e.what());
#endif
    }
    catch(...)
    {
#if 0
        std::ostream msg;
        print_once(msg << "\nrocblaslt error: "
                       << "Is hostDeviceUserArgs not match the size of the problem type? " << prob);
#endif
    }

    return status;
}

rocblaslt_status runKernelFromNewDeviceUserArguments(rocblaslt_handle       handle,
                                                     rocblaslt::RocGemmType gemmType,
                                                     std::shared_ptr<void>  gemmData,
                                                     void*                  deviceUserArgs,
                                                     hipStream_t            stream)
{
    rocblaslt_status status = rocblaslt_status_internal_error;
    try
    {
        std::shared_ptr<Tensile::MasterSolutionLibrary<Tensile::ContractionProblemGemm>> library;
        std::shared_ptr<hipDeviceProp_t>                                                 deviceProp;
        std::shared_ptr<Tensile::Hardware>                                               hardware;

        auto adapter = get_library_and_adapter(&library, &deviceProp, handle->device);

        if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GROUPED_GEMM)
        {
            std::shared_ptr<TensileDataGroupedGemm> data
                = std::static_pointer_cast<TensileDataGroupedGemm>(gemmData);
            for(auto& it : data->kernels)
            {
                uint8_t* arg = it.args.rawdata();
                memcpy(arg + 4, &deviceUserArgs, sizeof(void*));
            }
            static_cast<void>(adapter->launchKernels(data->kernels, stream, nullptr, nullptr));
        }
        else
        {
            return rocblaslt_status_not_implemented;
        }

        status = rocblaslt_status_success;
    }
    catch(const std::exception& e)
    {
#if 0
        std::ostream msg;
        print_once(msg << "\nrocblaslt error: " << (solution ? "" : "No ")
                       << "Tensile solution found, but exception thrown for " << prob << e.what());
#endif
    }
    catch(...)
    {
#if 0
        std::ostream msg;
        print_once(msg << "\nrocblaslt error: " << (solution ? "" : "No ")
                       << "Tensile solution found, but unknown exception thrown for " << prob);
#endif
    }

    return status;
}

rocblaslt_status runKernelFromDeviceUserArguments(rocblaslt_handle             handle,
                                                  rocblaslt::RocGemmType       gemmType,
                                                  size_t                       gemmCount,
                                                  std::shared_ptr<void>        gemmData,
                                                  const rocblaslt_matmul_algo& algo,
                                                  void*                        deviceUserArgs,
                                                  void*                        workspace,
                                                  hipStream_t                  stream)
{
    rocblaslt_status status = rocblaslt_status_internal_error;
    try
    {
        std::shared_ptr<Tensile::MasterSolutionLibrary<Tensile::ContractionProblemGemm>> library;
        std::shared_ptr<hipDeviceProp_t>                                                 deviceProp;
        std::shared_ptr<Tensile::Hardware>                                               hardware;

        auto adapter = get_library_and_adapter(&library, &deviceProp, handle->device);

        int* solutionIndex = (int*)algo.data;
        // don't overwrite data->algoIndex = *solutionIndex; here
        if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GROUPED_GEMM)
        {
            auto solution = library->getSolutionByIndex(*solutionIndex);
            std::shared_ptr<TensileDataGroupedGemm> data
                = std::static_pointer_cast<TensileDataGroupedGemm>(gemmData);
            auto kernel = solution->solveGroupedGemmGPU(
                data->problem.gemms, data->inputs, deviceUserArgs, workspace, stream);
            static_cast<void>(adapter->launchKernels(kernel, stream, nullptr, nullptr));
        }
        else
        {
            return rocblaslt_status_not_implemented;
        }

        status = rocblaslt_status_success;
    }
    catch(const std::exception& e)
    {
#if 0
        std::ostream msg;
        print_once(msg << "\nrocblaslt error: " << (solution ? "" : "No ")
                       << "Tensile solution found, but exception thrown for " << prob << e.what());
#endif
    }
    catch(...)
    {
#if 0
        std::ostream msg;
        print_once(msg << "\nrocblaslt error: " << (solution ? "" : "No ")
                       << "Tensile solution found, but unknown exception thrown for " << prob);
#endif
    }

    return status;
}

/******************************************************************************
 * getBestSolutions calls Tensile's findTopSolutions and converts to          *
 * rocblaslt_matmul_heuristic_result.                                         *
 ******************************************************************************/

void _convertToHeuristicResultArray(
    std::vector<std::shared_ptr<Tensile::ContractionSolution>>& solutions,
    int                                                         requestedAlgoCount,
    rocblaslt_matmul_heuristic_result                           heuristicResultsArray[],
    int*                                                        returnAlgoCount,
    size_t                                                      maxWorkSpaceBytes,
    const Tensile::ContractionProblemGemm&                      problem,
    size_t                                                      fallbackCount)
{
    *returnAlgoCount = std::min((int)solutions.size(), requestedAlgoCount);
    for(size_t i = 0; i < *returnAlgoCount; i++)
    {
        auto solution = solutions[i];
        memset(heuristicResultsArray[i].algo.data, 0, sizeof(heuristicResultsArray[i].algo.data));
        int* solutionIndex = (int*)(heuristicResultsArray[i].algo.data);
        *solutionIndex     = solution->index;
        heuristicResultsArray[i].algo.max_workspace_bytes = maxWorkSpaceBytes;
        heuristicResultsArray[i].algo.fallback            = fallbackCount-- > 0 ? true : false;
        heuristicResultsArray[i].state                    = rocblaslt_status_success;
        heuristicResultsArray[i].workspaceSize = solution->requiredWorkspaceSize(problem);
    }
    for(size_t i = *returnAlgoCount; i < requestedAlgoCount; i++)
    {
        heuristicResultsArray[i].state = rocblaslt_status_invalid_value;
    }
}

template <typename T>
inline auto getSolutions(
    const T&                                                                                inputs,
    const std::shared_ptr<Tensile::MasterSolutionLibrary<Tensile::ContractionProblemGemm>>& library,
    const std::shared_ptr<Tensile::Hardware>& hardware,
    Tensile::ContractionProblemGemm&          tensile_prob,
    bool                                      enableEpilogue,
    const int&                                requestedAlgoCount,
    int&                                      fallbackSize)
{
    const void *scaleAlphaVec = nullptr, *bias = nullptr, *E = nullptr;
    if constexpr(std::is_same<T, Tensile::ContractionInputs>::value)
    {
        scaleAlphaVec = inputs.scaleAlphaVec;
        bias          = inputs.bias;
        E             = inputs.e;
    }
    else
    {
        scaleAlphaVec = inputs.scaleAlphaVec;
        bias          = inputs.bias;
        E             = inputs.E;
    }

    std::vector<std::shared_ptr<Tensile::ContractionSolution>> solutions_fallback;
    // Fallback to original kernels
    if(!enableEpilogue && scaleAlphaVec == nullptr && bias == nullptr && E == nullptr
       && tensile_prob.activationEnumArg() == Tensile::ActivationType::None)
    {
        auto useBias          = tensile_prob.useBias();
        auto actType          = tensile_prob.activationType();
        auto useScaleAlphaVec = tensile_prob.useScaleAlphaVec();
        auto useE             = tensile_prob.useE();
        tensile_prob.setUseBias(false);
        tensile_prob.setActivationType(Tensile::ActivationType::None);
        tensile_prob.setUseScaleAlphaVec(false);
        tensile_prob.setUseE(false);
        solutions_fallback = library->findTopSolutions(tensile_prob, *hardware, requestedAlgoCount);
        // restore
        tensile_prob.setUseBias(useBias);
        tensile_prob.setActivationType(actType);
        tensile_prob.setUseScaleAlphaVec(useScaleAlphaVec);
        tensile_prob.setUseE(useE);
    }

    auto solutions = library->findTopSolutions(tensile_prob, *hardware, requestedAlgoCount);
    if(solutions_fallback.size() > 0)
    {
        solutions.insert(solutions.begin(), solutions_fallback.begin(), solutions_fallback.end());
    }
    fallbackSize = solutions_fallback.size();
    return solutions;
}

template <typename TiA, typename TiB, typename To, typename Tc>
rocblaslt_status getBestSolutions(RocblasltContractionProblem<TiA, TiB, To, Tc> prob,
                                  rocblaslt_handle                              handle,
                                  std::shared_ptr<void>                         gemmData,
                                  int                                           requestedAlgoCount,
                                  rocblaslt_matmul_heuristic_result heuristicResultsArray[],
                                  int*                              returnAlgoCount,
                                  size_t                            maxWorkSpaceBytes)
{
    std::shared_ptr<Tensile::MasterSolutionLibrary<Tensile::ContractionProblemGemm>> library;
    std::shared_ptr<hipDeviceProp_t>                                                 deviceProp;
    std::shared_ptr<Tensile::Hardware>                                               hardware;

    // auto &adapter =
    static_cast<void>(get_library_and_adapter(&library, &deviceProp, handle->device));

    hardware = Tensile::hip::GetDevice(*deviceProp);

    std::shared_ptr<TensileDataGemm> data = std::static_pointer_cast<TensileDataGemm>(gemmData);
    updateTensileProblem(false, prob, data->problem);

    bool enableEpilogue = prob.epilogue == ROCBLASLT_EPILOGUE_DEFAULT ? false : true;

    int  fallbackSize = 0;
    auto solutions    = getSolutions(
        prob, library, hardware, data->problem, enableEpilogue, requestedAlgoCount, fallbackSize);

    // when there is no solution for xfloat32, fallback comput_type to fp32
    if constexpr(std::is_same<TiA, float>{} && std::is_same<To, float>{}
                 && std::is_same<Tc, float>{})
        if(solutions.size() == 0 && prob.compute_type == rocblaslt_compute_f32_fast_xf32)
        {
            log_api(__func__, "no solutions found, try to fallback");
            data->problem.setF32XdlMathOp(Tensile::DataType::Float);
            solutions = getSolutions(
                prob, library, hardware, data->problem, enableEpilogue, requestedAlgoCount, fallbackSize);
        }

    _convertToHeuristicResultArray(solutions,
                                   requestedAlgoCount,
                                   heuristicResultsArray,
                                   returnAlgoCount,
                                   maxWorkSpaceBytes,
                                   data->problem,
                                   fallbackSize);

    return rocblaslt_status_success;
}

template <typename MyProblem>
rocblaslt_status getAllSolutions(MyProblem&                                      prob,
                                 rocblaslt_handle                                handle,
                                 std::vector<rocblaslt_matmul_heuristic_result>& heuristicResults,
                                 size_t                                          maxWorkSpaceBytes)
{
    std::shared_ptr<Tensile::MasterSolutionLibrary<Tensile::ContractionProblemGemm>> library;
    std::shared_ptr<hipDeviceProp_t>                                                 deviceProp;
    std::shared_ptr<Tensile::Hardware>                                               hardware;

    // auto &adapter =
    static_cast<void>(get_library_and_adapter(&library, &deviceProp, handle->device));

    hardware = Tensile::hip::GetDevice(*deviceProp);

    std::set<std::shared_ptr<Tensile::ContractionSolution>> solutions;
    std::shared_ptr<void>                                   tensile_prob;

    if constexpr(std::is_same<MyProblem, Tensile::ContractionProblemGemm>::value)
    {
        solutions = library->findAllSolutions(
            prob, *hardware, Tensile::SolutionLibrarySearchType::GEMM_TYPE_ONLY);
    }
    else if constexpr(std::is_same<MyProblem, Tensile::ContractionProblemGroupedGemm>::value)
    {
        solutions = library->findAllSolutionsGroupedGemm(
            prob.gemms, *hardware, Tensile::SolutionLibrarySearchType::GEMM_TYPE_ONLY);
    }
    log_api(__func__, "Found hardware solutions: ", solutions.size());

    // when there is no solution for xfloat32, fallback comput_type to fp32
    if(solutions.size() == 0 && prob.f32XdlMathOp() == Tensile::DataType::XFloat32)
    {
        prob.setF32XdlMathOp(Tensile::DataType::Float);
        if constexpr(std::is_same<MyProblem, Tensile::ContractionProblemGemm>::value)
        {
            solutions = library->findAllSolutions(
                prob, *hardware, Tensile::SolutionLibrarySearchType::GEMM_TYPE_ONLY);
        }
        else if constexpr(std::is_same<MyProblem, Tensile::ContractionProblemGroupedGemm>::value)
        {
            solutions = library->findAllSolutionsGroupedGemm(
                prob.gemms, *hardware, Tensile::SolutionLibrarySearchType::GEMM_TYPE_ONLY);
        }
    }

    heuristicResults.resize(solutions.size());

    int i = 0;
    for(auto solution : solutions)
    {
        memset(&heuristicResults[i], 0, sizeof(rocblaslt_matmul_heuristic_result));
        memset(heuristicResults[i].algo.data, 0, sizeof(heuristicResults[i].algo.data));
        int* solutionIndex                           = (int*)(heuristicResults[i].algo.data);
        *solutionIndex                               = solution->index;
        heuristicResults[i].algo.max_workspace_bytes = maxWorkSpaceBytes;
        heuristicResults[i].algo.fallback            = false;
        heuristicResults[i].state                    = rocblaslt_status_success;
        heuristicResults[i].workspaceSize            = 0;
        i++;
    }

    return rocblaslt_status_success;
}

template <typename TiA, typename TiB, typename To, typename Tc>
rocblaslt_status getAllSolutions(RocblasltContractionProblem<TiA, TiB, To, Tc>&  prob,
                                 rocblaslt_handle                                handle,
                                 std::vector<rocblaslt_matmul_heuristic_result>& heuristicResults,
                                 size_t                                          maxWorkSpaceBytes)
{
    auto tensile_prob = ConstructTensileProblem(prob);
    return getAllSolutions(tensile_prob, handle, heuristicResults, maxWorkSpaceBytes);
}

template <typename TiA, typename TiB, typename To, typename Tc>
rocblaslt_status getAllSolutions(std::vector<RocblasltContractionProblem<TiA, TiB, To, Tc>>& probs,
                                 rocblaslt_handle                                            handle,
                                 std::vector<rocblaslt_matmul_heuristic_result>& heuristicResults,
                                 size_t                                          maxWorkSpaceBytes)
{
    Tensile::ContractionProblemGroupedGemm tensile_probs;
    for(int i = 0; i < probs.size(); i++)
    {
        tensile_probs.gemms.push_back(ConstructTensileProblem(probs[i]));
        tensile_probs.gemms[i].setGroupedGemm(true);
    }
    return getAllSolutions(tensile_probs, handle, heuristicResults, maxWorkSpaceBytes);
}

rocblaslt_status
    getSolutionsFromIndex(rocblaslt_handle                                handle,
                          std::vector<int>&                               solutionIndex,
                          std::vector<rocblaslt_matmul_heuristic_result>& heuristicResults,
                          size_t                                          maxWorkSpaceBytes)
{
    std::shared_ptr<Tensile::MasterSolutionLibrary<Tensile::ContractionProblemGemm>> library;
    std::shared_ptr<hipDeviceProp_t>                                                 deviceProp;
    std::shared_ptr<Tensile::Hardware>                                               hardware;

    auto adapter = get_library_and_adapter(&library, &deviceProp, handle->device);
    hardware     = Tensile::hip::GetDevice(*deviceProp);

    int i = 0;
    for(auto index : solutionIndex)
    {
        auto solution = library->getSolutionByIndex(index);
        if(!solution)
            continue;
        rocblaslt_matmul_heuristic_result result;
        memset(&result, 0, sizeof(rocblaslt_matmul_heuristic_result));
        memset(result.algo.data, 0, sizeof(result.algo.data));
        int* solutionIndex              = (int*)(result.algo.data);
        *solutionIndex                  = solution->index;
        result.algo.max_workspace_bytes = maxWorkSpaceBytes;
        result.algo.fallback            = false;
        result.state                    = rocblaslt_status_success;
        result.workspaceSize            = 0;
        i++;
        heuristicResults.push_back(result);
    }

    return rocblaslt_status_success;
}

template <typename MyProblem, typename Inputs>
rocblaslt_status isSolutionSupported(rocblaslt_handle       handle,
                                     MyProblem&             tensile_prob,
                                     Inputs&                inputs,
                                     rocblaslt_matmul_algo* algo,
                                     size_t*                workspaceSizeInBytes)
{
    std::shared_ptr<Tensile::MasterSolutionLibrary<Tensile::ContractionProblemGemm>> library;
    std::shared_ptr<hipDeviceProp_t>                                                 deviceProp;
    std::shared_ptr<Tensile::Hardware>                                               hardware;

    auto adapter          = get_library_and_adapter(&library, &deviceProp, handle->device);
    hardware              = Tensile::hip::GetDevice(*deviceProp);
    *workspaceSizeInBytes = 0;

    int* solutionIndex = (int*)algo->data;
    // don't overwrite data->algoIndex = *solutionIndex; here
    if constexpr(std::is_same<MyProblem, Tensile::ContractionProblemGemm>::value)
    {
        auto        solution = library->getSolutionByIndex(tensile_prob, *hardware, *solutionIndex);
        const void *scaleAlphaVec = nullptr, *bias = nullptr, *E = nullptr;
        if constexpr(std::is_same<Inputs, Tensile::ContractionInputs>::value)
        {
            scaleAlphaVec = inputs.scaleAlphaVec;
            bias          = inputs.bias;
            E             = inputs.e;
        }
        else
        {
            scaleAlphaVec = inputs.scaleAlphaVec;
            bias          = inputs.bias;
            E             = inputs.E;
        }

        tensile_prob.setWorkspaceSize(algo->max_workspace_bytes);
        if(!(*solution->hardwarePredicate)(*hardware))
        {
            log_error(__func__, "Solution is not supported");
            return rocblaslt_status_invalid_value;
        }
        if(!(*solution->problemPredicate)(tensile_prob))
        {
            // Try fallback
            if(scaleAlphaVec == nullptr && bias == nullptr && E == nullptr
               && tensile_prob.activationEnumArg() == Tensile::ActivationType::None)
            {
                auto useBias          = tensile_prob.useBias();
                auto actType          = tensile_prob.activationType();
                auto useScaleAlphaVec = tensile_prob.useScaleAlphaVec();
                auto useE             = tensile_prob.useE();
                tensile_prob.setUseBias(false);
                tensile_prob.setActivationType(Tensile::ActivationType::None);
                tensile_prob.setUseScaleAlphaVec(false);
                tensile_prob.setUseE(false);
                bool isSup = (*solution->hardwarePredicate)(*hardware)
                             && (*solution->problemPredicate)(tensile_prob);
                if(isSup)
                    *workspaceSizeInBytes = solution->requiredWorkspaceSize(tensile_prob);
                tensile_prob.setUseBias(useBias);
                tensile_prob.setActivationType(actType);
                tensile_prob.setUseScaleAlphaVec(useScaleAlphaVec);
                tensile_prob.setUseE(useE);
                if(!isSup)
                {
                    log_error(__func__, "Solution is not supported");
                    return rocblaslt_status_invalid_value;
                }
                algo->fallback = true;
            }
            else
            {
                log_error(__func__, "Solution is not supported");
                return rocblaslt_status_invalid_value;
            }
        }
        else
        {
            *workspaceSizeInBytes = solution->requiredWorkspaceSize(tensile_prob);
        }
    }
    else if constexpr(std::is_same<MyProblem, Tensile::ContractionProblemGroupedGemm>::value)
    {
        auto solution
            = library->getSolutionByIndex(tensile_prob.gemms[0], *hardware, *solutionIndex);
        bool isSupported  = true;
        bool isNormalGemm = true;
        for(int i = 0; i < tensile_prob.gemms.size(); i++)
        {
            tensile_prob.gemms[i].setWorkspaceSize(algo->max_workspace_bytes);
            if(!((*solution->hardwarePredicate)(*hardware)
                 && (*solution->problemPredicate)(tensile_prob.gemms[i])))
            {
#if 0
                std::cout << "Match " << "[" << i << "]: " << solution->description();
                solution->problemPredicate->debugEval(tensile_prob.gemms[i], std::cout);
                std::cout << std::endl;
#endif
                isSupported = false;
                break;
            }
        }
        for(int i = 0; i < tensile_prob.gemms.size(); i++)
        {
            const void *scaleAlphaVec = nullptr, *bias = nullptr, *E = nullptr;
            if constexpr(std::is_same<Inputs, Tensile::ContractionGroupedInputs>::value)
            {
                scaleAlphaVec = inputs.grouped[i].scaleAlphaVec;
                bias          = inputs.grouped[i].bias;
                E             = inputs.grouped[i].e;
            }
            else
            {
                throw std::runtime_error("Unsupported mode.");
            }

            if(scaleAlphaVec != nullptr || bias != nullptr || E != nullptr
               || tensile_prob.gemms[i].activationEnumArg() != Tensile::ActivationType::None)
            {
                isNormalGemm = false;
                break;
            }
        }
        if(isNormalGemm && !isSupported)
        {
            isSupported = true;
            for(int i = 0; i < tensile_prob.gemms.size(); i++)
            {
                auto useBias          = tensile_prob.gemms[i].useBias();
                auto actType          = tensile_prob.gemms[i].activationType();
                auto useScaleAlphaVec = tensile_prob.gemms[i].useScaleAlphaVec();
                auto useE             = tensile_prob.gemms[i].useE();
                tensile_prob.gemms[i].setUseBias(false);
                tensile_prob.gemms[i].setActivationType(Tensile::ActivationType::None);
                tensile_prob.gemms[i].setUseScaleAlphaVec(false);
                tensile_prob.gemms[i].setUseE(false);
                if(!((*solution->hardwarePredicate)(*hardware)
                     && (*solution->problemPredicate)(tensile_prob.gemms[i])))
                {
                    isSupported = false;
                }
                tensile_prob.gemms[i].setUseBias(useBias);
                tensile_prob.gemms[i].setActivationType(actType);
                tensile_prob.gemms[i].setUseScaleAlphaVec(useScaleAlphaVec);
                tensile_prob.gemms[i].setUseE(useE);
                if(!isSupported)
                {
                    break;
                }
            }
        }
        if(!isSupported)
        {
            log_error(__func__, "Solution is not supported");
            return rocblaslt_status_invalid_value;
        }
        *workspaceSizeInBytes = solution->requiredWorkspaceSizeGroupedGemm(tensile_prob.gemms);
    }
    return rocblaslt_status_success;
}

template <typename TiA, typename TiB, typename To, typename Tc>
rocblaslt_status isSolutionSupported(rocblaslt_handle                               handle,
                                     RocblasltContractionProblem<TiA, TiB, To, Tc>& prob,
                                     std::shared_ptr<void>                          gemmData,
                                     rocblaslt_matmul_algo*                         algo,
                                     size_t* workspaceSizeInBytes)
{
    std::shared_ptr<TensileDataGemm> data = std::static_pointer_cast<TensileDataGemm>(gemmData);
    updateTensileProblem(false, prob, data->problem);
    return isSolutionSupported(handle, data->problem, prob, algo, workspaceSizeInBytes);
}

template <typename Tc>
void setRestrictions(Tensile::ContractionProblemGemm& tensile_prob, const Tc* alpha, const Tc* beta)
{
    tensile_prob.setAlphaRestriction(Tensile::toScalarValueEnum(*alpha));
    tensile_prob.setBetaRestriction(Tensile::toScalarValueEnum(*beta));
}

rocblaslt_status isSolutionSupported(rocblaslt_handle              handle,
                                     const rocblaslt::RocGemmType& gemmType,
                                     std::shared_ptr<void>         gemmData,
                                     rocblaslt_matmul_algo&        algo,
                                     size_t&                       workspaceSizeInBytes)
{
    if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GEMM)
    {
        std::shared_ptr<TensileDataGemm> data = std::static_pointer_cast<TensileDataGemm>(gemmData);
        if(data->problem.computeType() == Tensile::DataType::Float)
        {
            setRestrictions<float>(data->problem,
                                   std::get_if<float>(&data->inputs.alpha),
                                   std::get_if<float>(&data->inputs.beta));
        }
        else
        {
            return rocblaslt_status_not_implemented;
        }
        return isSolutionSupported(
            handle, data->problem, data->inputs, &algo, &workspaceSizeInBytes);
    }
    else if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GROUPED_GEMM)
    {
        std::shared_ptr<TensileDataGroupedGemm> data
            = std::static_pointer_cast<TensileDataGroupedGemm>(gemmData);
        if(data->problem.gemms[0].computeType() == Tensile::DataType::Float)
        {
            for(int i = 0; i < data->problem.gemms.size(); i++)
            {
                auto& tensile_prob = data->problem.gemms[i];
                setRestrictions<float>(tensile_prob,
                                       std::get_if<float>(&data->inputs.grouped[i].alpha),
                                       std::get_if<float>(&data->inputs.grouped[i].beta));
            }
        }
        else
        {
            return rocblaslt_status_not_implemented;
        }
        return isSolutionSupported(
            handle, data->problem, data->inputs, &algo, &workspaceSizeInBytes);
    }
    return rocblaslt_status_not_implemented;
}

rocblaslt_status getBestSolutions(rocblaslt_handle       handle,
                                  rocblaslt::RocGemmType gemmType,
                                  std::shared_ptr<void>  gemmData,
                                  const int              workspaceBytes,
                                  const int              requestedAlgoCount,
                                  std::vector<rocblaslt_matmul_heuristic_result>& heuristicResults)
{
    std::shared_ptr<Tensile::MasterSolutionLibrary<Tensile::ContractionProblemGemm>> library;
    std::shared_ptr<hipDeviceProp_t>                                                 deviceProp;
    std::shared_ptr<Tensile::Hardware>                                               hardware;

    // auto &adapter =
    static_cast<void>(get_library_and_adapter(&library, &deviceProp, handle->device));

    hardware = Tensile::hip::GetDevice(*deviceProp);

    if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GEMM)
    {
        std::shared_ptr<TensileDataGemm> data = std::static_pointer_cast<TensileDataGemm>(gemmData);
        int                              fallbackSize = 0;
        auto                             solutions    = getSolutions(data->inputs,
                                      library,
                                      hardware,
                                      data->problem,
                                      data->enableEpilogue,
                                      requestedAlgoCount,
                                      fallbackSize);

        // when there is no solution for xfloat32, fallback comput_type to fp32
        if(solutions.size() == 0 && data->problem.f32XdlMathOp() == Tensile::DataType::XFloat32)
        {
            data->problem.setF32XdlMathOp(Tensile::DataType::Float);
            solutions = getSolutions(data->inputs,
                                     library,
                                     hardware,
                                     data->problem,
                                     data->enableEpilogue,
                                     requestedAlgoCount,
                                     fallbackSize);
        }

        auto algoCount       = min(requestedAlgoCount, solutions.size());
        int  returnAlgoCount = 0;
        heuristicResults.clear();
        heuristicResults.resize(algoCount);
        _convertToHeuristicResultArray(solutions,
                                       algoCount,
                                       heuristicResults.data(),
                                       &returnAlgoCount,
                                       workspaceBytes,
                                       data->problem,
                                       fallbackSize);
    }
    else if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GROUPED_GEMM)
    {
        std::shared_ptr<TensileDataGroupedGemm> data
            = std::static_pointer_cast<TensileDataGroupedGemm>(gemmData);
        for(int i = 0; i < data->problem.gemms.size(); i++)
        {
            data->problem.gemms[i].setWorkspaceSize(workspaceBytes);
        }

        // Fallback to original kernels
        std::vector<std::shared_ptr<Tensile::ContractionSolution>> solutions_fallback;
        std::vector<bool>                    useBias, actHPA, useScaleAlphaVec;
        std::vector<Tensile::ActivationType> actType;
        if(!data->enableEpilogue)
        {
            bool enableEpilogue = true;
            for(int i = 0; i < data->problem.gemms.size(); i++)
            {
                if(data->inputs.grouped[i].scaleAlphaVec != nullptr
                   || data->inputs.grouped[i].bias != nullptr
                   || data->problem.gemms[i].activationEnumArg() != Tensile::ActivationType::None)
                {
                    enableEpilogue = false;
                    break;
                }
            }
            if(enableEpilogue)
            {
                for(int i = 0; i < data->problem.gemms.size(); i++)
                {
                    useBias.push_back(data->problem.gemms[i].useBias());
                    actType.push_back(data->problem.gemms[i].activationType());
                    useScaleAlphaVec.push_back(data->problem.gemms[i].useScaleAlphaVec());
                    data->problem.gemms[i].setUseBias(false);
                    data->problem.gemms[i].setActivationType(Tensile::ActivationType::None);
                    data->problem.gemms[i].setUseScaleAlphaVec(false);
                }
                solutions_fallback = library->findTopSolutionsGroupedGemm(
                    data->problem.gemms, *hardware, requestedAlgoCount);
                for(int i = 0; i < data->problem.gemms.size(); i++)
                {
                    data->problem.gemms[i].setUseBias(useBias[i]);
                    data->problem.gemms[i].setActivationType(actType[i]);
                    data->problem.gemms[i].setUseScaleAlphaVec(useScaleAlphaVec[i]);
                }
            }
        }

        auto solutions = library->findTopSolutionsGroupedGemm(
            data->problem.gemms, *hardware, requestedAlgoCount - solutions_fallback.size());
        solutions.insert(solutions.begin(), solutions_fallback.begin(), solutions_fallback.end());

        auto algoCount       = min(requestedAlgoCount, solutions.size());
        int  returnAlgoCount = 0;
        heuristicResults.clear();
        heuristicResults.resize(algoCount);

        _convertToHeuristicResultArray(solutions,
                                       algoCount,
                                       heuristicResults.data(),
                                       &returnAlgoCount,
                                       workspaceBytes,
                                       data->problem.gemms[0],
                                       solutions_fallback.size());
    }

    return rocblaslt_status_success;
}
/***************************************************************
 * ! \brief  Initialize rocblaslt for the current HIP device, to *
 * avoid costly startup time at the first call on that device. *
 ***************************************************************/
extern "C" void rocblaslt_createialize()
{
    static_cast<void>(get_library_and_adapter());
}

/******************************************************************************
 * Intantiate the cases of runContractionProblem which are needed to satisfy  *
 * rocblaslt dependencies. This file's template functions are not defined in a *
 * header file, in order to keep Tensile and rocblaslt separate. *
 ******************************************************************************/

// types
#define CREATEFUNCTION(TiA, TiB, To, Tc)                                                       \
    template rocblaslt_status runContractionProblem<TiA, TiB, To, Tc>(                         \
        rocblaslt_handle             handle,                                                   \
        const rocblaslt_matmul_algo* algo,                                                     \
        const RocblasltContractionProblem<TiA, TiB, To, Tc>&,                                  \
        std::shared_ptr<void>);                                                                \
    template rocblaslt_status gemmCreate(const RocblasltContractionProblem<TiA, TiB, To, Tc>&, \
                                         std::shared_ptr<void>& gemmData,                      \
                                         size_t&                gemmCount);                                   \
    template rocblaslt_status groupedGemmCreate<TiA, TiB, To, Tc>(                             \
        std::vector<RocblasltContractionProblem<TiA, TiB, To, Tc>>&,                           \
        std::shared_ptr<void>&,                                                                \
        size_t&);                                                                              \
    template rocblaslt_status getAllSolutions(                                                 \
        RocblasltContractionProblem<TiA, TiB, To, Tc>&  prob,                                  \
        rocblaslt_handle                                handle,                                \
        std::vector<rocblaslt_matmul_heuristic_result>& heuristicResults,                      \
        size_t                                          maxWorkSpaceBytes);                                                             \
    template rocblaslt_status getAllSolutions(                                                 \
        std::vector<RocblasltContractionProblem<TiA, TiB, To, Tc>>& probs,                     \
        rocblaslt_handle                                            handle,                    \
        std::vector<rocblaslt_matmul_heuristic_result>&             heuristicResults,          \
        size_t                                                      maxWorkSpaceBytes);                                                             \
    template rocblaslt_status isSolutionSupported(                                             \
        rocblaslt_handle                               handle,                                 \
        RocblasltContractionProblem<TiA, TiB, To, Tc>& prob,                                   \
        std::shared_ptr<void>                          gemmData,                               \
        rocblaslt_matmul_algo*                         algo,                                   \
        size_t*                                        workspaceSizeInBytes);                                                         \
    template rocblaslt_status getBestSolutions<TiA, TiB, To, Tc>(                              \
        RocblasltContractionProblem<TiA, TiB, To, Tc> prob,                                    \
        rocblaslt_handle                              handle,                                  \
        std::shared_ptr<void>                         gemmData,                                \
        int                                           requestedAlgoCount,                      \
        rocblaslt_matmul_heuristic_result             heuristicResultsArray[],                 \
        int*                                          returnAlgoCount,                         \
        size_t                                        maxWorkSpaceBytes);

CREATEFUNCTION(float, float, float, float)
CREATEFUNCTION(double, double, double, double)
CREATEFUNCTION(rocblaslt_half, rocblaslt_half, rocblaslt_half, float)
CREATEFUNCTION(rocblaslt_half, rocblaslt_half, float, float)
CREATEFUNCTION(rocblaslt_bfloat16, rocblaslt_bfloat16, rocblaslt_bfloat16, float)
CREATEFUNCTION(rocblaslt_f8, rocblaslt_f8, float, float)
CREATEFUNCTION(rocblaslt_f8, rocblaslt_bf8, float, float)
CREATEFUNCTION(rocblaslt_bf8, rocblaslt_f8, float, float)
CREATEFUNCTION(rocblaslt_f8, rocblaslt_f8, rocblaslt_half, float)
CREATEFUNCTION(rocblaslt_f8, rocblaslt_bf8, rocblaslt_half, float)
CREATEFUNCTION(rocblaslt_bf8, rocblaslt_f8, rocblaslt_half, float)
CREATEFUNCTION(rocblasltInt8, rocblasltInt8, int32_t, int32_t)
CREATEFUNCTION(rocblasltInt8, rocblasltInt8, rocblasltInt8, int32_t)
CREATEFUNCTION(rocblaslt_f8, rocblaslt_half, rocblaslt_half, float)
CREATEFUNCTION(rocblaslt_half, rocblaslt_f8, rocblaslt_half, float)

/***********************************************************************************
 * Whether Tensile has been initialized for at least one device (used for
 *testing) *
 ***********************************************************************************/
std::atomic_bool& rocblaslt_internal_tensile_is_initialized()
{
    static std::atomic_bool init;
    return init;
}
