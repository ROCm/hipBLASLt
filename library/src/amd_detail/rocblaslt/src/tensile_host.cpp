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

namespace
{
#ifndef WIN32
    std::string hipblaslt_so_path;

    int hipblaslt_dl_iterate_phdr_callback(struct dl_phdr_info* hdr_info, size_t size, void* data)
    {
        // uncomment to see all dependent .so files
        // fprintf(stderr, "hipblaslt so file: %s\n", hdr_info->dlpi_name);
        if(hdr_info->dlpi_name && strstr(hdr_info->dlpi_name, "libhipblaslt"))
        {
            hipblaslt_so_path = hdr_info->dlpi_name;
        }
        return 0;
    }
#endif
    /******************************************************
 * Map a hipblas data type to a corresponding Tensile type *
 ******************************************************/
    inline Tensile::DataType hipblasDatatype_to_tensile_type(hipblasDatatype_t type)
    {
        switch(type)
        {
        case HIPBLAS_R_16F:
            return Tensile::DataType::Half;
        case HIPBLAS_R_32F:
            return Tensile::DataType::Float;
        case HIPBLAS_R_16B:
            return Tensile::DataType::BFloat16;
        default:
            assert(!"hipblasDatatype_to_tensile_type: non-supported type");
            return Tensile::DataType::None;
        }
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
    constexpr auto tensile_datatype<rocblaslt_bfloat16> = Tensile::DataType::BFloat16;

    /*************************************************************************
 * Class for converting alpha and beta between rocblaslt and Tensile types *
 * By default, alpha and beta are the same type as Tc compute_type       *
 *************************************************************************/
    template <typename Ti, typename To = Ti, typename Tc = To>
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

    Tensile::DataType hip2TensileType(hipblasDatatype_t type)
    {
        switch(type)
        {
        case HIPBLAS_R_32F:
            return Tensile::DataType::Float;
        case HIPBLAS_R_16F:
            return Tensile::DataType::Half;
        case HIPBLAS_R_16B:
            return Tensile::DataType::BFloat16;
        default:
            throw std::runtime_error("Unsupported type.");
        }
        return Tensile::DataType::None;
    }

    Tensile::DataType roc2TensileType(rocblaslt_compute_type type)
    {
        switch(type)
        {
        case rocblaslt_compute_f32:
            return Tensile::DataType::Float;
        default:
            throw std::runtime_error("Unsupported type.");
        }
        return Tensile::DataType::None;
    }

    auto CreateTensileProblem(hipblasOperation_t     opA,
                              hipblasOperation_t     opB,
                              hipblasDatatype_t      typeA,
                              hipblasDatatype_t      typeB,
                              hipblasDatatype_t      typeC,
                              hipblasDatatype_t      typeD,
                              rocblaslt_compute_type typeCompute,
                              float                  alpha,
                              float                  beta,
                              size_t                 maxWorkspaceBytes)
    {
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

        size_t m = 1, n = 1, k = 1;
        size_t batch_count = 1;

        auto tensileAType = hip2TensileType(typeA);
        // clang-format off

        // If A is transposed, swap the free and bound dimensions and their ranks
        if(opA != HIPBLAS_OP_N)
        {
            a = {
                    "a",
                    tensileAType,
                    {k, m, batch_count},
                    {1, k, k * m}
                };
            freeIndex[0].i  = 1;
            boundIndex[0].a = 0;
        }
        else
        {
            a = {
                    "a",
                    tensileAType,
                    {m, k, batch_count},
                    {1, m, m * k}
                };
            freeIndex[0].i  = 0;
            boundIndex[0].a = 1;
        }

        // If B is transposed, swap the free and bound dimensions and their ranks
        if(opB != HIPBLAS_OP_N)
        {
            b = {
                    "b",
                    hip2TensileType(typeB),
                    {n, k, batch_count},
                    {1, n, n * k}
                };
            freeIndex[1].i  = 0;
            boundIndex[0].b = 1;
        }
        else
        {
            b = {
                    "b",
                    hip2TensileType(typeB),
                    {k, n, batch_count},
                    {1, k, k * n}
                };
            freeIndex[1].i  = 1;
            boundIndex[0].b = 0;
        }

        // clang-format on

        // Descriptor for input matrix C
        Tensile::TensorDescriptor c{
            "c", hip2TensileType(typeC), {m, n, batch_count}, {1, m, m * n}};

        // Descriptor for output matrix D
        Tensile::TensorDescriptor d{
            "d", hip2TensileType(typeD), {m, n, batch_count}, {1, m, m * n}};

        Tensile::TensorDescriptor e{"e"};
        Tensile::TensorDescriptor bias{"bias"};
        Tensile::TensorDescriptor scaleD{"scaleD"};

        // The ContractionProblemGemm
        Tensile::ContractionProblemGemm tensileProblem{a,
                                                       b,
                                                       c,
                                                       d,
                                                       e,
                                                       bias,
                                                       scaleD,
                                                       freeIndex,
                                                       batchIndex,
                                                       boundIndex,
                                                       beta,
                                                       maxWorkspaceBytes};

        auto tensileComputeType = roc2TensileType(typeCompute);
        tensileProblem.setAlphaType(tensileComputeType);
        tensileProblem.setBetaType(tensileComputeType);

        // HPA is active iff sizeof(compute type) > sizeof(input type)
        tensileProblem.setHighPrecisionAccumulate(Tensile::GetElementSize(tensileComputeType)
                                                  > Tensile::GetElementSize(tensileAType));

        // set batch mode
        tensileProblem.setStridedBatched(true);
        tensileProblem.setGroupedGemm(false);

        tensileProblem.setAlphaRestriction(Tensile::toScalarValueEnum(alpha));

        // Add problem predicates for CEqualsD
        tensileProblem.setCEqualsD(false);
        return tensileProblem;
    }

    /****************************************************************
 * Construct a Tensile Problem from a RocblasltContractionProblem *
 ****************************************************************/
    // Should remove this
    template <typename Ti, typename To, typename Tc>
    auto ConstructTensileProblem(const RocblasltContractionProblem<Ti, To, Tc>& prob)
    {
        // Tensile DataTypes corresponding to rocblaslt data types
        static constexpr Tensile::DataType Tensile_Ti = tensile_datatype<Ti>;
        static constexpr Tensile::DataType Tensile_To = tensile_datatype<To>;
        static constexpr Tensile::DataType Tensile_Tc = tensile_datatype<Tc>;

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
                    Tensile_Ti,
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
                    Tensile_Ti,
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
                    Tensile_Ti,
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
                    Tensile_Ti,
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
        Tensile::TensorDescriptor scaleD{"scaleD"};

        // The ContractionProblemGemm
        Tensile::ContractionProblemGemm tensileProblem{a,
                                                       b,
                                                       c,
                                                       d,
                                                       e,
                                                       bias,
                                                       scaleD,
                                                       freeIndex,
                                                       batchIndex,
                                                       boundIndex,
                                                       *prob.beta,
                                                       prob.workspaceSize};

        tensileProblem.setAlphaType(Tensile_Tc);
        tensileProblem.setBetaType(Tensile_Tc);

        // HPA is active iff sizeof(compute type) > sizeof(input type)
        tensileProblem.setHighPrecisionAccumulate(sizeof(Tc) > sizeof(Ti));

        // set batch mode
        tensileProblem.setStridedBatched(prob.strided_batch);
        tensileProblem.setGroupedGemm(prob.grouped_gemm);

        // alpha and beta are stored by value in Tensile::TypedContractionInputs
        // alpha and beta are copied from host to Tensile::TypedContractionInputs
        // If k==0, we do not need to dereference prob.alpha and can set
        // tensileAlpha=0 Not positive if this is necessary here as well
        typename AlphaBeta<Ti, To, Tc>::tensile_type tensileAlpha;
        if(prob.k)
            AlphaBeta<Ti, To, Tc>::copy(&tensileAlpha, prob.alpha);
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
            hipblasDatatype_to_tensile_type(prob.bias_type), biasSize, prob.gradient, biasSrc);

        // set ScaleD mode
        tensileProblem.setUseScaleD(true);
        tensileProblem.setScaleD(Tensile_Tc, d.sizes()[0]);

        // set Actvation
        if(is_act_enabled(prob.epilogue))
        {
            tensileProblem.setActivationType(Tensile::ActivationType::All);
            tensileProblem.setActivationHPA(sizeof(Tc) > sizeof(Ti));
            tensileProblem.setActivationEnumArg(getTensileActivationType(prob.epilogue));
        }
        else
        {
            tensileProblem.setActivationType(Tensile::ActivationType::None);
            tensileProblem.setActivationHPA(false);
        }

        // set use gradient
        tensileProblem.setUseGradient(is_grad_enabled(prob.epilogue));
        return tensileProblem;
    }

    template <typename Ti, typename To, typename Tc>
    void updateTensileProblem(const bool                                     fallback,
                              const RocblasltContractionProblem<Ti, To, Tc>& prob,
                              Tensile::ContractionProblemGemm&               tensileProblem)
    {
        // Tensile DataTypes corresponding to rocblaslt data types
        static constexpr Tensile::DataType Tensile_Tc = tensile_datatype<Tc>;

        // Should we update tensor only if the size changed? Or should we just
        // return error if size changed?

        // We set K=0 when alpha==0.
        // This makes alpha==0 a change in the problem, and not just a change in the
        // inputs. It optimizes all problems with alpha==0 into K=0 and alpha=(don't
        // care)
        auto k = prob.k; // && *prob.alpha ? prob.k : 0;

        // clang-format off

        // If A is transposed, swap the free and bound dimensions and their ranks
        if(prob.trans_a != HIPBLAS_OP_N)
        {
            tensileProblem.resizeTensor(Tensile::ContractionProblemGemm::TENSOR::A,
                    {k, prob.m, prob.batch_count},
                    {prob.row_stride_a, prob.col_stride_a, prob.batch_stride_a});
        }
        else
        {
            tensileProblem.resizeTensor(Tensile::ContractionProblemGemm::TENSOR::A,
                    {prob.m, k, prob.batch_count},
                    {prob.row_stride_a, prob.col_stride_a, prob.batch_stride_a});
        }

        // If B is transposed, swap the free and bound dimensions and their ranks
        if(prob.trans_b != HIPBLAS_OP_N)
        {
            tensileProblem.resizeTensor(Tensile::ContractionProblemGemm::TENSOR::B,
                    {prob.n, k, prob.batch_count},
                    {prob.row_stride_b, prob.col_stride_b, prob.batch_stride_b});
        }
        else
        {
            tensileProblem.resizeTensor(Tensile::ContractionProblemGemm::TENSOR::B,
                    {k, prob.n, prob.batch_count},
                    {prob.row_stride_b, prob.col_stride_b, prob.batch_stride_b});
        }

        // clang-format on

        // Descriptor for input matrix C
        tensileProblem.resizeTensor(Tensile::ContractionProblemGemm::TENSOR::C,
                                    {prob.m, prob.n, prob.batch_count},
                                    {prob.row_stride_c, prob.col_stride_c, prob.batch_stride_c});

        // Descriptor for output matrix D
        tensileProblem.resizeTensor(Tensile::ContractionProblemGemm::TENSOR::D,
                                    {prob.m, prob.n, prob.batch_count},
                                    {prob.row_stride_d, prob.col_stride_d, prob.batch_stride_d});

        tensileProblem.setBetaRestriction(Tensile::toScalarValueEnum((double)(*prob.beta)));
        tensileProblem.updateProblem();

        tensileProblem.setAlphaType(Tensile_Tc);
        tensileProblem.setBetaType(Tensile_Tc);

        // HPA is active iff sizeof(compute type) > sizeof(input type)
        tensileProblem.setHighPrecisionAccumulate(sizeof(Tc) > sizeof(Ti));

        // set batch mode
        tensileProblem.setStridedBatched(prob.strided_batch);

        // alpha and beta are stored by value in Tensile::TypedContractionInputs
        // alpha and beta are copied from host to Tensile::TypedContractionInputs
        // If k==0, we do not need to dereference prob.alpha and can set
        // tensileAlpha=0 Not positive if this is necessary here as well
        typename AlphaBeta<Ti, To, Tc>::tensile_type tensileAlpha;
        if(prob.k)
            AlphaBeta<Ti, To, Tc>::copy(&tensileAlpha, prob.alpha);
        else
            memset(&tensileAlpha, 0, sizeof(tensileAlpha));
        tensileProblem.setAlphaRestriction(Tensile::toScalarValueEnum(tensileAlpha));

        // Add problem predicates for CEqualsD
        tensileProblem.setCEqualsD(prob.C == prob.D);

        auto tensileAct = getTensileActivationType(prob.epilogue);
        tensileProblem.setActivationEnumArg(tensileAct);

        if(fallback && prob.bias == nullptr && prob.scaleD == nullptr && prob.E == nullptr
           && tensileAct == Tensile::ActivationType::None)
        {
            tensileProblem.setUseBias(false);
            tensileProblem.setActivationType(Tensile::ActivationType::None);
            tensileProblem.setActivationHPA(false);
            tensileProblem.setUseScaleD(false);
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
            tensileProblem.setBias(
                hipblasDatatype_to_tensile_type(prob.bias_type), biasSize, prob.gradient, biasSrc);

            // set ScaleD mode
            tensileProblem.setUseScaleD(true);
            tensileProblem.setScaleD(Tensile_Tc, d.sizes()[0]);

            // set Actvation
            if(is_act_enabled(prob.epilogue))
            {
                tensileProblem.setActivationType(Tensile::ActivationType::All);
                tensileProblem.setActivationHPA(sizeof(Tc) > sizeof(Ti));
            }
            else
            {
                tensileProblem.setActivationType(Tensile::ActivationType::None);
                tensileProblem.setActivationHPA(false);
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
    }

    /***************************************************************
 * Construct the inputs to a Tensile ContractionProblemGemm        *
 ***************************************************************/
    template <typename Ti, typename To, typename Tc>
    auto GetTensileInputs(const RocblasltContractionProblem<Ti, To, Tc>& prob)
    {
        // Tensile types corresponding to Ti, To, Tc
        using Tensile_Ti          = typename rocblaslt_to_tensile_type<Ti>::tensile_type;
        using Tensile_To          = typename rocblaslt_to_tensile_type<To>::tensile_type;
        using Tensile_Talpha_beta = typename AlphaBeta<Ti, To, Tc>::tensile_type;

        // Make sure rocblaslt and Tensile types are compatible
        // (Even if Ti=rocblaslt_int8x4, Tensile_Ti=Int8x4, they are both 32-byte)
        static_assert(sizeof(Tensile_Ti) == sizeof(Ti) && sizeof(Tensile_To) == sizeof(To),
                      "Tensile and rocblaslt types are not the same size");

        static_assert(std::is_standard_layout<Ti>{} && std::is_standard_layout<Tensile_Ti>{}
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
        inputs.bias   = reinterpret_cast<const void*>(prob.bias);
        inputs.scaleD = reinterpret_cast<const void*>(prob.scaleD);

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
#ifdef WIN32
            return ((_access(path.c_str(), 4) != -1) || (_access(path.c_str(), 6) != -1));
#else
            return access(path.c_str(), R_OK) == 0;
#endif
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
                dl_iterate_phdr(hipblaslt_dl_iterate_phdr_callback, NULL);
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
    Tensile::ContractionProblemGemm        problem;
    Tensile::ContractionInputs             inputs;
    std::vector<Tensile::KernelInvocation> kernels;
};

struct TensileDataGroupedGemm
{
    Tensile::ContractionProblemGroupedGemm problem;
    Tensile::ContractionGroupedInputs      inputs;
    std::vector<Tensile::KernelInvocation> kernels;
};

void initTensileGemmData(rocblaslt_handle       handle,
                         rocblaslt::RocGemmType gemmType,
                         hipblasOperation_t     opA,
                         hipblasOperation_t     opB,
                         hipblasDatatype_t      typeA,
                         hipblasDatatype_t      typeB,
                         hipblasDatatype_t      typeC,
                         hipblasDatatype_t      typeD,
                         rocblaslt_compute_type typeCompute,
                         size_t                 maxWorkspaceBytes,
                         std::shared_ptr<void>& gemmData)
{
    float alpha = 1.0;
    float beta  = 1.0;
    if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GEMM)
    {
        TensileDataGemm data;
        data.problem = CreateTensileProblem(
            opA, opB, typeA, typeB, typeC, typeD, typeCompute, alpha, beta, maxWorkspaceBytes);
        gemmData = std::static_pointer_cast<void>(std::make_shared<TensileDataGemm>(data));
        return;
    }
    else if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GROUPED_GEMM)
    {
        TensileDataGroupedGemm                  data;
        Tensile::ContractionProblemGroupedGemm& tensile_probs = data.problem;
        Tensile::ContractionGroupedInputs&      groupedInputs = data.inputs;

        tensile_probs.gemms.push_back(CreateTensileProblem(
            opA, opB, typeA, typeB, typeC, typeD, typeCompute, alpha, beta, maxWorkspaceBytes));
        groupedInputs.grouped.resize(1);

        gemmData = std::static_pointer_cast<void>(std::make_shared<TensileDataGroupedGemm>(data));
        return;
    }

    throw std::runtime_error("Gemm problem type initialization not implemented.");
}

/******************************************************************************
 * runContractionProblem calls Tensile to run a contraction problem described *
 * by RocblasltContractionProblem *
 ******************************************************************************/
template <typename Ti, typename To, typename Tc>
rocblaslt_status runContractionProblem(rocblaslt_handle                               handle,
                                       const rocblaslt_matmul_algo*                   algo,
                                       const RocblasltContractionProblem<Ti, To, Tc>& prob)
{
    rocblaslt_status status = rocblaslt_status_internal_error;
    try
    {
        std::shared_ptr<Tensile::MasterSolutionLibrary<Tensile::ContractionProblemGemm>> library;
        std::shared_ptr<hipDeviceProp_t>                                                 deviceProp;
        std::shared_ptr<Tensile::Hardware>                                               hardware;

        auto adapter = get_library_and_adapter(&library, &deviceProp, handle->device);

        hardware = Tensile::hip::GetDevice(*deviceProp);
        std::shared_ptr<Tensile::ContractionProblemGemm> tensile_prob;
        if(!algo->data2.ptr)
        {
            log_api(__func__, "Algo not initialized, may impact performance.");
            tensile_prob
                = std::make_shared<Tensile::ContractionProblemGemm>(ConstructTensileProblem(prob));
            // Try fallback
            if(algo->fallback && prob.scaleD == nullptr && prob.bias == nullptr && prob.E == nullptr
               && tensile_prob->activationEnumArg() == Tensile::ActivationType::None)
            {
                tensile_prob->setUseBias(false);
                tensile_prob->setActivationType(Tensile::ActivationType::None);
                tensile_prob->setActivationHPA(false);
                tensile_prob->setUseScaleD(false);
                tensile_prob->setUseE(false);
            }
        }
        else
        {
            tensile_prob
                = std::static_pointer_cast<Tensile::ContractionProblemGemm>(algo->data2.ptr);
            updateTensileProblem(algo->fallback, prob, *tensile_prob);
        }

        std::shared_ptr<Tensile::ContractionSolution> solution
            = std::static_pointer_cast<Tensile::ContractionSolution>(algo->data.ptr);

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
                solution->solve((*tensile_prob), GetTensileInputs(prob), *hardware),
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

template <typename Ti, typename To, typename Tc>
rocblaslt_status gemmCreate(RocblasltContractionProblem<Ti, To, Tc> const& problem,
                            std::shared_ptr<void>&                         gemmData,
                            size_t&                                        gemmCount)
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
            data->inputs = GetTensileInputs(problem);
        }
        else
        {
            TensileDataGemm data;
            data.problem = ConstructTensileProblem(problem);
            data.inputs  = GetTensileInputs(problem);

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

template <typename Ti, typename To, typename Tc>
rocblaslt_status groupedGemmCreate(std::vector<RocblasltContractionProblem<Ti, To, Tc>>& probs,
                                   std::shared_ptr<void>&                                gemmData,
                                   size_t&                                               gemmCount)
{
    gemmCount = probs.size();
    if(gemmCount == 0)
        return rocblaslt_status_success;
    rocblaslt_status status = rocblaslt_status_internal_error;
    try
    {
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
                // Check if pointer is valid
                if(probs[i].alpha == nullptr || probs[i].beta == nullptr || probs[i].A == nullptr
                   || probs[i].B == nullptr || probs[i].C == nullptr || probs[i].D == nullptr)
                {
                    log_error(__func__, "invalid data pointer");
                    return rocblaslt_status_invalid_pointer;
                }
                if(tensile_probs.gemms.size() != probs.size())
                    tensile_probs.gemms.push_back(ConstructTensileProblem(probs[i]));
                else
                    updateTensileProblem(false, probs[i], tensile_probs.gemms[i]);
                groupedInputs.grouped.push_back(GetTensileInputs(probs[i]));
            }
        }
        else
        {
            TensileDataGroupedGemm                  data;
            Tensile::ContractionProblemGroupedGemm& tensile_probs = data.problem;
            Tensile::ContractionGroupedInputs&      groupedInputs = data.inputs;

            for(int i = 0; i < probs.size(); i++)
            {
                // Check if pointer is valid
                if(probs[i].alpha == nullptr || probs[i].beta == nullptr || probs[i].A == nullptr
                   || probs[i].B == nullptr || probs[i].C == nullptr || probs[i].D == nullptr)
                {
                    log_error(__func__, "invalid data pointer");
                    return rocblaslt_status_invalid_pointer;
                }
                tensile_probs.gemms.push_back(ConstructTensileProblem(probs[i]));
                groupedInputs.grouped.push_back(GetTensileInputs(probs[i]));
            }

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

rocblaslt_status makeArgument(const rocblaslt::RocGemmType gemmType,
                              const rocblaslt_matmul_algo& algo,
                              void*                        workspace,
                              hipStream_t                  stream,
                              std::shared_ptr<void>        gemmData)
{
    rocblaslt_status status = rocblaslt_status_internal_error;
    try
    {
        if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GEMM)
        {
            std::shared_ptr<Tensile::Hardware> hardware;

            std::shared_ptr<Tensile::ContractionSolution> solution
                = std::static_pointer_cast<Tensile::ContractionSolution>(algo.data.ptr);

            std::shared_ptr<TensileDataGemm> data
                = std::static_pointer_cast<TensileDataGemm>(gemmData);

            data->inputs.ws = workspace;

            // Backup and restore settings
            bool                    useBias   = data->problem.useBias();
            Tensile::ActivationType actType   = data->problem.activationType();
            bool                    useActHPA = data->problem.activationHPA();
            bool                    useScaleD = data->problem.useScaleD();
            bool                    useE      = data->problem.useE();
            bool                    useGrad   = data->problem.useGradient();
            data->problem.setUseBias(solution->problemType.useBias);
            data->problem.setActivationType(solution->problemType.activationType);
            data->problem.setActivationHPA(solution->problemType.activationHPA);
            data->problem.setUseScaleD(solution->problemType.useScaleD);
            data->problem.setUseE(solution->problemType.useE);
            data->problem.setUseGradient(solution->problemType.useGradient);
            data->kernels = solution->solve(data->problem, data->inputs, *hardware);
            data->problem.setUseBias(useBias);
            data->problem.setActivationType(actType);
            data->problem.setActivationHPA(useActHPA);
            data->problem.setUseScaleD(useScaleD);
            data->problem.setUseE(useE);
            data->problem.setUseGradient(useGrad);
        }
        else if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GROUPED_GEMM)
        {
            std::shared_ptr<Tensile::Hardware> hardware;

            std::shared_ptr<Tensile::ContractionSolution> solution
                = std::static_pointer_cast<Tensile::ContractionSolution>(algo.data.ptr);

            std::shared_ptr<TensileDataGroupedGemm> data
                = std::static_pointer_cast<TensileDataGroupedGemm>(gemmData);

            for(int i = 0; i < data->inputs.grouped.size(); i++)
            {
                data->inputs.grouped[i].ws = workspace;
            }
            data->inputs.ws = workspace;

            // fallback to normal gemm if is normal kernel
            std::vector<bool>                    useBias, actHPA, useScaleD;
            std::vector<Tensile::ActivationType> actType;
            for(int i = 0; i < data->problem.gemms.size(); i++)
            {
                useBias.push_back(data->problem.gemms[i].useBias());
                actType.push_back(data->problem.gemms[i].activationType());
                actHPA.push_back(data->problem.gemms[i].activationHPA());
                useScaleD.push_back(data->problem.gemms[i].useScaleD());
                data->problem.gemms[i].setUseBias(solution->problemType.useBias);
                data->problem.gemms[i].setActivationType(solution->problemType.activationType);
                data->problem.gemms[i].setActivationHPA(solution->problemType.activationHPA);
                data->problem.gemms[i].setUseScaleD(solution->problemType.useScaleD);
            }
            data->kernels
                = solution->solveGroupedGemm(data->problem.gemms, data->inputs, *hardware, stream);
            for(int i = 0; i < data->problem.gemms.size(); i++)
            {
                data->problem.gemms[i].setUseBias(useBias[i]);
                data->problem.gemms[i].setActivationType(actType[i]);
                data->problem.gemms[i].setActivationHPA(actHPA[i]);
                data->problem.gemms[i].setUseScaleD(useScaleD[i]);
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

/******************************************************************************
 * ConstructRocblasltProblem creates RocblasltContractionProblem from mat     *
 * layout and descriptor for Tensile's findTopSolutions.                      *
 ******************************************************************************/
template <typename Ti, typename To, typename Tc>
RocblasltContractionProblem<Ti, To, Tc>
    ConstructRocblasltProblem(const rocblaslt_matmul_desc matmul_descr,
                              rocblaslt_matrix_layout     matA,
                              rocblaslt_matrix_layout     matB,
                              rocblaslt_matrix_layout     matC,
                              rocblaslt_matrix_layout     matD,
                              const Tc*                   alpha,
                              const Tc*                   beta,
                              size_t                      maxWorkSpaceBytes)
{
    rocblaslt_status   isValid   = rocblaslt_status_continue;
    bool               gradient  = false;
    hipblasOperation_t opA       = matmul_descr->op_A;
    hipblasOperation_t opB       = matmul_descr->op_B;
    const void*        e         = nullptr;
    const void*        bias      = nullptr;
    hipblasDatatype_t  bias_type = matmul_descr->bias_type == static_cast<hipblasDatatype_t>(0)
                                       ? matD->type
                                       : matmul_descr->bias_type;
    rocblaslt_epilogue epilogue  = matmul_descr->epilogue;
    if(is_e_enabled(epilogue))
    {
        if(matmul_descr->e == nullptr)
            isValid = rocblaslt_status_invalid_pointer;
        e = (void*)matmul_descr->e;
    }
    if(is_bias_enabled(epilogue))
        bias = (const void*)matmul_descr->bias;
    bool grouped_gemm = 0;

    const Tc* scaleD = nullptr;
    if(matmul_descr->scaleD)
        scaleD = (const Tc*)matmul_descr->scaleD;

    // matrix A
    int64_t num_rows_a     = matA->m;
    int64_t num_cols_a     = matA->n;
    int64_t lda            = matA->ld;
    int64_t batch_stride_a = matA->batch_stride;
    int     num_batches_a  = matA->batch_count;

    // matrix B
    int64_t ldb            = matB->ld;
    int64_t batch_stride_b = matB->batch_stride;
    int     num_batches_b  = matB->batch_count;

    // matrix C
    int64_t ldc            = matC->ld;
    int64_t batch_stride_c = matC->batch_stride;
    int     num_batches_c  = matC->batch_count;

    // matrix D
    int64_t num_rows_d     = matD->m;
    int64_t num_cols_d     = matD->n;
    int64_t ldd            = matD->ld;
    int64_t batch_stride_d = matD->batch_stride;
    int     num_batches_d  = matD->batch_count;

    // matrix E (Epilogue)
    int64_t num_rows_e     = num_rows_d;
    int64_t num_cols_e     = num_cols_d;
    int64_t lde            = matmul_descr->lde > 0 ? matmul_descr->lde : num_rows_e;
    int64_t batch_stride_e = matmul_descr->stride_e > 0 ? matmul_descr->stride_e : lde * num_cols_e;
    // int     num_batches_e  = num_batches_d;
    if(e != nullptr
       && ((matmul_descr->lde < num_rows_e)
           || (matmul_descr->stride_e < (num_cols_e * num_rows_e))))
        isValid = rocblaslt_status_invalid_value;

    int64_t m = num_rows_d;
    int64_t n = num_cols_d;
    int64_t k = (opA == HIPBLAS_OP_N) ? num_cols_a : num_rows_a;

    int8_t      dummy;
    const void* dummy_ptr = &dummy;
    if(isValid == rocblaslt_status_continue)
        isValid = validateMatmulArgs(m,
                                     n,
                                     k,
                                     dummy_ptr,
                                     dummy_ptr,
                                     dummy_ptr,
                                     dummy_ptr,
                                     dummy_ptr,
                                     dummy_ptr,
                                     num_batches_a,
                                     num_batches_b,
                                     num_batches_c,
                                     num_batches_d,
                                     batch_stride_a,
                                     batch_stride_b,
                                     batch_stride_c,
                                     batch_stride_d);
    if(isValid != rocblaslt_status_continue)
    {
        m = 0;
        n = 0;
        k = 0;
    }
    RocblasltContractionProblem<Ti, To, Tc> problem{opA,
                                                    opB,
                                                    m,
                                                    n,
                                                    k,
                                                    alpha,
                                                    nullptr,
                                                    nullptr,
                                                    lda,
                                                    batch_stride_a,
                                                    nullptr,
                                                    nullptr,
                                                    ldb,
                                                    batch_stride_b,
                                                    beta,
                                                    nullptr,
                                                    nullptr,
                                                    ldc,
                                                    batch_stride_c,
                                                    nullptr,
                                                    nullptr,
                                                    ldd,
                                                    batch_stride_d,
                                                    (Tc*)e,
                                                    nullptr,
                                                    lde,
                                                    batch_stride_e,
                                                    num_batches_a,
                                                    true,
                                                    grouped_gemm,
                                                    gradient,
                                                    bias,
                                                    scaleD,
                                                    bias_type,
                                                    epilogue,
                                                    nullptr,
                                                    maxWorkSpaceBytes,
                                                    nullptr};
    return problem;
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
        auto solution                           = solutions[i];
        heuristicResultsArray[i].algo.data.ptr  = std::static_pointer_cast<void>(solution);
        heuristicResultsArray[i].algo.data2.ptr = std::static_pointer_cast<void>(
            std::make_shared<Tensile::ContractionProblemGemm>(problem));
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
    const int&                                requestedAlgoCount,
    int&                                      fallbackSize)
{
    const void *scaleD = nullptr, *bias = nullptr, *E = nullptr;
    if constexpr(std::is_same<T, Tensile::ContractionInputs>::value)
    {
        scaleD = inputs.scaleD;
        bias   = inputs.bias;
        E      = inputs.e;
    }
    else
    {
        scaleD = inputs.scaleD;
        bias   = inputs.bias;
        E      = inputs.E;
    }

    std::vector<std::shared_ptr<Tensile::ContractionSolution>> solutions_fallback;
    // Fallback to original kernels
    if(scaleD == nullptr && bias == nullptr && E == nullptr
       && tensile_prob.activationEnumArg() == Tensile::ActivationType::None)
    {
        auto useBias   = tensile_prob.useBias();
        auto actType   = tensile_prob.activationType();
        auto actHPA    = tensile_prob.activationHPA();
        auto useScaleD = tensile_prob.useScaleD();
        auto useE      = tensile_prob.useE();
        tensile_prob.setUseBias(false);
        tensile_prob.setActivationType(Tensile::ActivationType::None);
        tensile_prob.setActivationHPA(false);
        tensile_prob.setUseScaleD(false);
        tensile_prob.setUseE(false);
        solutions_fallback = library->findTopSolutions(tensile_prob, *hardware, requestedAlgoCount);
        // restore
        tensile_prob.setUseBias(useBias);
        tensile_prob.setActivationType(actType);
        tensile_prob.setActivationHPA(actHPA);
        tensile_prob.setUseScaleD(useScaleD);
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

template <typename Ti, typename To, typename Tc>
rocblaslt_status getBestSolutions(RocblasltContractionProblem<Ti, To, Tc> prob,
                                  rocblaslt_handle                        handle,
                                  int                                     requestedAlgoCount,
                                  rocblaslt_matmul_heuristic_result       heuristicResultsArray[],
                                  int*                                    returnAlgoCount,
                                  size_t                                  maxWorkSpaceBytes)
{
    std::shared_ptr<Tensile::MasterSolutionLibrary<Tensile::ContractionProblemGemm>> library;
    std::shared_ptr<hipDeviceProp_t>                                                 deviceProp;
    std::shared_ptr<Tensile::Hardware>                                               hardware;

    // auto &adapter =
    static_cast<void>(get_library_and_adapter(&library, &deviceProp, handle->device));

    hardware          = Tensile::hip::GetDevice(*deviceProp);
    auto tensile_prob = ConstructTensileProblem(prob);
    int  fallbackSize = 0;
    auto solutions
        = getSolutions(prob, library, hardware, tensile_prob, requestedAlgoCount, fallbackSize);

    _convertToHeuristicResultArray(solutions,
                                   requestedAlgoCount,
                                   heuristicResultsArray,
                                   returnAlgoCount,
                                   maxWorkSpaceBytes,
                                   tensile_prob,
                                   fallbackSize);

    return rocblaslt_status_success;
}

template <typename MyProblem>
rocblaslt_status getAllSolutions(MyProblem&                          prob,
                                 rocblaslt_handle                    handle,
                                 rocblaslt_matmul_heuristic_result** heuristicResults,
                                 int*                                returnAlgoCount,
                                 size_t                              maxWorkSpaceBytes)
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
        solutions = library->findAllSolutions(prob, *hardware, true);
    }
    else if constexpr(std::is_same<MyProblem, Tensile::ContractionProblemGroupedGemm>::value)
    {
        solutions = library->findAllSolutionsGroupedGemm(prob.gemms, *hardware, true);
    }
    log_api(__func__, "Found hardware solutions: ", solutions.size());

    if(solutions.size() > 0)
        *heuristicResults = (rocblaslt_matmul_heuristic_result*)malloc(
            solutions.size() * sizeof(rocblaslt_matmul_heuristic_result));
    else
        *heuristicResults = NULL;

    int i = 0;
    for(auto solution : solutions)
    {
        memset(&(*heuristicResults)[i], 0, sizeof(rocblaslt_matmul_heuristic_result));
        (*heuristicResults)[i].algo.data.ptr = std::static_pointer_cast<void>(solution);
        if constexpr(std::is_same<MyProblem, Tensile::ContractionProblemGemm>::value)
        {
            (*heuristicResults)[i].algo.data2.ptr = std::static_pointer_cast<void>(
                std::make_shared<Tensile::ContractionProblemGemm>(prob));
        }
        else if constexpr(std::is_same<MyProblem, Tensile::ContractionProblemGroupedGemm>::value)
        {
            (*heuristicResults)[i].algo.data2.ptr = std::static_pointer_cast<void>(
                std::make_shared<Tensile::ContractionProblemGemm>(prob.gemms[0]));
        }
        (*heuristicResults)[i].algo.max_workspace_bytes = maxWorkSpaceBytes;
        (*heuristicResults)[i].algo.fallback            = false;
        (*heuristicResults)[i].state                    = rocblaslt_status_success;
        (*heuristicResults)[i].workspaceSize            = 0;
        i++;
    }
    *returnAlgoCount = solutions.size();

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
        solutions = library->findAllSolutions(prob, *hardware, true);
    }
    else if constexpr(std::is_same<MyProblem, Tensile::ContractionProblemGroupedGemm>::value)
    {
        solutions = library->findAllSolutionsGroupedGemm(prob.gemms, *hardware, true);
    }
    log_api(__func__, "Found hardware solutions: ", solutions.size());

    heuristicResults.resize(solutions.size());

    int i = 0;
    for(auto solution : solutions)
    {
        memset(&heuristicResults[i], 0, sizeof(rocblaslt_matmul_heuristic_result));
        heuristicResults[i].algo.data.ptr = std::static_pointer_cast<void>(solution);
        if constexpr(std::is_same<MyProblem, Tensile::ContractionProblemGemm>::value)
        {
            heuristicResults[i].algo.data2.ptr = std::static_pointer_cast<void>(
                std::make_shared<Tensile::ContractionProblemGemm>(prob));
        }
        else if constexpr(std::is_same<MyProblem, Tensile::ContractionProblemGroupedGemm>::value)
        {
            heuristicResults[i].algo.data2.ptr = std::static_pointer_cast<void>(
                std::make_shared<Tensile::ContractionProblemGemm>(prob.gemms[0]));
        }
        heuristicResults[i].algo.max_workspace_bytes = maxWorkSpaceBytes;
        heuristicResults[i].algo.fallback            = false;
        heuristicResults[i].state                    = rocblaslt_status_success;
        heuristicResults[i].workspaceSize            = 0;
        i++;
    }

    return rocblaslt_status_success;
}

template <typename Ti, typename To, typename Tc>
rocblaslt_status getAllSolutions(RocblasltContractionProblem<Ti, To, Tc>&        prob,
                                 rocblaslt_handle                                handle,
                                 std::vector<rocblaslt_matmul_heuristic_result>& heuristicResults,
                                 size_t                                          maxWorkSpaceBytes)
{
    auto tensile_prob = ConstructTensileProblem(prob);
    return getAllSolutions(tensile_prob, handle, heuristicResults, maxWorkSpaceBytes);
}

template <typename Ti, typename To, typename Tc>
rocblaslt_status getAllSolutions(std::vector<RocblasltContractionProblem<Ti, To, Tc>>& probs,
                                 rocblaslt_handle                                      handle,
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

template <typename MyProblem, typename Inputs>
rocblaslt_status isSolutionSupported(MyProblem&             tensile_prob,
                                     Inputs&                inputs,
                                     rocblaslt_matmul_algo* algo,
                                     size_t*                workspaceSizeInBytes)
{
    *workspaceSizeInBytes = 0;
    std::shared_ptr<Tensile::ContractionSolution> solution
        = std::static_pointer_cast<Tensile::ContractionSolution>(algo->data.ptr);

    if constexpr(std::is_same<MyProblem, Tensile::ContractionProblemGemm>::value)
    {
        const void *scaleD = nullptr, *bias = nullptr, *E = nullptr;
        if constexpr(std::is_same<Inputs, Tensile::ContractionInputs>::value)
        {
            scaleD = inputs.scaleD;
            bias   = inputs.bias;
            E      = inputs.e;
        }
        else
        {
            scaleD = inputs.scaleD;
            bias   = inputs.bias;
            E      = inputs.E;
        }

        tensile_prob.setWorkspaceSize(algo->max_workspace_bytes);
        if(!(*solution->problemPredicate)(tensile_prob))
        {
            // Try fallback
            if(scaleD == nullptr && bias == nullptr && E == nullptr
               && tensile_prob.activationEnumArg() == Tensile::ActivationType::None)
            {
                auto useBias   = tensile_prob.useBias();
                auto actType   = tensile_prob.activationType();
                auto actHPA    = tensile_prob.activationHPA();
                auto useScaleD = tensile_prob.useScaleD();
                auto useE      = tensile_prob.useE();
                tensile_prob.setUseBias(false);
                tensile_prob.setActivationType(Tensile::ActivationType::None);
                tensile_prob.setActivationHPA(false);
                tensile_prob.setUseScaleD(false);
                tensile_prob.setUseE(false);
                bool isSup = (*solution->problemPredicate)(tensile_prob);
                tensile_prob.setUseBias(useBias);
                tensile_prob.setActivationType(actType);
                tensile_prob.setActivationHPA(actHPA);
                tensile_prob.setUseScaleD(useScaleD);
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
        *workspaceSizeInBytes = solution->requiredWorkspaceSize(tensile_prob);
    }
    else if constexpr(std::is_same<MyProblem, Tensile::ContractionProblemGroupedGemm>::value)
    {
        bool isSupported  = true;
        bool isNormalGemm = true;
        for(int i = 0; i < tensile_prob.gemms.size(); i++)
        {
            tensile_prob.gemms[i].setWorkspaceSize(algo->max_workspace_bytes);
            if(!(*solution->problemPredicate)(tensile_prob.gemms[i]))
            {
                isSupported = false;
                break;
            }
        }
        for(int i = 0; i < tensile_prob.gemms.size(); i++)
        {
            const void *scaleD = nullptr, *bias = nullptr, *E = nullptr;
            if constexpr(std::is_same<Inputs, Tensile::ContractionGroupedInputs>::value)
            {
                scaleD = inputs.grouped[i].scaleD;
                bias   = inputs.grouped[i].bias;
                E      = inputs.grouped[i].e;
            }
            else
            {
                throw std::runtime_error("Unsupported mode.");
            }

            if(scaleD != nullptr || bias != nullptr || E != nullptr
               || tensile_prob.gemms[i].activationEnumArg() != Tensile::ActivationType::None)
            {
                isNormalGemm = false;
                break;
            }
        }
        bool isFallbackSupported = true;
        if(isNormalGemm && !isSupported)
        {
            for(int i = 0; i < tensile_prob.gemms.size(); i++)
            {
                auto useBias   = tensile_prob.gemms[i].useBias();
                auto actType   = tensile_prob.gemms[i].activationType();
                auto actHPA    = tensile_prob.gemms[i].activationHPA();
                auto useScaleD = tensile_prob.gemms[i].useScaleD();
                tensile_prob.gemms[i].setUseBias(false);
                tensile_prob.gemms[i].setActivationType(Tensile::ActivationType::None);
                tensile_prob.gemms[i].setActivationHPA(false);
                tensile_prob.gemms[i].setUseScaleD(false);
                if(!(*solution->problemPredicate)(tensile_prob.gemms[i]))
                {
                    isFallbackSupported = false;
                }
                tensile_prob.gemms[i].setUseBias(useBias);
                tensile_prob.gemms[i].setActivationType(actType);
                tensile_prob.gemms[i].setActivationHPA(actHPA);
                tensile_prob.gemms[i].setUseScaleD(useScaleD);
                if(!isFallbackSupported)
                    break;
            }
        }
        if(!isSupported && !isFallbackSupported)
        {
            log_error(__func__, "Solution is not supported");
            return rocblaslt_status_invalid_value;
        }
    }
    return rocblaslt_status_success;
}

template <typename Ti, typename To, typename Tc>
rocblaslt_status isSolutionSupported(RocblasltContractionProblem<Ti, To, Tc>& prob,
                                     rocblaslt_matmul_algo*                   algo,
                                     size_t*                                  workspaceSizeInBytes)
{
    std::shared_ptr<Tensile::ContractionProblemGemm> tensile_prob
        = std::static_pointer_cast<Tensile::ContractionProblemGemm>(algo->data2.ptr);
    updateTensileProblem(false, prob, *tensile_prob);
    tensile_prob->setWorkspaceSize(prob.workspaceSize);
    return isSolutionSupported(*tensile_prob, prob, algo, workspaceSizeInBytes);
}

template <typename Tc>
void setRestrictions(Tensile::ContractionProblemGemm& tensile_prob, const Tc* alpha, const Tc* beta)
{
    tensile_prob.setAlphaRestriction(Tensile::toScalarValueEnum(*alpha));
    tensile_prob.setBetaRestriction(Tensile::toScalarValueEnum(*beta));
}

rocblaslt_status isSolutionSupported(const rocblaslt::RocGemmType& gemmType,
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
        return isSolutionSupported(data->problem, data->inputs, &algo, &workspaceSizeInBytes);
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
        return isSolutionSupported(data->problem, data->inputs, &algo, &workspaceSizeInBytes);
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
        auto                             solutions    = getSolutions(
            data->inputs, library, hardware, data->problem, requestedAlgoCount, fallbackSize);

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
        std::vector<bool>                                          useBias, actHPA, useScaleD;
        std::vector<Tensile::ActivationType>                       actType;
        bool                                                       normal_gemm = 1;
        for(int i = 0; i < data->problem.gemms.size(); i++)
        {
            if(data->inputs.grouped[i].scaleD != nullptr || data->inputs.grouped[i].bias != nullptr
               || data->problem.gemms[i].activationEnumArg() != Tensile::ActivationType::None)
            {
                normal_gemm = 0;
                break;
            }
        }
        if(normal_gemm)
        {
            for(int i = 0; i < data->problem.gemms.size(); i++)
            {
                useBias.push_back(data->problem.gemms[i].useBias());
                actType.push_back(data->problem.gemms[i].activationType());
                actHPA.push_back(data->problem.gemms[i].activationHPA());
                useScaleD.push_back(data->problem.gemms[i].useScaleD());
                data->problem.gemms[i].setUseBias(false);
                data->problem.gemms[i].setActivationType(Tensile::ActivationType::None);
                data->problem.gemms[i].setActivationHPA(false);
                data->problem.gemms[i].setUseScaleD(false);
            }
            solutions_fallback = library->findTopSolutionsGroupedGemm(
                data->problem.gemms, *hardware, requestedAlgoCount);
            for(int i = 0; i < data->problem.gemms.size(); i++)
            {
                data->problem.gemms[i].setUseBias(useBias[i]);
                data->problem.gemms[i].setActivationType(actType[i]);
                data->problem.gemms[i].setActivationHPA(actHPA[i]);
                data->problem.gemms[i].setUseScaleD(useScaleD[i]);
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
                                       0);
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
#define CREATEFUNCTION(Ti, To, Tc)                                                               \
    template rocblaslt_status runContractionProblem<Ti, To, Tc>(                                 \
        rocblaslt_handle             handle,                                                     \
        const rocblaslt_matmul_algo* algo,                                                       \
        const RocblasltContractionProblem<Ti, To, Tc>&);                                         \
    template rocblaslt_status gemmCreate(const RocblasltContractionProblem<Ti, To, Tc>&,         \
                                         std::shared_ptr<void>& gemmData,                        \
                                         size_t&                gemmCount);                                     \
    template rocblaslt_status groupedGemmCreate<Ti, To, Tc>(                                     \
        std::vector<RocblasltContractionProblem<Ti, To, Tc>>&, std::shared_ptr<void>&, size_t&); \
    template RocblasltContractionProblem<Ti, To, Tc> ConstructRocblasltProblem<Ti, To, Tc>(      \
        const rocblaslt_matmul_desc matmul_descr,                                                \
        rocblaslt_matrix_layout     matA,                                                        \
        rocblaslt_matrix_layout     matB,                                                        \
        rocblaslt_matrix_layout     matC,                                                        \
        rocblaslt_matrix_layout     matD,                                                        \
        const Tc*                   alpha,                                                       \
        const Tc*                   beta,                                                        \
        size_t                      maxWorkSpaceBytes);                                                               \
    template rocblaslt_status getAllSolutions(                                                   \
        RocblasltContractionProblem<Ti, To, Tc>&        prob,                                    \
        rocblaslt_handle                                handle,                                  \
        std::vector<rocblaslt_matmul_heuristic_result>& heuristicResults,                        \
        size_t                                          maxWorkSpaceBytes);                                                               \
    template rocblaslt_status getAllSolutions(                                                   \
        std::vector<RocblasltContractionProblem<Ti, To, Tc>>& probs,                             \
        rocblaslt_handle                                      handle,                            \
        std::vector<rocblaslt_matmul_heuristic_result>&       heuristicResults,                  \
        size_t                                                maxWorkSpaceBytes);                                                               \
    template rocblaslt_status isSolutionSupported(RocblasltContractionProblem<Ti, To, Tc>& prob, \
                                                  rocblaslt_matmul_algo*                   algo, \
                                                  size_t* workspaceSizeInBytes);                 \
    template rocblaslt_status getBestSolutions<Ti, To, Tc>(                                      \
        RocblasltContractionProblem<Ti, To, Tc> prob,                                            \
        rocblaslt_handle                        handle,                                          \
        int                                     requestedAlgoCount,                              \
        rocblaslt_matmul_heuristic_result       heuristicResultsArray[],                         \
        int*                                    returnAlgoCount,                                 \
        size_t                                  maxWorkSpaceBytes);

CREATEFUNCTION(float, float, float)
CREATEFUNCTION(rocblaslt_half, rocblaslt_half, float)
CREATEFUNCTION(rocblaslt_bfloat16, rocblaslt_bfloat16, float)

/***********************************************************************************
 * Whether Tensile has been initialized for at least one device (used for
 *testing) *
 ***********************************************************************************/
std::atomic_bool& rocblaslt_internal_tensile_is_initialized()
{
    static std::atomic_bool init;
    return init;
}
