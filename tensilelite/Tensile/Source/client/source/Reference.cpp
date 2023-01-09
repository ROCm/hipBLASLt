/**
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "Reference.hpp"
#include "DataInitialization.hpp"
#include "Tensile/Debug.hpp"
#include "Tensile/Utils.hpp"

#include <cstddef>

namespace Tensile
{
    namespace Client
    {
        template <typename T>
        struct Transform
        {
            inline static T Input(T const& val, bool conj)
            {
                return val;
            }
        };

        template <typename T>
        struct Transform<std::complex<T>>
        {
            inline static std::complex<T> Input(std::complex<T> const& val, bool conj)
            {
                if(conj)
                    return std::conj(val);

                return val;
            }
        };

        // coord is vector with coordinates for dimensions in the anchor tensor
        // tensor is tensor descriptor for a or b
        // sumCoord is the coordinate in the sum dimension associated with the zero-pad
        bool inZeroPad(ContractionProblem const&          problem,
                       ContractionProblem::ZeroPad const& zp,
                       const TensorDescriptor&            tensor,
                       const std::vector<int64_t>&        anchorCoord,
                       int64_t                            sumCoord)
        {
            if(zp.valid())
            {
                // Check to see if the element coordinate is below or above the zero-pad
                // range The comparison is done in the element domain.
                assert(zp.anchorPos != -1); // ensure initialized.
                int64_t anchorRelCoord = anchorCoord[zp.anchorPos] * tensor.strides()[zp.anchorPos]
                                         + sumCoord * tensor.strides()[zp.boundPos];
                // elementEdge calculation:
                // size of anchor dim is in the output space, so add filter size-1 to get
                // input spatial dim, then subtract padEnd anchorStride is typically spatial
                // stride (W,H) * convolution stride boundPos stride is typically spatial
                // stride (W,H) * dilation padStart, padEnd are pre-scaled by spatial stride
                int64_t elementEdge
                    = tensor.sizes().at(zp.anchorPos) * tensor.strides()[zp.anchorPos]
                      + (tensor.sizes().at(zp.boundPos) - 1) * tensor.strides()[zp.boundPos]
                      - zp.padEnd;

                bool rv = anchorRelCoord < zp.padStart || anchorRelCoord >= elementEdge;
                return rv;
            }
            else
            {
                return false;
            }
        }

        void throwException(const std::string& msg)
        {
            throw std::runtime_error(msg.c_str());
        }

        template <typename Accumulator, typename TypeL, typename TypeR>
        inline Accumulator multiply(TypeL l, TypeR r)
        {
            /* Transform the data type from TypeL/TypeR to Accumulator if TypeL!=ACC or TypeR!=ACC, but filter out cases, I8/I32/I32 and I8x4/I32/I32
             *
             * There are three cases of doing multiplication and their conditions to do transform or not are as below.
             * 1. AxB : (A!=ACC or B!=ACC) and A!=I8 and A!=I8x4
             * 2. Alpha x rC :  (Alpha!=ACC or rC!=ACC)
             * 3. Beta x C : (Beta!=ACC or C!=ACC)
            */
            constexpr bool needAccumCast
                = !(std::is_same<TypeL, Accumulator>() && std::is_same<TypeR, Accumulator>())
                  && !std::is_same<TypeL, Int8>() //case I8/I32/I32, I8 be implicitly cast to int.
                  && !std::is_same<TypeL, Int8x4>(); //case I8x4/I32/I32, I8x4 overloading the op*.

            using LMultT = std::conditional_t<needAccumCast, Accumulator, TypeL>;
            using RMultT = std::conditional_t<needAccumCast, Accumulator, TypeR>;
            return static_cast<Accumulator>(static_cast<LMultT>(l) * static_cast<RMultT>(r));
        }

        template <typename T, typename Accumulator>
        typename std::enable_if<std::is_same<int8_t, T>::value, T>::type
            SaturateCast(Accumulator val)
        {
            if(std::is_same<Accumulator, float>::value)
                val = std::nearbyint(val); //round to even

            if(val > static_cast<Accumulator>(127))
                val = static_cast<Accumulator>(127);
            else if(val < static_cast<Accumulator>(-128))
                val = static_cast<Accumulator>(-128);
            return static_cast<T>(val);
        }

        template <typename T, typename Accumulator>
        typename std::enable_if<!std::is_same<int8_t, T>::value, T>::type
            SaturateCast(Accumulator val)
        {
            return static_cast<T>(val);
        }

        template <typename Accumulator>
        typename std::enable_if<std::is_same<Half, Accumulator>::value
                                    || std::is_same<float, Accumulator>::value
                                    || std::is_same<double, Accumulator>::value
                                    || std::is_same<BFloat16, Accumulator>::value
                                    || std::is_same<int32_t, Accumulator>::value
                                    || std::is_same<int8_t, Accumulator>::value,
                                Accumulator>::type
            GetBias(DataType biasType, void const* biasptr, int pos, bool aConjugate)
        {
            switch(biasType)
            {
            case DataType::Float:
            {
                auto bptr = static_cast<float const*>(biasptr);
                return multiply<Accumulator>(Transform<float>::Input(bptr[pos], aConjugate), 1);
            }
            break;
            case DataType::Double:
            {
                auto bptr = static_cast<double const*>(biasptr);
                return multiply<Accumulator>(Transform<double>::Input(bptr[pos], aConjugate), 1);
            }
            break;
            case DataType::Half:
            {
                auto bptr = static_cast<Half const*>(biasptr);
                return multiply<Accumulator>(Transform<Half>::Input(bptr[pos], aConjugate), 1);
            }
            break;
            case DataType::Int32:
            {
                auto bptr = static_cast<int32_t const*>(biasptr);
                return multiply<Accumulator>(Transform<int32_t>::Input(bptr[pos], aConjugate), 1);
            }
            break;
            case DataType::BFloat16:
            {
                auto bptr = static_cast<BFloat16 const*>(biasptr);
                return multiply<Accumulator>(Transform<BFloat16>::Input(bptr[pos], aConjugate), 1);
            }
            break;
            case DataType::Int8:
            {
                auto bptr = static_cast<int8_t const*>(biasptr);
                return multiply<Accumulator>(Transform<int8_t>::Input(bptr[pos], aConjugate), 1);
            }
            break;
            case DataType::ComplexFloat:
            case DataType::ComplexDouble:
            case DataType::Int8x4:
            case DataType::Count:;
            }
            return DataInitialization::getValue<Accumulator, InitMode::Zero>();
        }

        template <typename Accumulator>
        typename std::enable_if<!std::is_same<Half, Accumulator>::value
                                    && !std::is_same<float, Accumulator>::value
                                    && !std::is_same<double, Accumulator>::value
                                    && !std::is_same<BFloat16, Accumulator>::value
                                    && !std::is_same<int32_t, Accumulator>::value
                                    && !std::is_same<int8_t, Accumulator>::value,
                                Accumulator>::type
            GetBias(DataType biasType, void const* biasptr, int pos, bool aConjugate)
        {
            return DataInitialization::getValue<Accumulator, InitMode::Zero>();
        }

        template <typename T>
        typename std::enable_if<std::is_same<float, T>::value || std::is_same<Half, T>::value
                                    || std::is_same<BFloat16, T>::value,
                                T>::type
            Activation(ActivationType activationType,
                       T              val,
                       ActivationType activationType2,
                       std::vector<T> args)
        {
            // Only cast to float in BFloat16
            constexpr bool needCast = std::is_same<BFloat16, T>();
            using castT             = std::conditional_t<needCast, float, T>;
            auto new_type
                = activationType == ActivationType::All ? activationType2 : activationType;
            if(new_type == ActivationType::Abs)
            {
                return static_cast<T>(std::max(static_cast<castT>(val), -static_cast<castT>(val)));
            }
            else if(new_type == ActivationType::Clippedrelu)
            {
                if(val >= args[0])
                    return static_cast<T>(
                        std::min(static_cast<castT>(val), static_cast<castT>(args[1])));
                return static_cast<T>(0.0);
            }
            else if(new_type == ActivationType::Exp)
            {
                return static_cast<T>(exp(static_cast<castT>(val)));
            }
            else if(new_type == ActivationType::Gelu)
            {
                auto castedVal = static_cast<castT>(val);
                auto k0        = static_cast<castT>(0.7978845608028654);
                auto k1        = static_cast<castT>(0.044715);
                // float(0.5 * x * (1 + tanh(k0 * x * (1 + k1 * x * x))));
                auto tmp = (static_cast<castT>(1)
                            + multiply<castT>(k1, multiply<castT>(castedVal, castedVal)));
                tmp      = multiply<castT>(k0, multiply<castT>(castedVal, tmp));
                tmp      = static_cast<castT>(1) + static_cast<castT>(tanh(tmp));
                tmp = multiply<castT>(static_cast<castT>(0.5f), multiply<castT>(castedVal, tmp));
                return static_cast<T>(tmp);
            }
            else if(new_type == ActivationType::Leakyrelu)
            {
                assert((args.size() == getAdditionalArgNum(activationType)));
                auto tmp = static_cast<castT>(val);
                tmp      = tmp > static_cast<castT>(0.f) ? tmp : multiply<castT>(tmp, args[0]);
                return (T)(tmp);
            }
            else if(new_type == ActivationType::Relu)
            {
                return (T)(std::max(0.f, static_cast<float>(val)));
            }
            else if(new_type == ActivationType::Sigmoid)
            {
                return static_cast<T>(1.f
                                      / (1.f + static_cast<castT>(exp(-static_cast<castT>(val)))));
            }
            else if(new_type == ActivationType::Tanh)
            {
                return multiply<T>(
                    tanh(multiply<castT>(static_cast<castT>(val), static_cast<castT>(args[0]))),
                    static_cast<castT>(args[1]));
            }
            return val;
        }

        template <typename T>
        typename std::enable_if<std::is_same<double, T>::value || std::is_same<int32_t, T>::value,
                                T>::type
            Activation(ActivationType activationType,
                       T              val,
                       ActivationType activationType2,
                       std::vector<T> args)
        {
            auto new_type
                = activationType == ActivationType::All ? activationType2 : activationType;
            if(new_type == ActivationType::Abs)
            {
                return static_cast<T>(std::abs(val));
            }
            else if(new_type == ActivationType::Clippedrelu)
            {
                if(val >= args[0])
                    return static_cast<T>(std::min(val, args[1]));
                return static_cast<T>(0);
            }
            else if(new_type == ActivationType::Relu)
            {
                return static_cast<T>(std::max(static_cast<T>(0.0), val));
            }
            else if(new_type == ActivationType::Leakyrelu)
            {
                assert((args.size() == getAdditionalArgNum(activationType)));
                val = val > 0 ? val : val * args[0];
                return val;
            }
            return val;
        }

        template <typename T>
        typename std::enable_if<!std::is_same<Half, T>::value && !std::is_same<float, T>::value
                                    && !std::is_same<double, T>::value
                                    && !std::is_same<BFloat16, T>::value
                                    && !std::is_same<int32_t, T>::value,
                                T>::type
            Activation(ActivationType activationType,
                       T              val,
                       ActivationType activationType2,
                       std::vector<T> args)
        {
            return val;
        }

        template <typename Inputs, typename Accumulator>
        void ReferenceSolution<Inputs, Accumulator>::SolveCPU(ContractionProblem const& problem,
                                                              Inputs const&             inputs,
                                                              size_t validationStride)
        {
            auto const& freeIndicesA = problem.freeIndicesA();
            auto const& freeIndicesB = problem.freeIndicesB();
            auto const& batchIndices = problem.batchIndices();
            auto const& boundIndices = problem.boundIndices();

            auto const& a = problem.a();
            auto const& b = problem.b();
            auto const& c = problem.c();
            auto const& d = problem.d();

            bool aConjugate = false;
            bool bConjugate = false;

            for(auto const& op : problem.aOps())
                if(op.type == TensorOp::Type::ComplexConjugate)
                    aConjugate = true;

            for(auto const& op : problem.bOps())
                if(op.type == TensorOp::Type::ComplexConjugate)
                    bConjugate = true;

            std::vector<size_t> freeASize(freeIndicesA.size());
            std::vector<size_t> freeBSize(freeIndicesB.size());
            std::vector<size_t> batchSize(batchIndices.size());
            std::vector<size_t> boundSize(boundIndices.size());

            for(int i = 0; i < freeASize.size(); i++)
                freeASize[i] = problem.freeSizeA(i);
            for(int i = 0; i < freeBSize.size(); i++)
                freeBSize[i] = problem.freeSizeB(i);
            for(int i = 0; i < batchSize.size(); i++)
                batchSize[i] = problem.batchSize(i);
            for(int i = 0; i < boundSize.size(); i++)
                boundSize[i] = problem.boundSize(i);

            auto boundCount = CoordCount(boundSize.begin() + 1, boundSize.end());

            if(inputs.alpha != static_cast<typename Inputs::AlphaType>(0))
            {
                if(inputs.a == nullptr || inputs.b == nullptr)
                {
                    std::ostringstream msg;
                    msg << "Unsupported nullptr for";
                    if(!inputs.a)
                        msg << " A";
                    if(!inputs.b)
                        msg << " B";
                    msg << " when Alpha !=0";

                    throw std::runtime_error(msg.str());
                }
            }

#pragma omp parallel for
            for(size_t dNum = 0; dNum < d.totalLogicalElements(); dNum += validationStride)
            {
                std::vector<int64_t> aCoord(a.dimensions());
                std::vector<int64_t> bCoord(b.dimensions());
                std::vector<int64_t> cCoord(c.dimensions());
                std::vector<int64_t> dCoord(d.dimensions());

                CoordNumbered(
                    dNum, dCoord.begin(), dCoord.end(), d.sizes().begin(), d.sizes().end());

                for(size_t i = 0; i < problem.batchIndices().size(); i++)
                {
                    auto const& idx   = problem.batchIndices()[i];
                    size_t      coord = dCoord[idx.d];

                    aCoord[idx.a] = coord;
                    bCoord[idx.b] = coord;
                    cCoord[idx.c] = coord;
                }

                for(size_t i = 0; i < problem.freeIndices().size(); i++)
                {
                    auto const& idx   = problem.freeIndices()[i];
                    size_t      coord = dCoord[idx.d];

                    cCoord[idx.c] = coord;

                    if(idx.isA)
                        aCoord[idx.i] = coord;
                    else
                        bCoord[idx.i] = coord;
                }

                Accumulator value(0);

                // Check short-circuit for alpha = 0
                if(inputs.alpha != static_cast<typename Inputs::AlphaType>(0))
                {
                    for(size_t boundNum = 0; boundNum < boundCount; boundNum++)
                    {
                        std::vector<int64_t> bound(problem.boundIndices().size());
                        CoordNumbered(boundNum,
                                      bound.begin() + 1,
                                      bound.end(),
                                      boundSize.begin() + 1,
                                      boundSize.end());
                        bool aInZeroPad = false;
                        bool bInZeroPad = false;

                        for(int i = 1; i < bound.size(); i++)
                        {
                            auto const& zpA           = problem.boundIndices()[i].aZeroPad;
                            auto const& zpB           = problem.boundIndices()[i].bZeroPad;
                            aCoord[boundIndices[i].a] = bound[i];
                            bCoord[boundIndices[i].b] = bound[i];

                            if(problem.boundIndices()[i].aMirror)
                                aCoord[boundIndices[i].a]
                                    = boundSize[i] - aCoord[boundIndices[i].a] - 1;
                            if(problem.boundIndices()[i].bMirror)
                                bCoord[boundIndices[i].b]
                                    = boundSize[i] - bCoord[boundIndices[i].b] - 1;

                            if(zpA.valid())
                            {
                                auto sumCoord = bound.at(problem.toBoundsPos(zpA.boundIndex));
                                if(problem.boundIndices()[i].aMirror)
                                    sumCoord = boundSize[i] - sumCoord - 1;

                                if(inZeroPad(problem, zpA, a, aCoord, sumCoord))
                                    aInZeroPad = true;
                            }
                            if(zpB.valid())
                            {
                                auto sumCoord = bound.at(problem.toBoundsPos(zpB.boundIndex));
                                if(problem.boundIndices()[i].bMirror)
                                    sumCoord = boundSize[i] - sumCoord - 1;
                                if(inZeroPad(problem, zpB, b, bCoord, sumCoord))
                                    bInZeroPad = true;
                            }
                        }

                        size_t aIndex = a.index(aCoord);
                        size_t bIndex = b.index(bCoord);
                        for(int i = 1; i < bound.size(); i++)
                        {
                            auto const& zpA = problem.boundIndices()[i].aZeroPad;
                            auto const& zpB = problem.boundIndices()[i].bZeroPad;

                            aIndex -= zpA.padStart;
                            bIndex -= zpB.padStart;
                        }

                        auto aStride = problem.a().strides()[boundIndices[0].a];
                        auto bStride = problem.b().strides()[boundIndices[0].b];

                        // innermost bound calculation:
                        for(size_t i = 0; i < boundSize[0]; i++)
                        {
                            auto const& zpA = problem.boundIndices()[0].aZeroPad;
                            auto const& zpB = problem.boundIndices()[0].bZeroPad;
                            size_t      aI
                                = problem.boundIndices()[0].aMirror ? (boundSize[0] - i - 1) : i;
                            size_t bI
                                = problem.boundIndices()[0].bMirror ? (boundSize[0] - i - 1) : i;

                            typename Inputs::AType aVal(0);
                            typename Inputs::BType bVal(0);
                            if(!aInZeroPad && !inZeroPad(problem, zpA, a, aCoord, aI))
                                aVal = Transform<typename Inputs::AType>::Input(
                                    inputs.a[aIndex + (aI * aStride) - zpA.padStart], aConjugate);
                            if(!bInZeroPad && !inZeroPad(problem, zpB, b, bCoord, bI))
                                bVal = Transform<typename Inputs::BType>::Input(
                                    inputs.b[bIndex + (bI * bStride) - zpB.padStart], bConjugate);

                            value += multiply<Accumulator>(aVal, bVal);

                            if(0)
                            {
                                std::cout << " bound=" << bound[0] << "," << bound[1]
                                          << " dNum=" << dNum << " value=" << value
                                          << " aInZeroPad=" << aInZeroPad << " aindex=" << aIndex
                                          << " +offset="
                                          << (int64_t)(i * aStride) - zpA.padStart
                                          //<< " aVal=" << aVal // disable int8
                                          << "\n";
                            }
                        }
                    }
                }

                auto cIndex = c.index(cCoord);
                auto dIndex = d.index(dCoord);

                // Ensure zero*nan returns zero
                auto beta = inputs.beta;
                auto zero = static_cast<typename Inputs::BetaType>(0);

                auto resultD = multiply<Accumulator>(inputs.alpha, value)
                               + ((beta == zero) ? static_cast<Accumulator>(zero)
                                                 : multiply<Accumulator>(beta, inputs.c[cIndex]));
                // bias
                if(problem.useBias() && inputs.bias)
                {
                    int         pos = int(dNum % problem.d().sizes()[0]);
                    Accumulator bias
                        = GetBias<Accumulator>(problem.biasType(), inputs.bias, pos, aConjugate);
                    resultD += bias;
                }
                // Activation adds here
                std::vector<Accumulator> actArgs;
                for(int i = 0; i < inputs.activationArgs.size(); i++)
                    actArgs.push_back(static_cast<Accumulator>(inputs.activationArgs[i]));
                resultD = Activation(
                    problem.activationType(), resultD, problem.activationEnumArg(), actArgs);
                if(problem.useScaleD())
                {
                    int  pos = int(dNum % problem.d().sizes()[0]);
                    auto scaleD
                        = multiply<Accumulator>(Transform<typename Inputs::AlphaType>::Input(
                                                    inputs.scaleD[pos], aConjugate),
                                                1);
                    resultD *= scaleD;
                }
                inputs.d[dIndex] = SaturateCast<typename Inputs::DType>(resultD);
            }
        }

        void SolveCPU(ContractionProblem const& problem,
                      ContractionInputs const&  inputs,
                      size_t                    validationStride)
        {
            // retreive alpha/beta type set via setAlpha/BetaType()
            auto alphaType = problem.alphaType();
            auto betaType  = problem.betaType();
            auto biasType  = problem.biasType();

            // Backward-compatible: when setAlpha/BetaType() wasn't called, use the old way
            // Could remove after rocBLAS is updated
            if(alphaType == DataType::None)
            {
                alphaType = problem.a().dataType() == DataType::BFloat16 ? DataType::Float
                                                                         : problem.d().dataType();
            }
            if(betaType == DataType::None)
            {
                betaType = alphaType;
            }
            if(biasType == DataType::None)
            {
                biasType = problem.d().dataType();
            }

            auto contractionInputsTypeId = ContractionInputs::TypeId(problem.a().dataType(),
                                                                     problem.b().dataType(),
                                                                     problem.c().dataType(),
                                                                     problem.d().dataType(),
                                                                     alphaType,
                                                                     betaType);

            switch(contractionInputsTypeId)
            {
            case ContractionInputs_S_S_S::TypeId():
            {
                auto const& typedInputs = dynamic_cast<ContractionInputs_S_S_S const&>(inputs);
                return ReferenceSolution<ContractionInputs_S_S_S>::SolveCPU(
                    problem, typedInputs, validationStride);
            }
            case ContractionInputs_D_D_D::TypeId():
            {
                auto const& typedInputs = dynamic_cast<ContractionInputs_D_D_D const&>(inputs);
                return ReferenceSolution<ContractionInputs_D_D_D>::SolveCPU(
                    problem, typedInputs, validationStride);
            }
            case ContractionInputs_C_C_C::TypeId():
            {
                auto const& typedInputs = dynamic_cast<ContractionInputs_C_C_C const&>(inputs);
                return ReferenceSolution<ContractionInputs_C_C_C>::SolveCPU(
                    problem, typedInputs, validationStride);
            }
            case ContractionInputs_Z_Z_Z::TypeId():
            {
                auto const& typedInputs = dynamic_cast<ContractionInputs_Z_Z_Z const&>(inputs);
                return ReferenceSolution<ContractionInputs_Z_Z_Z>::SolveCPU(
                    problem, typedInputs, validationStride);
            }
#ifdef TENSILE_USE_HALF
            case ContractionInputs_H_H_H::TypeId():
            {
                auto const& typedInputs = dynamic_cast<ContractionInputs_H_H_H const&>(inputs);

                if(problem.highPrecisionAccumulate())
                {
                    return ReferenceSolution<ContractionInputs_H_H_H, float>::SolveCPU(
                        problem, typedInputs, validationStride);
                }
                else
                {
                    return ReferenceSolution<ContractionInputs_H_H_H>::SolveCPU(
                        problem, typedInputs, validationStride);
                }
            }
            case ContractionInputs_H_S_S::TypeId():
            {
                auto const& typedInputs = dynamic_cast<ContractionInputs_H_S_S const&>(inputs);
                return ReferenceSolution<ContractionInputs_H_S_S>::SolveCPU(
                    problem, typedInputs, validationStride);
            }
            case ContractionInputs_H_H_S::TypeId():
            {
                auto const& typedInputs = dynamic_cast<ContractionInputs_H_H_S const&>(inputs);
                return ReferenceSolution<ContractionInputs_H_H_S, float>::SolveCPU(
                    problem, typedInputs, validationStride);
            }
#endif // TENSILE_USE_HALF
            case ContractionInputs_I8x4_I32_I32::TypeId():
            {
                auto const& typedInputs
                    = dynamic_cast<ContractionInputs_I8x4_I32_I32 const&>(inputs);
                return ReferenceSolution<ContractionInputs_I8x4_I32_I32>::SolveCPU(
                    problem, typedInputs, validationStride);
            }
            case ContractionInputs_I32_I32_I32::TypeId():
            {
                auto const& typedInputs
                    = dynamic_cast<ContractionInputs_I32_I32_I32 const&>(inputs);
                return ReferenceSolution<ContractionInputs_I32_I32_I32>::SolveCPU(
                    problem, typedInputs, validationStride);
            }
            case ContractionInputs_I8_I8_I32::TypeId():
            {
                auto const& typedInputs = dynamic_cast<ContractionInputs_I8_I8_I32 const&>(inputs);
                return ReferenceSolution<ContractionInputs_I8_I8_I32, int32_t>::SolveCPU(
                    problem, typedInputs, validationStride);
            }
            case ContractionInputs_I8_I32_I32::TypeId():
            {
                auto const& typedInputs = dynamic_cast<ContractionInputs_I8_I32_I32 const&>(inputs);
                return ReferenceSolution<ContractionInputs_I8_I32_I32>::SolveCPU(
                    problem, typedInputs, validationStride);
            }
            case ContractionInputs_I8_I32_S::TypeId():
            {
                auto const& typedInputs = dynamic_cast<ContractionInputs_I8_I32_S const&>(inputs);
                return ReferenceSolution<ContractionInputs_I8_I32_S, float>::SolveCPU(
                    problem, typedInputs, validationStride);
            }
            case ContractionInputs_I8_I8_S::TypeId():
            {
                auto const& typedInputs = dynamic_cast<ContractionInputs_I8_I8_S const&>(inputs);
                return ReferenceSolution<ContractionInputs_I8_I8_S, float>::SolveCPU(
                    problem, typedInputs, validationStride);
            }
#ifdef TENSILE_USE_BF16
            case ContractionInputs_B_B_S::TypeId():
            {
                auto const& typedInputs = dynamic_cast<ContractionInputs_B_B_S const&>(inputs);

                if(problem.highPrecisionAccumulate())
                {
                    return ReferenceSolution<ContractionInputs_B_B_S, float>::SolveCPU(
                        problem, typedInputs, validationStride);
                }
                else
                {
                    return ReferenceSolution<ContractionInputs_B_B_S>::SolveCPU(
                        problem, typedInputs, validationStride);
                }
            }
            case ContractionInputs_B_S_S::TypeId():
            {
                auto const& typedInputs = dynamic_cast<ContractionInputs_B_S_S const&>(inputs);
                return ReferenceSolution<ContractionInputs_B_S_S>::SolveCPU(
                    problem, typedInputs, validationStride);
            }
#endif // TENSILE_USE_BF16

            default:;
            }

            throw std::runtime_error("Data type not implemented.");
        }

        // A is activation, B is weights
        // Assume packed.
        template <typename Inputs, typename Accumulator>
        void ReferenceSolution<Inputs, Accumulator>::SolveCPUConvolution(
            ConvolutionProblem const& convProblem,
            ContractionProblem const& problem,
            Inputs const&             inputs)
        {
            const bool db1 = Debug::Instance().printConvolutionReference1();
            const bool db2 = Debug::Instance().printConvolutionReference2();

            if(static_cast<typename Inputs::DType>(inputs.beta)
               != static_cast<typename Inputs::DType>(0.0))
                throw std::runtime_error("convolution requires beta==0");

            ConvolutionProblem::LoopCounts counts;
            counts.setupForData(convProblem, problem);

            convProblem.validate(problem, counts);

            TensorDescriptor activationTensor = convProblem.setupDataActivation(counts, problem);
            TensorDescriptor weightTensor     = convProblem.setupForwardWeights(counts, problem);
            TensorDescriptor outputTensor     = convProblem.setupDataOutput(counts, problem);

            auto formatA = counts.formatA();
            auto formatB = counts.formatB();
            auto formatD = counts.formatD();

            size_t padShift
                = std::accumulate(problem.aZeroPad().begin(),
                                  problem.aZeroPad().end(),
                                  0,
                                  [](size_t sum, const ContractionProblem::ZeroPad& zp) {
                                      return sum + zp.padStart;
                                  });
            if(db1)
            {
                std::cout << "SolveCPUConvolution:\n";
                std::cout << "  activationTensor=" << activationTensor << "\n";
                std::cout << "counts:" << std::endl << counts.description() << "\n";
            }

            // Loops always traverse in same order but addressing in memory can be flexible to support different activation
            // and filter formats
            size_t spatialCoordCount = CoordCount(counts.scount.begin(), counts.scount.end());
#pragma omp parallel for collapse(3)
            for(size_t cout = 0; cout < counts.coutCount; cout++)
                for(size_t spatialIndex = 0; spatialIndex < spatialCoordCount; spatialIndex++)
                    for(size_t n = 0; n < counts.batchCount; n++)
                    {
                        std::vector<size_t> spatialCoord(ConvolutionProblem::MaxNumSpatialDims, 0);

                        CoordNumbered(spatialIndex,
                                      spatialCoord.begin(),
                                      spatialCoord.end(),
                                      counts.scount.begin(),
                                      counts.scount.end());

                        Accumulator value(0);
                        size_t      filterCoordCount
                            = CoordCount(counts.filterCount.begin(), counts.filterCount.end());
                        for(size_t cin = 0; cin < counts.cinCount; cin++)
                            for(size_t filterIndex = 0; filterIndex < filterCoordCount;
                                filterIndex++)
                            {

                                std::vector<size_t> filterCoord(counts.filterCount.size(), 0);
                                CoordNumbered(filterIndex,
                                              filterCoord.begin(),
                                              filterCoord.end(),
                                              counts.filterCount.begin(),
                                              counts.filterCount.end());

                                // Save coordinates from the looop and compute memeory index
                                // Each component stores in appropriate memory order
                                std::vector<int64_t> aCoord(activationTensor.dimensions(), 0);
                                std::vector<int64_t> bCoord(weightTensor.dimensions(), 0);

                                aCoord[formatA.batchPosition()]   = n;
                                aCoord[formatA.channelPosition()] = cin;
                                for(auto i = 0; i < formatA.spatialPositions().size(); i++)
                                    aCoord[formatA.spatialPositions()[i]] = spatialCoord[i];

                                // add filters to address calc, if they have non-unit strides:
                                for(int fi = 0; fi < counts.filterCount.size(); fi++)
                                {
                                    auto fp = formatA.filterPositions()[fi];
                                    if(fp != ConvolutionProblem::InvalidPos)
                                        aCoord[fp] = filterCoord[fi];
                                }

                                bCoord[formatB.weights().coutPosition()] = cout;
                                bCoord[formatB.weights().cinPosition()]  = cin;
                                for(int fi = 0; fi < counts.filterCount.size(); fi++)
                                {
                                    auto fp = formatB.weights().filterPositions()[fi];
                                    if(fp != ConvolutionProblem::InvalidPos)
                                        bCoord[fp] = filterCoord[fi];
                                }

                                auto aIndex     = activationTensor.index(aCoord) - padShift;
                                bool inZeroPads = std::accumulate(
                                    problem.aZeroPad().begin(),
                                    problem.aZeroPad().end(),
                                    false,
                                    [&](bool ret, const ContractionProblem::ZeroPad& zp) {
                                        return ret
                                               || inZeroPad(problem,
                                                            zp,
                                                            activationTensor,
                                                            aCoord,
                                                            aCoord[zp.boundPos]);
                                    });

                                auto aVal = inZeroPads ? static_cast<typename Inputs::AType>(0.0)
                                                       : Transform<typename Inputs::AType>::Input(
                                                           inputs.a[aIndex], false);

                                auto bIndex = weightTensor.index(bCoord);
                                auto bVal   = Transform<typename Inputs::BType>::Input(
                                    inputs.b[bIndex], false);

                                if(db2)
                                {
                                    std::cout << "  n,cin,spatialCoord,cout=" << n << "," << cin
                                              << ","
                                              << "," << cout << ","
                                              << " spatialCoord[2,1,0]=" << spatialCoord[2] << ","
                                              << spatialCoord[1] << "," << spatialCoord[0]
                                              << " filterCoord[2,1,0]=" << filterCoord[2] << ","
                                              << filterCoord[1] << "," << filterCoord[0]
                                              << " aIndex=" << aIndex << " bIndex=" << bIndex
                                              << " aVal=" << aVal << " bVal=" << bVal << "\n";
                                }
                                value += multiply<Accumulator>(aVal, bVal);
                            }
                        std::vector<size_t> dCoord(outputTensor.dimensions(), 0);
                        dCoord[formatD.activation().batchPosition()]   = n;
                        dCoord[formatD.activation().channelPosition()] = cout;
                        for(auto i = 0; i < formatD.activation().spatialPositions().size(); i++)
                            dCoord[formatD.activation().spatialPositions()[i]] = spatialCoord[i];

                        auto dIndex = outputTensor.index(dCoord);
                        if(db1)
                        {
                            std::cout << "output: [n,spatialCoord,cout=" << n << ","
                                      << "," << cout << "]"
                                      << " spatialCoord[2,1,0]=" << spatialCoord[2] << ","
                                      << spatialCoord[1] << "," << spatialCoord[0]
                                      << " dIndex=" << dIndex << " value=" << value << "\n";
                        }
                        inputs.d[dIndex] = static_cast<typename Inputs::DType>(
                            multiply<Accumulator>(inputs.alpha, value));
                    }
        }

        void SolveCPUConvolution(ConvolutionProblem const& convProblem,
                                 ContractionProblem const& problem,
                                 ContractionInputs&        inputs)
        {
            // std::cout << "SolveCPUConvolution:" << convProblem << " (vs " << problem <<
            // ")\n";

            if(problem.a().dataType() == DataType::Float
               && problem.b().dataType() == DataType::Float
               && problem.c().dataType() == DataType::Float
               && problem.d().dataType() == DataType::Float)
            {
                auto& typedInputs = dynamic_cast<TypedContractionInputs<float>&>(inputs);
                return ReferenceSolution<TypedContractionInputs<float>>::SolveCPUConvolution(
                    convProblem, problem, typedInputs);
            }
            else
            {
                throw std::runtime_error("Data type not implemented for conv-vs-contract.");
            }
        }
    } // namespace Client
} // namespace Tensile
