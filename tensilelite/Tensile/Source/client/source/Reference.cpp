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

#include "Reference.hpp"
#include "DataInitialization.hpp"
#include "Tensile/Debug.hpp"
#include "Tensile/Utils.hpp"
#include "TypedId.hpp"

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

        template <typename Accumulator, typename Type>
        inline Accumulator cast(Type val)
        {
            /* Transform the data type from TypeL/TypeR to Accumulator if TypeL!=ACC or TypeR!=ACC, but filter out cases, I8/I32/I32 and I8x4/I32/I32
             *
             * There are three cases of doing multiplication and their conditions to do transform or not are as below.
             * 1. AxB : (A!=ACC or B!=ACC) and A!=I8 and A!=I8x4
             * 2. Alpha x rC :  (Alpha!=ACC or rC!=ACC)
             * 3. Beta x C : (Beta!=ACC or C!=ACC)
            */
            constexpr bool needAccumCast
                = !std::is_same<Type, Accumulator>()
                  && !std::is_same<Type, Int8>() //case I8/I32/I32, I8 be implicitly cast to int.
                  && !std::is_same<Type, Int8x4>(); //case I8x4/I32/I32, I8x4 overloading the op*.

            using MultT = std::conditional_t<needAccumCast, Accumulator, Type>;
            return static_cast<MultT>(val);
        }

        template <typename T, typename Accumulator>
        typename std::enable_if<std::is_same<int8_t, T>::value, T>::type
            SaturateCast(Accumulator val)
        {
            if constexpr(std::is_same<Accumulator, Half>::value
                         || std::is_same<Accumulator, BFloat16>::value)
            {
                float tmp = std::nearbyint((float)val); //round to even
                if(tmp > static_cast<float>(127))
                    tmp = static_cast<float>(127);
                else if(tmp < static_cast<float>(-128))
                    tmp = static_cast<float>(-128);
                return static_cast<T>(tmp);
            }
            else
            {
                if constexpr(std::is_same<Accumulator, float>::value
                             || std::is_same<Accumulator, double>::value)
                    val = std::nearbyint(val); //round to even
                if(val > static_cast<Accumulator>(127))
                    val = static_cast<Accumulator>(127);
                else if(val < static_cast<Accumulator>(-128))
                    val = static_cast<Accumulator>(-128);
                return static_cast<T>(val);
            }
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
            GetValue(DataType dataType, void const* voidPtr, int pos, bool aConjugate)
        {
            switch(dataType)
            {
            case DataType::Float:
            {
                auto typedPtr = static_cast<float const*>(voidPtr);
                return cast<Accumulator>(Transform<float>::Input(typedPtr[pos], aConjugate));
            }
            break;
            case DataType::Double:
            {
                auto typedPtr = static_cast<double const*>(voidPtr);
                return cast<Accumulator>(Transform<double>::Input(typedPtr[pos], aConjugate));
            }
            break;
            case DataType::Half:
            {
                auto typedPtr = static_cast<Half const*>(voidPtr);
                return cast<Accumulator>(Transform<Half>::Input(typedPtr[pos], aConjugate));
            }
            break;
            case DataType::Int32:
            {
                auto typedPtr = static_cast<int32_t const*>(voidPtr);
                return cast<Accumulator>(Transform<int32_t>::Input(typedPtr[pos], aConjugate));
            }
            break;
            case DataType::BFloat16:
            {
                auto typedPtr = static_cast<BFloat16 const*>(voidPtr);
                return cast<Accumulator>(Transform<BFloat16>::Input(typedPtr[pos], aConjugate));
            }
            break;
            case DataType::Int8:
            {
                auto typedPtr = static_cast<int8_t const*>(voidPtr);
                return cast<Accumulator>(Transform<int8_t>::Input(typedPtr[pos], aConjugate));
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
            GetValue(DataType biasType, void const* biasptr, int pos, bool aConjugate)
        {
            return DataInitialization::getValue<Accumulator, InitMode::Zero>();
        }

        template <typename Accumulator,
                  std::enable_if_t<std::is_same<Half, Accumulator>::value
                                       || std::is_same<float, Accumulator>::value
                                       || std::is_same<double, Accumulator>::value
                                       || std::is_same<BFloat16, Accumulator>::value
                                       || std::is_same<int32_t, Accumulator>::value
                                       || std::is_same<int8_t, Accumulator>::value,
                                   bool> = true>
        void SetValue(DataType dataType, Accumulator& src, void* dstPtr, int pos)
        {
            switch(dataType)
            {
            case DataType::Float:
            {
                auto typedPtr = static_cast<float*>(dstPtr);
                typedPtr[pos] = SaturateCast<float>(src);
            }
            break;
            case DataType::Double:
            {
                auto typedPtr = static_cast<double*>(dstPtr);
                typedPtr[pos] = SaturateCast<double>(src);
            }
            break;
            case DataType::Half:
            {
                auto typedPtr = static_cast<Half*>(dstPtr);
                typedPtr[pos] = SaturateCast<Half>(src);
            }
            break;
            case DataType::Int32:
            {
                auto typedPtr = static_cast<int32_t*>(dstPtr);
                typedPtr[pos] = SaturateCast<int32_t>(src);
            }
            break;
            case DataType::BFloat16:
            {
                auto typedPtr = static_cast<BFloat16*>(dstPtr);
                typedPtr[pos] = SaturateCast<BFloat16>(src);
            }
            break;
            case DataType::Int8:
            {
                auto typedPtr = static_cast<int8_t*>(dstPtr);
                typedPtr[pos] = SaturateCast<int8_t>(src);
            }
            break;
            case DataType::ComplexFloat:
            case DataType::ComplexDouble:
            case DataType::Int8x4:
            case DataType::Count:;
            }
        }

        template <typename Accumulator,
                  std::enable_if_t<!std::is_same<Half, Accumulator>::value
                                       && !std::is_same<float, Accumulator>::value
                                       && !std::is_same<double, Accumulator>::value
                                       && !std::is_same<BFloat16, Accumulator>::value
                                       && !std::is_same<int32_t, Accumulator>::value
                                       && !std::is_same<int8_t, Accumulator>::value,
                                   bool> = true>
        void SetValue(DataType dataType, Accumulator& src, void* dstPtr, int pos)
        {
            switch(dataType)
            {
            case DataType::Float:
            case DataType::Double:
            case DataType::Half:
            case DataType::Int32:
            case DataType::BFloat16:
            case DataType::Int8:
                break;
            case DataType::ComplexFloat:
            case DataType::ComplexDouble:
                break;
            case DataType::Int8x4:
            {
                throw std::runtime_error("Not supported yet.");
            }
            break;
            case DataType::Count:;
            }
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
            else if(new_type == ActivationType::DGelu)
            {
                auto castedVal = static_cast<float>(val);
                auto k0        = 0.0535161f;
                auto k1        = 0.398942f;
                auto k2        = 0.0356774f;
                auto k3        = 0.797885f;
                // Original: (0.0535161x3 + 0.398942x) x cosh-2(0.0356774x3 + 0.797885x)
                // x1 = (0.0535161 * pow(x, 3) + 0.398942 * x)
                // xx = 0.0356774 * pow(x, 3)+ 0.797885 * x
                // x2 = 4/pow(math.exp(-xx) + math.exp(xx),2)
                // 0.5 * math.tanh(xx) + x1 * x2 + 0.5
                float pow3 = castedVal * castedVal * castedVal;
                float x1   = k0 * pow3 + k1 * castedVal;
                float xx   = k2 * pow3 + k3 * castedVal;
                float x2   = 4 / pow(exp(-xx) + exp(xx), 2);
                float tmp  = 0.5 * tanh(xx) + x1 * x2 + 0.5;
                return static_cast<T>(tmp);
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
            else if(new_type == ActivationType::DGelu)
            {
                throw std::runtime_error("Unsupported type dgelu.");
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
        void ReferenceSolution<Inputs, Accumulator>::SolveCPU(ContractionProblemGemm const& problem,
                                                              ContractionInputs const&      inputs,
                                                              size_t elementsToValidate)
        {
            size_t validationStride = 1;
            if(elementsToValidate > 0 && elementsToValidate < problem.d().totalLogicalElements())
                validationStride
                    = NextPrime(problem.d().totalAllocatedElements() / elementsToValidate);

            // Convert void* to pointers
            typename Inputs::AType const* aPtr = (typename Inputs::AType const*)inputs.a;
            typename Inputs::BType const* bPtr = (typename Inputs::BType const*)inputs.b;
            typename Inputs::CType const* cPtr = (typename Inputs::CType const*)inputs.c;
            typename Inputs::DType*       dPtr = (typename Inputs::DType*)inputs.d;

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

            if(DataTypeInfo::Get(problem.a().dataType()).isComplex)
            {
                aConjugate = true;
            }

            if(DataTypeInfo::Get(problem.b().dataType()).isComplex)
            {
                bConjugate = true;
            }

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

            if(std::get<typename Inputs::AlphaType>(inputs.alpha)
               != static_cast<typename Inputs::AlphaType>(0))
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
                if(std::get<typename Inputs::AlphaType>(inputs.alpha)
                   != static_cast<typename Inputs::AlphaType>(0))
                {
                    for(size_t boundNum = 0; boundNum < boundCount; boundNum++)
                    {
                        std::vector<int64_t> bound(problem.boundIndices().size());
                        CoordNumbered(boundNum,
                                      bound.begin() + 1,
                                      bound.end(),
                                      boundSize.begin() + 1,
                                      boundSize.end());

                        for(int i = 1; i < bound.size(); i++)
                        {
                            aCoord[boundIndices[i].a] = bound[i];
                            bCoord[boundIndices[i].b] = bound[i];

                            if(problem.boundIndices()[i].aMirror)
                                aCoord[boundIndices[i].a]
                                    = boundSize[i] - aCoord[boundIndices[i].a] - 1;
                            if(problem.boundIndices()[i].bMirror)
                                bCoord[boundIndices[i].b]
                                    = boundSize[i] - bCoord[boundIndices[i].b] - 1;
                        }

                        size_t aIndex = a.index(aCoord);
                        size_t bIndex = b.index(bCoord);

                        auto aStride = problem.a().strides()[boundIndices[0].a];
                        auto bStride = problem.b().strides()[boundIndices[0].b];

                        // innermost bound calculation:
                        for(size_t i = 0; i < boundSize[0]; i++)
                        {
                            size_t aI
                                = problem.boundIndices()[0].aMirror ? (boundSize[0] - i - 1) : i;
                            size_t bI
                                = problem.boundIndices()[0].bMirror ? (boundSize[0] - i - 1) : i;

                            typename Inputs::AType aVal(0);
                            typename Inputs::BType bVal(0);
                            aVal = Transform<typename Inputs::AType>::Input(
                                aPtr[aIndex + (aI * aStride)], aConjugate);
                            bVal = Transform<typename Inputs::BType>::Input(
                                bPtr[bIndex + (bI * bStride)], bConjugate);

                            value += multiply<Accumulator>(aVal, bVal);
                        }
                    }
                }

                auto cIndex = c.index(cCoord);
                auto dIndex = d.index(dCoord);

                // Ensure zero*nan returns zero
                Accumulator alpha = constVariantCast<Accumulator>(inputs.alpha);
                Accumulator beta  = constVariantCast<Accumulator>(inputs.beta);
                auto        zero  = static_cast<Accumulator>(0);

                auto resultD = multiply<Accumulator>(alpha, value)
                               + ((beta == zero) ? static_cast<Accumulator>(zero)
                                                 : multiply<Accumulator>(beta, cPtr[cIndex]));

                // bias
                if(problem.useBias() && inputs.bias)
                {
                    int         pos = int(dNum % problem.d().sizes()[0]);
                    Accumulator bias
                        = GetValue<Accumulator>(problem.biasType(), inputs.bias, pos, aConjugate);
                    resultD += bias;
                }
                // E
                if(problem.useE() && !problem.useGradient())
                {
                    typename Inputs::BetaType* ePtr = (typename Inputs::BetaType*)inputs.e;
                    auto                       eIndex
                        = problem.tensors()[ContractionProblemGemm::TENSOR::E].index(dCoord);
                    ePtr[eIndex] = SaturateCast<typename Inputs::BetaType>(resultD);
                }
                // Activation adds here
                std::vector<Accumulator> actArgs;
                for(int i = 0; i < inputs.activationArgs.size(); i++)
                    actArgs.push_back(constVariantCast<Accumulator>(inputs.activationArgs[i]));
                if(problem.useGradient() && problem.activationType() != ActivationType::None
                   && problem.activationEnumArg() != ActivationType::None)
                {
                    Accumulator dataE = static_cast<Accumulator>(0);
                    if(problem.useE())
                    {
                        typename Inputs::BetaType* ePtr = (typename Inputs::BetaType*)inputs.e;
                        auto                       eIndex
                            = problem.tensors()[ContractionProblemGemm::TENSOR::E].index(dCoord);
                        dataE = GetValue<Accumulator>(
                            problem.betaType(), inputs.e, eIndex, aConjugate);
                    }
                    dataE = Activation(
                        problem.activationType(), dataE, problem.activationEnumArg(), actArgs);
                    resultD *= dataE;
                }
                else
                {
                    resultD = Activation(
                        problem.activationType(), resultD, problem.activationEnumArg(), actArgs);
                }

                if(problem.useScaleD())
                {
                    int         pos    = int(dNum % problem.d().sizes()[0]);
                    Accumulator scaleD = GetValue<Accumulator>(
                        problem.alphaType(), inputs.scaleD, pos, aConjugate);
                    resultD *= scaleD;
                }
                dPtr[dIndex] = SaturateCast<typename Inputs::DType>(resultD);
            }
        }

        template <typename Inputs, typename Accumulator>
        void ReferenceSolution<Inputs, Accumulator>::SolveCPU(
            ContractionProblemGroupedGemm const& problem,
            ContractionGroupedInputs const&      inputs,
            size_t                               elementsToValidate)
        {
            for(int idx = 0; idx < problem.gemms.size(); idx++)
            {
                ReferenceSolution<Inputs, Accumulator>::SolveCPU(
                    problem.gemms[idx], inputs.grouped[idx], elementsToValidate);
            }
        }

        uint32_t getInputContractionInputsTypeId(ContractionProblemGemm const& problem)
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

            if(problem.useE())
            {
                if(alphaType != betaType)
                {
                    throw std::runtime_error("Alpha type and beta type must be the same.");
                }
                if(problem.tensors()[ContractionProblemGemm::TENSOR::E].dataType() != betaType)
                {
                    throw std::runtime_error("E type and beta type must be the same.");
                }
            }

            return Tensile::GemmTypeId(problem.a().dataType(),
                                       problem.b().dataType(),
                                       problem.c().dataType(),
                                       problem.d().dataType(),
                                       alphaType,
                                       betaType);
        }

        template <typename Problem, typename Inputs>
        void SolveCPUTemplates(uint32_t const& contractionInputsTypeId,
                               Problem const&  problem,
                               Inputs const&   inputs,
                               size_t          elementsToValidate)
        {
            bool isHPA = false;
            if constexpr(std::is_same<ContractionProblemGemm, Problem>::value)
            {
                isHPA = problem.highPrecisionAccumulate();
            }
            else if constexpr(std::is_same<ContractionProblemGroupedGemm, Problem>::value)
            {
                isHPA = problem.gemms[0].highPrecisionAccumulate();
            }

            switch(contractionInputsTypeId)
            {
            case TypedGemm_S_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_S_S_S>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_D_D_D::TypeId():
            {
                return ReferenceSolution<TypedGemm_D_D_D>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_C_C_C::TypeId():
            {
                return ReferenceSolution<TypedGemm_C_C_C>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_Z_Z_Z::TypeId():
            {
                return ReferenceSolution<TypedGemm_Z_Z_Z>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
#ifdef TENSILE_USE_HALF
            case TypedGemm_H_H_H::TypeId():
            {
                if(isHPA)
                {
                    return ReferenceSolution<TypedGemm_H_H_H, float>::SolveCPU(
                        problem, inputs, elementsToValidate);
                }
                else
                {
                    return ReferenceSolution<TypedGemm_H_H_H>::SolveCPU(
                        problem, inputs, elementsToValidate);
                }
            }
            case TypedGemm_H_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_H_S_S>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_H_H_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_H_H_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
#endif // TENSILE_USE_HALF
            case TypedGemm_I8x4_I32_I32::TypeId():
            {
                return ReferenceSolution<TypedGemm_I8x4_I32_I32>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_I32_I32_I32::TypeId():
            {
                return ReferenceSolution<TypedGemm_I32_I32_I32>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_I8_I8_I32::TypeId():
            {
                return ReferenceSolution<TypedGemm_I8_I8_I32, int32_t>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_I8_I32_I32::TypeId():
            {
                return ReferenceSolution<TypedGemm_I8_I32_I32>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_I8_I32_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_I8_I32_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_I8_I8_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_I8_I8_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
#ifdef TENSILE_USE_BF16
            case TypedGemm_B_B_S::TypeId():
            {
                if(isHPA)
                {
                    return ReferenceSolution<TypedGemm_B_B_S, float>::SolveCPU(
                        problem, inputs, elementsToValidate);
                }
                else
                {
                    return ReferenceSolution<TypedGemm_B_B_S>::SolveCPU(
                        problem, inputs, elementsToValidate);
                }
            }
            case TypedGemm_B_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_B_S_S>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
#endif // TENSILE_USE_BF16

            default:;
            }

            throw std::runtime_error("Data type not implemented.");
        }

        void SolveCPU(ContractionProblem const* problem,
                      ProblemInputs const*      inputs,
                      size_t                    elementsToValidate)
        {
            if(auto groupedProblem = dynamic_cast<ContractionProblemGroupedGemm const*>(problem))
            {
                if(auto refInput = dynamic_cast<ContractionGroupedInputs const*>(inputs))
                {
                    auto contractionInputsTypeId
                        = getInputContractionInputsTypeId(groupedProblem->gemms[0]);
                    SolveCPUTemplates(
                        contractionInputsTypeId, *groupedProblem, *refInput, elementsToValidate);
                }
                else
                    throw std::runtime_error("Unable to cast input to ContractionGroupedInputs.");
            }
            else if(auto gemmProblem = dynamic_cast<ContractionProblemGemm const*>(problem))
            {
                if(auto refInput = dynamic_cast<ContractionInputs const*>(inputs))
                {
                    auto contractionInputsTypeId = getInputContractionInputsTypeId(*gemmProblem);
                    SolveCPUTemplates(
                        contractionInputsTypeId, *gemmProblem, *refInput, elementsToValidate);
                }
                else
                    throw std::runtime_error("Unable to cast input to ContractionInputs.");
            }
            else
            {
                throw std::runtime_error("[Reference] Failed to cast to any ContractionProblem");
            }
        }
    } // namespace Client
} // namespace Tensile
