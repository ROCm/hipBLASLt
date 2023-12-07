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

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/KernelLanguageTypes.hpp>
#include <Tensile/Predicates.hpp>

#include <array>
#include <cstddef>
#include <limits>
#include <vector>

namespace Tensile
{
    namespace Predicates
    {
        /**
 * \addtogroup Predicates
 * @{
 */
        /**
 * @brief ContractionProblemGemm predicates
 */
        namespace Contraction
        {
            struct Free0SizeMultiple
                : public Predicate_CRTP<Free0SizeMultiple, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                size_t index;
                size_t value;

                Free0SizeMultiple() = default;
                Free0SizeMultiple(size_t index, size_t value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "Free0SizeMultiple";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return (!problem.transposeC01() ? problem.freeSizeA(index)
                                                    : problem.freeSizeB(index))
                               % value
                           == 0;
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    int freeSize = !problem.transposeC01() ? problem.freeSizeA(index)
                                                           : problem.freeSizeB(index);
                    return debugEvalCmp(problem, stream, "prob", freeSize, "%", "sol", value);
                }
            };

            struct Free1SizeMultiple
                : public Predicate_CRTP<Free1SizeMultiple, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                size_t index;
                size_t value;

                Free1SizeMultiple() = default;
                Free1SizeMultiple(size_t index, size_t value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "Free1SizeMultiple";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return (!problem.transposeC01() ? problem.freeSizeB(index)
                                                    : problem.freeSizeA(index))
                               % value
                           == 0;
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    int freeSize = !problem.transposeC01() ? problem.freeSizeB(index)
                                                           : problem.freeSizeA(index);
                    return debugEvalCmp(problem, stream, "prob", freeSize, "%", "sol", value);
                }
            };

            struct BatchSizeMultiple
                : public Predicate_CRTP<BatchSizeMultiple, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                size_t index;
                size_t value;

                BatchSizeMultiple() = default;
                BatchSizeMultiple(size_t index, size_t value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "BatchSizeMultiple";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return problem.batchSize(index) % value == 0;
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    return debugEvalCmp(
                        problem, stream, "prob", problem.batchSize(index), "%", "sol", value);
                }
            };

            struct BatchSizeEqual : public Predicate_CRTP<BatchSizeEqual, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                size_t index;
                size_t value;

                BatchSizeEqual() = default;
                BatchSizeEqual(size_t index, size_t value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "BatchSizeEqual";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return problem.batchSize(index) == value;
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    return debugEvalCmp(
                        problem, stream, "prob", problem.batchSize(index), "==", "sol", value);
                }
            };

            struct BoundSizeMultiple
                : public Predicate_CRTP<BoundSizeMultiple, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                int64_t index;
                size_t  value;

                BoundSizeMultiple() = default;
                BoundSizeMultiple(size_t index, size_t value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "BoundSizeMultiple";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    if(index < 0)
                        return problem.boundSize(problem.boundIndices().size() + index) % value
                               == 0;
                    else
                        return problem.boundSize(index) % value == 0;
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    int index_mod = index < 0 ? problem.boundIndices().size() + index : index;
                    return debugEvalCmp(
                        problem, stream, "prob", problem.boundSize(index_mod), "%", "sol", value);
                }
            };

            struct ProblemSizeEqual
                : public Predicate_CRTP<ProblemSizeEqual, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                size_t index;
                size_t value;

                ProblemSizeEqual() = default;
                ProblemSizeEqual(size_t index, size_t value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "ProblemSizeEqual";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return problem.problemSizes()[index] == value;
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    return debugEvalCmp(
                        problem, stream, "prob", problem.problemSizes()[index], "==", "sol", value);
                }
            };

            struct MaxProblemSizeGreaterThan
                : public Predicate_CRTP<MaxProblemSizeGreaterThan, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                size_t value;

                MaxProblemSizeGreaterThan() = default;
                MaxProblemSizeGreaterThan(size_t value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "MaxProblemSizeGreaterThan";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return problem.maxProblemSize() > value;
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    return debugEvalCmp(
                        problem, stream, "prob", problem.maxProblemSize(), ">", "sol", value);
                }
            };

            // If the tensor contains no free dimensions, then the first batch dimension
            // serves as the leading free size
            struct LeadingFree0SizesGreaterOrEqual
                : public Predicate_CRTP<LeadingFree0SizesGreaterOrEqual, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                size_t value;

                LeadingFree0SizesGreaterOrEqual() = default;
                LeadingFree0SizesGreaterOrEqual(size_t value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "LeadingFree0SizesGreaterOrEqual";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    assert(problem.batchIndices().size() <= 1);
                    // TODO: this is not quite right since it assumes batchSize(0) is lowest
                    // order in index assignments
                    //   If tensor contains multiple batch dims this may not be true.
                    //   Really should modify Contractions.py to select SizeN >= value, based on
                    //   desired index requirement
                    return (problem.freeIndicesA().size() ? problem.freeSizeA(0) >= value
                                                          : problem.batchSize(0) >= value);
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    int size = problem.freeIndicesA().size() ? problem.freeSizeA(0)
                                                             : problem.batchSize(0);
                    return debugEvalCmp(problem, stream, "prob", size, ">=", "sol", value);
                }
            };

            struct LeadingFree1SizesGreaterOrEqual
                : public Predicate_CRTP<LeadingFree1SizesGreaterOrEqual, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                size_t value;

                LeadingFree1SizesGreaterOrEqual() = default;
                LeadingFree1SizesGreaterOrEqual(size_t value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "LeadingFree1SizesGreaterOrEqual";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    assert(problem.batchIndices().size() <= 1);
                    // TODO: this is not quite right since it assumes batchSize(0) is lowest
                    // order in index assignments
                    //   If tensor contains multiple batch dims this may not be true.
                    //   Really should modify Contractions.py to select SizeN >= value, based on
                    //   desired index requirement
                    if(problem.groupedGemm())
                        return (problem.freeIndicesB().size()
                                    ? problem.freeSizeB(0) >= value || problem.freeSizeB(0) == 0
                                    : problem.batchSize(0) >= value || problem.freeSizeB(0) == 0);
                    return (problem.freeIndicesB().size() ? problem.freeSizeB(0) >= value
                                                          : problem.batchSize(0) >= value);
                }
                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    int size = problem.freeIndicesB().size() ? problem.freeSizeB(0)
                                                             : problem.batchSize(0);
                    return debugEvalCmp(problem, stream, "prob", size, ">=", "sol", value);
                }
            };

            struct SizeEqual : public Predicate_CRTP<SizeEqual, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                size_t index;
                size_t value;

                SizeEqual() = default;
                SizeEqual(size_t index, size_t value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "SizeEqual";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return problem.size(index) == value;
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    return debugEvalCmp(
                        problem, stream, "prob", problem.size(index), "==", "sol", value);
                }
            };

            struct SizeGreaterThan : public Predicate_CRTP<SizeGreaterThan, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                size_t index;
                size_t value;

                SizeGreaterThan() = default;
                SizeGreaterThan(size_t index, size_t value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "SizeGreaterThan";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return (problem.size(index) > value);
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    return debugEvalCmp(
                        problem, stream, "prob", problem.size(index), ">", "sol", value);
                }
            };

            struct SizeLessThan : public Predicate_CRTP<SizeLessThan, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                size_t index;
                size_t value;

                SizeLessThan() = default;
                SizeLessThan(size_t index, size_t value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "SizeLessThan";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return (problem.size(index) < value);
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    return debugEvalCmp(
                        problem, stream, "prob", problem.size(index), "<", "sol", value);
                }
            };

            struct SizeMultiple : public Predicate_CRTP<SizeMultiple, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                size_t index;
                size_t value;

                SizeMultiple() = default;
                SizeMultiple(size_t index, size_t value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "SizeMultiple";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return ((problem.size(index) % value) == 0);
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    return debugEvalCmp(
                        problem, stream, "prob", problem.size(index), "%", "sol", value);
                }
            };

            struct Range
            {
                size_t min = 0;
                size_t max = std::numeric_limits<size_t>::max();
            };

            inline std::ostream& operator<<(std::ostream& stream, Range const& range)
            {
                return stream << "min: " << range.min << ", max: " << range.max;
            }

            struct SizeInRange : public Predicate_CRTP<SizeInRange, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                size_t index;
                Range  value;

                SizeInRange() = default;
                SizeInRange(size_t index, Range value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "SizeInRange";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return (problem.size(index) >= value.min) && (problem.size(index) < value.max);
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    return debugEvalCmp(problem,
                                        stream,
                                        "prob_low",
                                        problem.size(index),
                                        ">=",
                                        "sol_low",
                                        value.min,
                                        "prob_high",
                                        problem.size(index),
                                        "<",
                                        "sol_high",
                                        value.max);
                }
            };

            struct StrideAEqual : public Predicate_CRTP<StrideAEqual, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                size_t index;
                size_t value;

                StrideAEqual() = default;
                StrideAEqual(size_t index, size_t value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "StrideAEqual";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return problem.a().strides()[index] == value;
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    return debugEvalCmp(
                        problem, stream, "prob", problem.a().strides()[index], "==", "sol", value);
                }
            };

            struct StrideBEqual : public Predicate_CRTP<StrideBEqual, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                size_t index;
                size_t value;

                StrideBEqual() = default;
                StrideBEqual(size_t index, size_t value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "StrideBEqual";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return problem.b().strides()[index] == value;
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    return debugEvalCmp(
                        problem, stream, "prob", problem.b().strides()[index], "==", "sol", value);
                }
            };

            struct StrideCEqual : public Predicate_CRTP<StrideCEqual, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                size_t index;
                size_t value;

                StrideCEqual() = default;
                StrideCEqual(size_t index, size_t value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "StrideCEqual";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return problem.c().strides()[index] == value;
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    return debugEvalCmp(
                        problem, stream, "prob", problem.c().strides()[index], "==", "sol", value);
                }
            };

            struct StrideDEqual : public Predicate_CRTP<StrideDEqual, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                size_t index;
                size_t value;

                StrideDEqual() = default;
                StrideDEqual(size_t index, size_t value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "StrideDEqual";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return problem.d().strides()[index] == value;
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    return debugEvalCmp(
                        problem, stream, "prob", problem.d().strides()[index], "==", "sol", value);
                }
            };

            struct LDCEqualsLDD : public Predicate_CRTP<LDCEqualsLDD, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = false
                };
                static std::string Type()
                {
                    return "LDCEqualsLDD";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return problem.c().strides()[1] == problem.d().strides()[1];
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    return debugEvalCmp(problem,
                                        stream,
                                        "prob_c_stride",
                                        problem.c().strides()[1],
                                        "==",
                                        "prob_d_stride",
                                        problem.d().strides()[1]);
                }
            };

            struct CEqualsD : public Predicate_CRTP<CEqualsD, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = false
                };
                static std::string Type()
                {
                    return "CEqualsD";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return problem.cEqualsD();
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    return debugEvalCmp(
                        problem, stream, "prob", problem.cEqualsD(), "==", "sol", true);
                }
            };

            struct BetaZero : public Predicate_CRTP<BetaZero, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = false
                };
                BetaZero() = default;

                static std::string Type()
                {
                    return "BetaZero";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return problem.beta() == 0.0;
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    return debugEvalCmp(problem, stream, "prob", problem.beta(), "==", "sol", 0.0);
                }
            };

            struct BetaOne : public Predicate_CRTP<BetaOne, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = false
                };
                BetaOne() = default;

                static std::string Type()
                {
                    return "BetaOne";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return problem.beta() == 1.0;
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    return debugEvalCmp(problem, stream, "prob", problem.beta(), "==", "sol", 1.0);
                }
            };

            struct HighPrecisionAccumulateEqual
                : public Predicate_CRTP<HighPrecisionAccumulateEqual, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                bool value;

                HighPrecisionAccumulateEqual() = default;
                HighPrecisionAccumulateEqual(bool value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "HighPrecisionAccumulate";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return problem.highPrecisionAccumulate() == value;
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    return debugEvalCmp(problem,
                                        stream,
                                        "prob",
                                        problem.highPrecisionAccumulate(),
                                        "==",
                                        "sol",
                                        value);
                }
            };

            struct KernelLanguageCompatible
                : public Predicate_CRTP<KernelLanguageCompatible, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                KernelLanguage value;

                KernelLanguageCompatible() = default;
                KernelLanguageCompatible(KernelLanguage value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "KernelLanguageCompatible";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return problem.kernelLanguage() == value
                           || problem.kernelLanguage() == KernelLanguage::Any;
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    return debugEvalCmp(problem,
                                        stream,
                                        "prob",
                                        problem.kernelLanguage(),
                                        "==",
                                        "sol",
                                        value,
                                        "prob",
                                        problem.kernelLanguage(),
                                        "==",
                                        "sol_any",
                                        KernelLanguage::Any);
                }
            };

            struct DeterministicModeEqual
                : public Predicate_CRTP<DeterministicModeEqual, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                bool value;

                DeterministicModeEqual() = default;
                DeterministicModeEqual(bool value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "DeterministicMode";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return problem.deterministicMode() == value;
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    return debugEvalCmp(
                        problem, stream, "prob", problem.deterministicMode(), "==", "sol", value);
                }
            };

            struct AlphaValue : public Predicate_CRTP<AlphaValue, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };

                ScalarValue value;

                AlphaValue() = default;
                AlphaValue(ScalarValue value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "AlphaValue";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return problem.alphaRestriction() == value || value == ScalarValue::Any;
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    return debugEvalCmp(problem,
                                        stream,
                                        "prob",
                                        problem.alphaRestriction(),
                                        "==",
                                        "sol",
                                        value,
                                        "sol",
                                        value,
                                        "==",
                                        "sol_any",
                                        ScalarValue::Any);
                }
            };

            struct BetaValue : public Predicate_CRTP<BetaValue, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };

                ScalarValue value;

                BetaValue() = default;
                BetaValue(ScalarValue value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "BetaValue";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return problem.betaRestriction() == value || value == ScalarValue::Any;
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    return debugEvalCmp(problem,
                                        stream,
                                        "prob",
                                        problem.betaRestriction(),
                                        "==",
                                        "sol",
                                        value,
                                        "sol",
                                        value,
                                        "==",
                                        "sol_any",
                                        ScalarValue::Any);
                }
            };

            struct TypesEqual : public Predicate_CRTP<TypesEqual, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                TypesEqual() = default;

                std::array<DataType, 5> value;

                static std::string Type()
                {
                    return "TypesEqual";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return problem.a().dataType() == value[0] && problem.b().dataType() == value[1]
                           && problem.c().dataType() == value[2]
                           && problem.d().dataType() == value[3]
                           && problem.computeInputType() == value[4];
                }

                virtual std::string toString() const override
                {
                    return concatenate(this->type(),
                                       "(a:",
                                       value[0],
                                       ", b:",
                                       value[1],
                                       ", c:",
                                       value[2],
                                       ", d:",
                                       value[3],
                                       ", compute input type:",
                                       value[4]);
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    return debugEvalCmp(problem,
                                        stream,
                                        "prob_a",
                                        problem.a().dataType(),
                                        "==",
                                        "sol_a",
                                        value[0],
                                        "prob_b",
                                        problem.b().dataType(),
                                        "==",
                                        "sol_b",
                                        value[1],
                                        "prob_c",
                                        problem.c().dataType(),
                                        "==",
                                        "sol_c",
                                        value[2],
                                        "prob_d",
                                        problem.d().dataType(),
                                        "==",
                                        "sol_d",
                                        value[3],
                                        "prob_compute",
                                        problem.computeInputType(),
                                        "==",
                                        "sol_compute",
                                        value[4]);
                }
            };

            struct OperationIdentifierEqual
                : public Predicate_CRTP<OperationIdentifierEqual, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                OperationIdentifierEqual() = default;

                std::string value;

                static std::string Type()
                {
                    return "OperationIdentifierEqual";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return problem.operationIdentifier() == value;
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    return debugEvalCmp(
                        problem, stream, "prob", problem.operationIdentifier(), "==", "sol", value);
                }
            };

            struct BufferLoadOffsetLimitCheck
                : public Predicate_CRTP<BufferLoadOffsetLimitCheck, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                BufferLoadCheckPacket value;

                BufferLoadOffsetLimitCheck() = default;
                BufferLoadOffsetLimitCheck(BufferLoadCheckPacket value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "BufferLoadOffsetLimitCheck";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    const uint64_t TWO_POW_32 = 4294967296;
                    return (problem.a().strides()[1] * value.depthUorMT0 + value.shiftPtrElemA)
                                   * problem.a().elementBytes()
                               < TWO_POW_32
                           && (problem.b().strides()[1] * value.depthUorMT1 + value.shiftPtrElemB)
                                      * problem.b().elementBytes()
                                  < TWO_POW_32;
                }

                virtual std::string toString() const override
                {
                    return concatenate(this->type(),
                                       "(DU/MT0:",
                                       value.depthUorMT0,
                                       ", DU/MT1:",
                                       value.depthUorMT1,
                                       ", ShiftPtrPadElementA:",
                                       value.shiftPtrElemA,
                                       ", ShiftPtrPadElementB:",
                                       value.shiftPtrElemB,
                                       ")");
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    bool rv = (*this)(problem);

                    stream << rv << ": " << *this << ": ("
                           << " (" << problem.a().strides()[1] << " * " << value.depthUorMT0
                           << " + " << value.shiftPtrElemA << ") * " << problem.a().elementBytes()
                           << " < 4294967296 && "
                           << " (" << problem.b().strides()[1] << " * " << value.depthUorMT1
                           << " + " << value.shiftPtrElemB << ") * " << problem.b().elementBytes()
                           << " < 4294967296"
                           << ")" << std::endl;

                    return rv;
                }
            };

            struct BufferLoadOffsetLimitCheck_Beta
                : public Predicate_CRTP<BufferLoadOffsetLimitCheck_Beta, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                size_t value;

                BufferLoadOffsetLimitCheck_Beta() = default;
                BufferLoadOffsetLimitCheck_Beta(size_t value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "BufferLoadOffsetLimitCheck_Beta";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    if(problem.c().empty() || problem.beta() == 0)
                    {
                        return true;
                    }
                    else
                    {
                        const uint64_t TWO_POW_32 = 4294967296;
                        return problem.c().strides()[1] * problem.c().elementBytes() * value
                               < TWO_POW_32;
                    }
                }

                virtual std::string toString() const override
                {
                    return concatenate(this->type(), "(MT1:", value, ")");
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    bool rv = (*this)(problem);

                    stream << rv << ": " << *this << ": (" << problem.c().strides()[1] << " * "
                           << problem.c().elementBytes() << " * " << value << " < 4294967296"
                           << ")" << std::endl;

                    return rv;
                }
            };

            struct BufferStoreOffsetLimitCheck
                : public Predicate_CRTP<BufferStoreOffsetLimitCheck, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                size_t value;

                BufferStoreOffsetLimitCheck() = default;
                BufferStoreOffsetLimitCheck(size_t value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "BufferStoreOffsetLimitCheck";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    const uint64_t TWO_POW_32 = 4294967296;
                    return problem.d().strides()[1] * problem.d().elementBytes() * value
                           < TWO_POW_32;
                }

                virtual std::string toString() const override
                {
                    return concatenate(this->type(), "(MT1:", value, ")");
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    bool rv = (*this)(problem);
                    stream << rv << ": " << *this << ": (" << problem.d().strides()[1] << " * "
                           << problem.d().elementBytes() << " * " << value << " < 4294967296"
                           << ")" << std::endl;
                    return rv;
                }
            };

            struct WorkspaceCheck : public Predicate_CRTP<WorkspaceCheck, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
#define MAX_GSU_WORKSPACE_SIZE 128 * 1024 * 1024
                size_t             index;
                std::array<int, 3> value;

                WorkspaceCheck() = default;
                WorkspaceCheck(size_t index, std::array<int, 3> value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "WorkspaceCheck";
                }

                static size_t
                    reductionSize(ContractionProblemGemm const& problem, int& elemC, int& elemBias)
                {
                    size_t reductionSize = 0;
                    // 2d reduction
                    if(problem.useGradient() && problem.useBias()
                       && problem.getParams().biasEnum() != DataType::None)
                    {
                        if(problem.biasSrc() == ContractionProblemGemm::TENSOR::D && (elemC == 0))
                            reductionSize += problem.d().totalLogicalElements()
                                             * problem.computeTypeElementSize();
                        else if(problem.biasSrc() == ContractionProblemGemm::TENSOR::A)
                        {
                            reductionSize += problem.freeSizeA(0) * elemBias;
                        }
                        else if(problem.biasSrc() == ContractionProblemGemm::TENSOR::B)
                        {
                            reductionSize += problem.freeSizeB(0) * elemBias;
                        }
                    }
                    return reductionSize;
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    int gsu = problem.getParams().gsu() > 0 ? problem.getParams().gsu() : value[2];
                    int gsuMultiplier = gsu > 1 ? gsu : 0;
                    int elemC         = value[0] * gsuMultiplier;
                    int elemBias      = value[1] * gsuMultiplier;
                    size_t rs         = reductionSize(problem, elemC, elemBias);
                    if(problem.d().totalLogicalElements() * elemC > MAX_GSU_WORKSPACE_SIZE)
                        return 0;

                    if(problem.groupedGemm())
                        return problem.workspaceSizeGroupedGemm() <= problem.workspaceSize();
                    else
                        return problem.d().totalLogicalElements() * elemC + rs
                               <= problem.workspaceSize();
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    int gsu = problem.getParams().gsu() > 0 ? problem.getParams().gsu() : value[2];
                    int gsuMultiplier = gsu > 1 ? gsu : 0;
                    int elemC         = value[0] * gsuMultiplier;
                    int elemBias      = value[1] * gsuMultiplier;
                    size_t rs         = reductionSize(problem, elemC, elemBias);

                    if(problem.d().totalLogicalElements() * elemC > MAX_GSU_WORKSPACE_SIZE)
                        return debugEvalCmp(problem,
                                            stream,
                                            "prob",
                                            problem.d().totalLogicalElements() * elemC,
                                            "<=",
                                            "max gsu workspace size",
                                            MAX_GSU_WORKSPACE_SIZE);

                    if(problem.groupedGemm())
                        return debugEvalCmp(problem,
                                            stream,
                                            "prob",
                                            problem.workspaceSizeGroupedGemm(),
                                            "<=",
                                            "max",
                                            problem.workspaceSize());
                    return debugEvalCmp(problem,
                                        stream,
                                        "prob",
                                        problem.d().totalLogicalElements() * value[0] + rs,
                                        "<=",
                                        "max",
                                        problem.workspaceSize());
                }
            };

            struct PersistentKernelCheck
                : public Predicate_CRTP<PersistentKernelCheck, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = false
                };
                PersistentKernelCheck() = default;

                static std::string Type()
                {
                    return "PersistentKernelCheck";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return problem.getPersistentKernelEligibility();
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    bool rv = (*this)(problem);
                    stream << rv << ": " << this->type() << std::endl;
                    return rv;
                }
            };

            struct GlobalSplitUCheckMinK
                : public Predicate_CRTP<GlobalSplitUCheckMinK, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };

                std::array<int, 2> value;

                GlobalSplitUCheckMinK() = default;
                GlobalSplitUCheckMinK(std::array<int, 2> value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "GlobalSplitUCheckMinK";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    size_t minK
                        = (problem.getParams().gsu() > 0 ? problem.getParams().gsu() : value[1]);
                    if(minK == 1)
                        minK = 0;
                    minK *= value[0];
                    return problem.boundSize(0) >= minK;
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    size_t minK
                        = (problem.getParams().gsu() > 0 ? problem.getParams().gsu() : value[1]);
                    if(minK == 1)
                        minK = 0;
                    minK *= value[0];
                    return debugEvalCmp(
                        problem, stream, "prob", problem.boundSize(0), ">=", "sol", minK);
                }
            };

            struct CDStridesEqual : public Predicate_CRTP<CDStridesEqual, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = false
                };

                CDStridesEqual() = default;

                static std::string Type()
                {
                    return "CDStridesEqual";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return problem.c().strides() == problem.d().strides();
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    return debugEvalCmp(problem,
                                        stream,
                                        "prob_c_stride",
                                        problem.c().strides(),
                                        "==",
                                        "prob_d_stride",
                                        problem.d().strides());
                }
            };

            struct StridedBatchedEqual
                : public Predicate_CRTP<StridedBatchedEqual, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                bool value;

                StridedBatchedEqual() = default;
                StridedBatchedEqual(bool value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "StridedBatched";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return problem.stridedBatched() == value;
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    return debugEvalCmp(
                        problem, stream, "prob", problem.stridedBatched(), "==", "sol", value);
                }
            };

            struct GroupedGemmEqual
                : public Predicate_CRTP<GroupedGemmEqual, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                bool value;

                GroupedGemmEqual() = default;
                GroupedGemmEqual(bool value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "GroupedGemm";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return problem.groupedGemm() == value;
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    return debugEvalCmp(
                        problem, stream, "prob", problem.groupedGemm(), "==", "sol", value);
                }
            };

            struct CUEfficiency : public Predicate_CRTP<CUEfficiency, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = false
                };

                CUEfficiency() = default;

                static std::string Type()
                {
                    return "CUEfficiency";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    if(problem.performanceMetric() == PerformanceMetric::CUEfficiency)
                    {
                        return true;
                    }
                    else if(problem.performanceMetric() == PerformanceMetric::Auto)
                    {
                        // True if total flops below a constant threshold
                        // Current threshold chosen naively as the flops for a
                        // 1500x1500 square matrix multiply
                        return problem.flopCount() < size_t(1500ll * 1500ll * 1500ll * 2);
                    }
                    else
                    {
                        return false;
                    }
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    bool rv = (*this)(problem);
                    if(problem.performanceMetric() == PerformanceMetric::CUEfficiency)
                        stream
                            << rv << ": " << this->type()
                            << " (problem.performanceMetric() == PerformanceMetric::CUEfficiency)"
                            << std::endl;
                    else if(problem.performanceMetric() == PerformanceMetric::Auto)
                        stream << rv << ": " << this->type()
                               << " ((problem.performanceMetric() == PerformanceMetric::Auto) &&"
                               << " (problem.flopCount() < 1500 * 1500 * 1500 * 2))" << std::endl;
                    else
                        stream << rv << ": " << this->type()
                               << " ((problem.performanceMetric() != "
                                  "PerformanceMetric::CUEfficiency) &&"
                               << " (problem.performanceMetric() != PerformanceMetric::Auto))"
                               << std::endl;
                    return rv;
                }
            };

            struct Experimental : public Predicate_CRTP<Experimental, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = false
                };

                Experimental() = default;

                static std::string Type()
                {
                    return "Experimental";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return (problem.performanceMetric() == PerformanceMetric::Experimental);
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    return debugEvalCmp(problem,
                                        stream,
                                        "prob",
                                        problem.performanceMetric(),
                                        "==",
                                        "sol: PerformanceMetric::Experimental",
                                        PerformanceMetric::Experimental);
                }
            };

            struct Fp16AltImpl : public Predicate_CRTP<Fp16AltImpl, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = false
                };

                Fp16AltImpl() = default;

                static std::string Type()
                {
                    return "Fp16AltImpl";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return problem.fp16AltImpl();
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    bool rv = (*this)(problem);
                    stream << rv << ": " << this->type() << std::endl;
                    return rv;
                }
            };

            struct EqualityMatching
                : public Predicate_CRTP<EqualityMatching, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = false
                };

                EqualityMatching() = default;

                static std::string Type()
                {
                    return "EqualityMatching";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return true;
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    bool rv = (*this)(problem);
                    stream << rv << ": " << this->type() << std::endl;
                    return rv;
                }
            };

            struct FreeSizeMatching
                : public Predicate_CRTP<FreeSizeMatching, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = false
                };

                FreeSizeMatching() = default;

                static std::string Type()
                {
                    return "FreeSizeMatching";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return true;
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    bool rv = (*this)(problem);
                    stream << rv << ": " << this->type() << std::endl;
                    return rv;
                }
            };

            struct UseGradientEqual
                : public Predicate_CRTP<UseGradientEqual, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                bool value;

                UseGradientEqual() = default;
                UseGradientEqual(bool value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "UseGradient";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return problem.useGradient() == value;
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    return debugEvalCmp(
                        problem, stream, "prob", problem.useGradient(), "==", "sol", value);
                }
            };

            // Activation
            struct ActivationEqual : public Predicate_CRTP<ActivationEqual, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                ActivationType value;

                ActivationEqual() = default;
                ActivationEqual(ActivationType value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "Activation";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return problem.activationType() == value;
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    return debugEvalCmp(
                        problem, stream, "prob", problem.activationType(), "==", "sol", value);
                }
            };

            struct ActivationEnumWhiteList
                : public Predicate_CRTP<ActivationEnumWhiteList, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                ActivationEnumWhiteList() = default;

                std::vector<ActivationType> value;

                static std::string Type()
                {
                    return "ActivationEnumWhiteList";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    if(problem.activationType() == ActivationType::All)
                    {
                        for(size_t i = 0; i < value.size(); i++)
                        {
                            if(value[i] == problem.getParams().activationEnum())
                            {
                                return true;
                            }
                        }
                        return false;
                    }
                    return true;
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    bool        rv        = (*this)(problem);
                    std::string actString = "";
                    for(size_t i = 0; i < value.size(); i++)
                    {
                        actString += ToString(value[i]);
                        if(i < value.size() - 1)
                            actString += ", ";
                    }
                    stream << rv << ": " << this->type()
                           << " (The supported activations are: " + actString << ")" << std::endl;
                    return rv;
                }
            };

            struct ActivationComputeTypeEqual
                : public Predicate_CRTP<ActivationComputeTypeEqual, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                DataType value;

                ActivationComputeTypeEqual() = default;
                ActivationComputeTypeEqual(DataType value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "ActivationComputeType";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    if(problem.activationType() == ActivationType::None)
                        return true;
                    return problem.activationComputeType() == value;
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    bool rv = (*this)(problem);
                    if(problem.activationType() != ActivationType::None)
                        return debugEvalCmp(problem,
                                            stream,
                                            "prob",
                                            problem.activationComputeType(),
                                            "==",
                                            "sol",
                                            value);
                    return rv;
                }
            };

            struct ActivationNoGuardEqual
                : public Predicate_CRTP<ActivationNoGuardEqual, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                bool value;

                ActivationNoGuardEqual() = default;
                ActivationNoGuardEqual(bool value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "ActivationNoGuard";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return problem.activationNoGuard() == value;
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    return debugEvalCmp(
                        problem, stream, "prob", problem.activationNoGuard(), "==", "sol", value);
                }
            };

            struct UseBiasEqual : public Predicate_CRTP<UseBiasEqual, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                bool value;

                UseBiasEqual() = default;
                UseBiasEqual(bool value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "UseBias";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return problem.useBias() == value;
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    return debugEvalCmp(
                        problem, stream, "prob", problem.useBias(), "==", "sol", value);
                }
            };

            struct UseEEqual : public Predicate_CRTP<UseEEqual, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                bool value;

                UseEEqual() = default;
                UseEEqual(bool value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "UseE";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return problem.useE() == value;
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    return debugEvalCmp(
                        problem, stream, "prob", problem.useE(), "==", "sol", value);
                }
            };

            struct UseScaleABEqual : public Predicate_CRTP<UseScaleABEqual, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                bool value;

                UseScaleABEqual() = default;
                UseScaleABEqual(bool value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "UseScaleAB";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return problem.useScaleAB() == value;
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    return debugEvalCmp(
                        problem, stream, "prob", problem.useScaleAB(), "==", "sol", value);
                }
            };

            struct UseScaleCDEqual : public Predicate_CRTP<UseScaleCDEqual, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                bool value;

                UseScaleCDEqual() = default;
                UseScaleCDEqual(bool value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "UseScaleCD";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return problem.useScaleCD() == value;
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    return debugEvalCmp(
                        problem, stream, "prob", problem.useScaleCD(), "==", "sol", value);
                }
            };

            struct UseScaleAlphaVecEqual
                : public Predicate_CRTP<UseScaleAlphaVecEqual, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                bool value;

                UseScaleAlphaVecEqual() = default;
                UseScaleAlphaVecEqual(bool value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "UseScaleAlphaVec";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return problem.useScaleAlphaVec() == value;
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    return debugEvalCmp(
                        problem, stream, "prob", problem.useScaleAlphaVec(), "==", "sol", value);
                }
            };

            struct BiasDataTypeWhiteList
                : public Predicate_CRTP<BiasDataTypeWhiteList, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                BiasDataTypeWhiteList() = default;

                std::vector<DataType> value;

                static std::string Type()
                {
                    return "BiasDataTypeWhiteList";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    if(problem.useBias())
                    {
                        for(size_t i = 0; i < value.size(); i++)
                        {
                            if(value[i] == problem.bias().dataType())
                            {
                                return true;
                            }
                        }
                        return false;
                    }
                    return true;
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    bool        rv         = (*this)(problem);
                    std::string biasString = "";
                    for(size_t i = 0; i < value.size(); i++)
                    {
                        biasString += ToString(value[i]);
                        if(i < value.size() - 1)
                            biasString += ", ";
                    }
                    stream << rv << ": " << this->type()
                           << " (The supported bias types are: " + biasString << ")" << std::endl;
                    return rv;
                }
            };

            struct BiasSrcWhiteList
                : public Predicate_CRTP<BiasSrcWhiteList, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                BiasSrcWhiteList() = default;

                // This is actually the index of the problem type
                std::vector<int> value;

                static std::string Type()
                {
                    return "BiasSrcWhiteList";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    if(problem.useBias())
                    {
                        auto& tensor = problem.tensor(ContractionProblemGemm::TENSOR::BIAS);
                        if(tensor.sizes().size() == 0)
                            return false;

                        for(size_t i = 0; i < value.size(); i++)
                        {
                            if(value[i] == static_cast<int>(problem.biasSrc()))
                            {
                                // Check if the length is set correctly.
                                auto& length = tensor.sizes()[0];
                                if(problem.biasSrc() == ContractionProblemGemm::TENSOR::A
                                   || problem.biasSrc() == ContractionProblemGemm::TENSOR::D)
                                {
                                    if(length != problem.d().sizes()[0])
                                        return false;
                                }
                                else if(problem.biasSrc() == ContractionProblemGemm::TENSOR::B)
                                {
                                    if(length != problem.d().sizes()[1])
                                        return false;
                                }
                                return true;
                            }
                        }
                        return false;
                    }
                    return true;
                }

                virtual std::string toString() const override
                {
                    std::string biasString = "";
                    for(size_t i = 0; i < value.size(); i++)
                    {
                        biasString += ToString(value[i]) + ", ";
                    }
                    return std::string("The supported bias source are: " + biasString);
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    bool        rv         = (*this)(problem);
                    std::string biasString = "";
                    for(size_t i = 0; i < value.size(); i++)
                    {
                        biasString += ToString(value[i]);
                        if(i < value.size() - 1)
                            biasString += ", ";
                    }
                    stream << rv << ": " << this->type()
                           << " (The supported bias source are: " + biasString << ")" << std::endl;
                    return rv;
                }
            };

            struct Sparse : public Predicate_CRTP<Sparse, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                int value;

                Sparse() = default;
                Sparse(int value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "Sparse";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return problem.sparse() == value;
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    return debugEvalCmp(
                        problem, stream, "prob", problem.sparse(), "==", "sol", value);
                }
            };

            struct F32XdlMathOpEqual
                : public Predicate_CRTP<F32XdlMathOpEqual, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                DataType value;

                F32XdlMathOpEqual() = default;
                F32XdlMathOpEqual(DataType value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "F32XdlMathOp";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    return problem.f32XdlMathOp() == value;
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    return debugEvalCmp(
                        problem, stream, "prob", problem.f32XdlMathOp(), "==", "sol", value);
                }
            };

            struct SupportDeviceUserArguments
                : public Predicate_CRTP<SupportDeviceUserArguments, ContractionProblemGemm>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                bool value;

                SupportDeviceUserArguments() = default;
                SupportDeviceUserArguments(bool value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "SupportDeviceUserArguments";
                }

                virtual bool operator()(ContractionProblemGemm const& problem) const override
                {
                    if(problem.getUseDeviceUserArguments())
                        return problem.getUseDeviceUserArguments() == value;
                    return true;
                }

                virtual bool debugEval(ContractionProblemGemm const& problem,
                                       std::ostream&                 stream) const override
                {
                    bool rv = (*this)(problem);
                    if(problem.getUseDeviceUserArguments())
                        return debugEvalCmp(problem,
                                            stream,
                                            "prob",
                                            problem.getUseDeviceUserArguments(),
                                            "==",
                                            "sol",
                                            value);
                    return rv;
                }
            };
        } // namespace Contraction

        /**
 * @}
 */
    } // namespace Predicates
} // namespace Tensile
