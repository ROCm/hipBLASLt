/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2024 Advanced Micro Devices, Inc.
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
#include <algorithm>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt_datatype2string.hpp>
#include <hipblaslt_init.hpp>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

struct MatrixTransformIO
{
    MatrixTransformIO()                 = default;
    virtual ~MatrixTransformIO()        = default;
    virtual void*  getBuf(size_t i)     = 0;
    virtual size_t elemNumBytes() const = 0;

private:
    MatrixTransformIO(const MatrixTransformIO&)            = delete;
    MatrixTransformIO(MatrixTransformIO&&)                 = delete;
    MatrixTransformIO& operator=(const MatrixTransformIO&) = delete;
    MatrixTransformIO& operator=(MatrixTransformIO&&)      = delete;
};

template <typename DType>
struct TypedMatrixTransformIO : public MatrixTransformIO
{
    TypedMatrixTransformIO(int64_t m, int64_t n, int64_t b, hipblaslt_initialization initMethod)
    {
        hipMalloc(&this->a, m * n * b * sizeof(DType));
        hipMalloc(&this->b, m * n * b * sizeof(DType));
        hipMalloc(&this->c, m * n * b * sizeof(DType));
        init(this->a, m * n * b, initMethod);
        init(this->b, m * n * b, initMethod);
    }

    ~TypedMatrixTransformIO() override
    {
        hipFree(a);
        hipFree(b);
        hipFree(c);
    }

    void* getBuf(size_t i) override
    {
        void* buf[] = {a, b, c};
        return buf[i];
    }

    size_t elemNumBytes() const override
    {
        return sizeof(DType);
    }

private:
    void init(DType* buf, size_t len, hipblaslt_initialization initMethod)
    {
        std::vector<DType> ref(len);

        switch(initMethod)
        {
        case hipblaslt_initialization::rand_int:
            hipblaslt_init<DType>(ref.data(), ref.size(), 1, 1);
            break;
        case hipblaslt_initialization::trig_float:
            hipblaslt_init_cos<DType>(ref.data(), ref.size(), 1, 1);
            break;
        case hipblaslt_initialization::hpl:
            hipblaslt_init_hpl<DType>(ref.data(), ref.size(), 1, 1);
            break;
        case hipblaslt_initialization::special:
            hipblaslt_init_alt_impl_big<DType>(ref.data(), ref.size(), 1, 1);
            break;
        default:
            break;
        }

        hipMemcpy(buf, ref.data(), len * sizeof(DType), hipMemcpyHostToDevice);
    }

private:
    DType* a{};
    DType* b{};
    DType* c{};
};

using MatrixTransformIOPtr = std::unique_ptr<MatrixTransformIO>;
MatrixTransformIOPtr makeMatrixTransformIOPtr(
    hipDataType datatype, int64_t m, int64_t n, int64_t b, hipblaslt_initialization init)
{
    if(datatype == HIP_R_32F)
    {
        return std::make_unique<TypedMatrixTransformIO<hipblasLtFloat>>(m, n, b, init);
    }
    else if(datatype == HIP_R_16F)
    {
        return std::make_unique<TypedMatrixTransformIO<hipblasLtHalf>>(m, n, b, init);
    }
    else if(datatype == HIP_R_16BF)
    {
        return std::make_unique<TypedMatrixTransformIO<hipblasLtBfloat16>>(m, n, b, init);
    }
    else if(datatype == HIP_R_8I)
    {
        return std::make_unique<TypedMatrixTransformIO<int8_t>>(m, n, b, init);
    }
    else if(datatype == HIP_R_32I)
    {
        return std::make_unique<TypedMatrixTransformIO<int32_t>>(m, n, b, init);
    }
    return nullptr;
}

hipDataType str2Datatype(const std::string& typeStr)
{
    if(typeStr == "fp32")
    {
        return HIP_R_32F;
    }
    else if(typeStr == "fp16")
    {
        return HIP_R_16F;
    }
    else if(typeStr == "bf16")
    {
        return HIP_R_16BF;
    }
    else if(typeStr == "i8")
    {
        return HIP_R_8I;
    }
    else if(typeStr == "i32")
    {
        return HIP_R_32I;
    }

    return HIPBLASLT_DATATYPE_INVALID;
}

static void printUsage(const std::string& programName)
{
    std::cout << "This sample demostrates using hipblasLtMatrixTransform to compute C = "
                 "alpha*trans_a(A) + beta*trans_b(B)\n"
              << "Usage: " << programName << " <options>\n"
              << "options:\n"
              << "\t-h, --help\t\t\tShow this help message\n"
              << "\t-m, --m\t\t\t\tSize of dim 0 of matrix C, default is 4\n"
              << "\t-n, --n\t\t\t\tSize of dim 1 of matrix C, default is 8\n"
              << "\t-alpha, --alpha\t\t\tValue of alpha, default 1.0\n"
              << "\t-beta, --beta\t\t\tValue of beta, default 1.0\n"
              << "\t--trans_a\t\t\tTranspose matrix A, default is 0\n"
              << "\t--trans_b\t\t\tTranspose matrix B, default is 0\n"
              << "\t--lda\t\t\t\tLeading dimension of matrix A, default is 0 (auto-deduction)\n"
              << "\t--ldb\t\t\t\tLeading dimension of matrix B, default is 0 (auto-deduction)\n"
              << "\t--ldc\t\t\t\tLeading dimension of matrix C, default is 0 (auto-deduction)\n"
              << "\t--batch_size\t\t\tNumber of batches, default is 1\n"
              << "\t--batch_stride\t\t\tStride between two consecutive batches, default is m*n\n"
              << "\t--datatype\t\t\tDatatype of matrices, default is fp32. Allowed values: i8, "
                 "i32, fp16, bf16 and fp32\n"
              << "\t--scale_datatype\t\tDatatype of scales, default is fp32. Allowed values: fp16 "
                 "and fp32\n"
              << "\t--row_maj_a\t\t\tSpecify matrix A is row-major, default is 0\n"
              << "\t--row_maj_b\t\t\tSpecify matrix B is row-major, default is 0\n"
              << "\t--row_maj_c\t\t\tSpecify matrix C is row-major, default is 0\n"
              << "\t--initialization \t\tInitialize matrix data. Options: rand_int, trig_float, "
                 "hpl(floating). (default is hpl)\n";
}

static int parseArguments(int                       argc,
                          char*                     argv[],
                          hipDataType&              datatype,
                          hipDataType&              scaleDatatype,
                          int64_t&                  m,
                          int64_t&                  n,
                          float&                    alpha,
                          float&                    beta,
                          bool&                     transA,
                          bool&                     transB,
                          uint32_t&                 ldA,
                          uint32_t&                 ldB,
                          uint32_t&                 ldC,
                          bool&                     rowMajA,
                          bool&                     rowMajB,
                          bool&                     rowMajC,
                          int32_t&                  batchSize,
                          int64_t&                  batchStride,
                          hipblaslt_initialization& init)
{
    if(argc >= 2)
    {
        for(int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];

            if((arg.at(0) == '-') || ((arg.at(0) == '-') && (arg.at(1) == '-')))
            {
                if((arg == "-m") && (i + 1 < argc))
                {
                    m = atoi(argv[++i]);
                }
                else if((arg == "-n") && (i + 1 < argc))
                {
                    n = atoi(argv[++i]);
                }
                else if(arg == "--alpha" || arg == "-alpha")
                {
                    alpha = atof(argv[++i]);
                }
                else if(arg == "--beta" || arg == "-beta")
                {
                    beta = atof(argv[++i]);
                }
                else if(arg == "--trans_a")
                {
                    transA = (atoi(argv[++i]) > 0);
                }
                else if(arg == "--trans_b")
                {
                    transB = (atoi(argv[++i]) > 0);
                }
                else if(arg == "--ld_a")
                {
                    ldA = atoi(argv[++i]);
                }
                else if(arg == "--ld_b")
                {
                    ldB = atoi(argv[++i]);
                }
                else if(arg == "--ld_c")
                {
                    ldC = atoi(argv[++i]);
                }
                else if(arg == "--batch_size")
                {
                    batchSize = atoi(argv[++i]);
                }
                else if(arg == "--batch_stride")
                {
                    batchStride = atoi(argv[++i]);
                }
                else if(arg == "--datatype")
                {
                    datatype = str2Datatype(argv[++i]);

                    if(datatype == HIPBLASLT_DATATYPE_INVALID)
                    {
                        std::cerr << "Invalid datatype\n";
                        exit(EXIT_FAILURE);
                    }
                }
                else if(arg == "--scale_datatype")
                {
                    scaleDatatype = str2Datatype(argv[++i]);
                }
                else if(arg == "--row_maj_a")
                {
                    rowMajA = (atoi(argv[++i]) > 0);
                }
                else if(arg == "--row_maj_b")
                {
                    rowMajB = (atoi(argv[++i]) > 0);
                }
                else if(arg == "--row_maj_c")
                {
                    rowMajC = (atoi(argv[++i]) > 0);
                }
                else if(arg == "--initialization" || arg == "--init")
                {
                    const std::string initStr{argv[++i]};

                    if(initStr != "rand_int" && initStr != "trig_float" && initStr != "hpl")
                    {
                        std::cerr << "Invalid initialization type: " << initStr << '\n';
                        return EXIT_FAILURE;
                    }

                    init = string2hipblaslt_initialization(initStr);
                }
                else if(arg == "--help" || arg == "-h")
                {
                    printUsage(argv[0]);
                    exit(EXIT_SUCCESS);
                }
            }
            else
            {
                std::cerr << "error with " << arg << std::endl;
                std::cerr << "option must start with - or --" << std::endl << std::endl;
                exit(EXIT_FAILURE);
                return EXIT_FAILURE;
            }
        }
    }

    return EXIT_SUCCESS;
}

template <bool RowMaj>
int64_t getLeadingDimSize(int64_t numRows, int64_t numCols)
{
    return RowMaj ? numCols : numRows;
}

template <bool RowMaj>
uint32_t getOffset(uint32_t row, uint32_t col, uint32_t ld)
{
    if constexpr(RowMaj)
    {
        return ld * row + col;
    }
    else
    {
        return ld * col + row;
    }
}

template <typename DType, typename ScaleType, bool RowMajA, bool RowMajB, bool RowMajC>
void cpuTransform(DType*       c,
                  const DType* a,
                  const DType* b,
                  ScaleType    alpha,
                  ScaleType    beta,
                  bool         transA,
                  bool         transB,
                  uint32_t     m,
                  uint32_t     n,
                  uint32_t     ldA,
                  uint32_t     ldB,
                  uint32_t     ldC,
                  uint32_t     batchSize,
                  uint32_t     batchStride)
{
    for(uint32_t k = 0; k < batchSize; ++k)
    {
        const int64_t batchOffset = k * int64_t(batchStride);

        for(uint32_t i = 0; i < m; ++i)
        {
            for(uint32_t j = 0; j < n; ++j)
            {
                const auto offsetA
                    = transA ? getOffset<RowMajA>(j, i, ldA) : getOffset<RowMajA>(i, j, ldA);
                const auto offsetB
                    = transB ? getOffset<RowMajB>(j, i, ldB) : getOffset<RowMajB>(i, j, ldB);
                const auto offsetC = getOffset<RowMajC>(i, j, ldC);
                c[batchOffset + offsetC]
                    = a[batchOffset + offsetA] * alpha + b[batchOffset + offsetB] * beta;
            }
        }
    }
}

template <typename SrcDType, typename DType>
void printMatrix(const SrcDType* d,
                 std::size_t     m,
                 std::size_t     n,
                 std::size_t     strideM,
                 std::size_t     strideN,
                 bool            rowMaj)
{
    std::vector<SrcDType> buf(m * n);
    (void)hipMemcpy(buf.data(), d, m * n * sizeof(SrcDType), hipMemcpyHostToDevice);
    auto getOffset = [strideM, strideN](std::size_t row, std::size_t col) -> std::size_t {
        return col * strideN + row * strideM;
    };

    if(rowMaj)
    {
        std::cout << "[\n";
        for(size_t i = 0; i < m; ++i)
        {
            std::cout << "[";
            for(size_t j = 0; j < n; ++j)
            {
                std::cout << static_cast<DType>(buf[getOffset(i, j)]) << ", ";
            }
            std::cout << "],\n";
        }
        std::cout << "]\n";
    }
    else
    {
        std::cout << "[\n";
        for(size_t i = 0; i < n; ++i)
        {
            std::cout << "[";
            for(size_t j = 0; j < m; ++j)
            {
                std::cout << static_cast<DType>(buf[getOffset(j, i)]) << ", ";
            }
            std::cout << "],\n";
        }
        std::cout << "]\n";
    }
}

template <typename SrcDType, typename DType = float>
void printResult(const void* a,
                 const void* b,
                 const void* c,
                 size_t      shapeA0,
                 size_t      shapeA1,
                 size_t      shapeB0,
                 size_t      shapeB1,
                 size_t      shapeC0,
                 size_t      shapeC1,
                 size_t      strideA0,
                 size_t      strideA1,
                 size_t      strideB0,
                 size_t      strideB1,
                 size_t      strideC0,
                 size_t      strideC1,
                 bool        rowMajA,
                 bool        rowMajB,
                 bool        rowMajC)
{
    std::cout << "A:\n";
    printMatrix<SrcDType, DType>((const SrcDType*)a, shapeA0, shapeA1, strideA0, strideA1, rowMajA);
    std::cout << "B:\n";
    printMatrix<SrcDType, DType>((const SrcDType*)b, shapeB0, shapeB1, strideB0, strideB1, rowMajB);
    std::cout << "C:\n";
    printMatrix<SrcDType, DType>((const SrcDType*)c, shapeC0, shapeC1, strideC0, strideC1, rowMajC);
}

int main(int argc, char** argv)
{
    int64_t                  m         = 4;
    int64_t                  n         = 8;
    int32_t                  batchSize = 1;
    float                    alpha     = 1;
    float                    beta      = 1;
    auto                     transA    = false;
    auto                     transB    = false;
    auto                     rowMajA   = false;
    auto                     rowMajB   = false;
    auto                     rowMajC   = false;
    int64_t                  batchStride{};
    uint32_t                 ldA{};
    uint32_t                 ldB{};
    uint32_t                 ldC{};
    hipblaslt_initialization init{hipblaslt_initialization::hpl};
    hipDataType              datatype{HIP_R_32F};
    hipDataType              scaleDatatype{HIP_R_32F};
    parseArguments(argc,
                   argv,
                   datatype,
                   scaleDatatype,
                   m,
                   n,
                   alpha,
                   beta,
                   transA,
                   transB,
                   ldA,
                   ldB,
                   ldC,
                   rowMajA,
                   rowMajB,
                   rowMajC,
                   batchSize,
                   batchStride,
                   init);

    if(!batchStride)
    {
        batchStride = m * n;
    }

    hipblasLtOrder_t orderA = rowMajA ? HIPBLASLT_ORDER_ROW : HIPBLASLT_ORDER_COL;
    hipblasLtOrder_t orderB = rowMajB ? HIPBLASLT_ORDER_ROW : HIPBLASLT_ORDER_COL;
    hipblasLtOrder_t orderC = rowMajC ? HIPBLASLT_ORDER_ROW : HIPBLASLT_ORDER_COL;
    auto             tA     = transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    auto             tB     = transB ? HIPBLAS_OP_T : HIPBLAS_OP_N;

    auto  inputs = makeMatrixTransformIOPtr(datatype, m, n, batchSize, init);
    void* dA     = inputs->getBuf(0);
    void* dB     = inputs->getBuf(1);
    void* dC     = inputs->getBuf(2);

    hipblasLtMatrixTransformDesc_t desc;
    auto                   hipblasLtErr = hipblasLtMatrixTransformDescCreate(&desc, scaleDatatype);
    hipblasLtPointerMode_t pMode        = HIPBLASLT_POINTER_MODE_HOST;
    hipblasLtErr                        = hipblasLtMatrixTransformDescSetAttribute(
        desc,
        hipblasLtMatrixTransformDescAttributes_t::HIPBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE,
        &pMode,
        sizeof(pMode));

    if(hipblasLtErr)
    {
        return EXIT_FAILURE;
    }

    std::pair<int64_t, int64_t> shapeA;
    std::pair<int64_t, int64_t> shapeB;
    shapeA.first  = transA ? n : m;
    shapeA.second = transA ? m : n;
    shapeB.first  = transB ? n : m;
    shapeB.second = transB ? m : n;

    if(!ldA || !ldB || !ldC)
    {
        ldA = rowMajA ? getLeadingDimSize<true>(shapeA.first, shapeA.second)
                      : getLeadingDimSize<false>(shapeA.first, shapeA.second);
        ldB = rowMajB ? getLeadingDimSize<true>(shapeB.first, shapeB.second)
                      : getLeadingDimSize<false>(shapeB.first, shapeB.second);
        ldC = rowMajC ? getLeadingDimSize<true>(m, n) : getLeadingDimSize<false>(m, n);
    }

    hipblasLtErr = hipblasLtMatrixTransformDescSetAttribute(
        desc, HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &tA, sizeof(tA));
    hipblasLtErr = hipblasLtMatrixTransformDescSetAttribute(
        desc, HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSB, &tB, sizeof(tB));
    hipblasLtMatrixLayout_t layoutA, layoutB, layoutC;
    hipblasLtErr
        = hipblasLtMatrixLayoutCreate(&layoutA, datatype, shapeA.first, shapeA.second, ldA);
    hipblasLtErr
        = hipblasLtMatrixLayoutCreate(&layoutB, datatype, shapeB.first, shapeB.second, ldB);
    hipblasLtErr = hipblasLtMatrixLayoutCreate(&layoutC, datatype, m, n, ldC);
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutA,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
        &orderA,
        sizeof(orderA));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutB,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
        &orderB,
        sizeof(orderB));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutC,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
        &orderC,
        sizeof(orderC));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutA,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batchSize,
        sizeof(batchSize));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutB,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batchSize,
        sizeof(batchSize));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutC,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batchSize,
        sizeof(batchSize));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutA,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batchStride,
        sizeof(batchStride));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutB,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batchStride,
        sizeof(batchStride));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutC,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batchStride,
        sizeof(batchStride));
    hipblasLtHandle_t handle{};
    hipblasLtErr = hipblasLtCreate(&handle);
    hipblasLtErr = hipblasLtMatrixTransform(
        handle, desc, &alpha, dA, layoutA, &beta, dB, layoutB, dC, layoutC, nullptr);

    if(hipblasLtErr)
    {
        std::cerr << "Unable to launch hipblasLtMatrixTransform\n";
        return EXIT_FAILURE;
    }

    auto err            = hipStreamSynchronize(nullptr);
    const auto strideA0 = rowMajA ? ldA : 1;
    const auto strideA1 = rowMajA ? 1 : ldA;
    const auto strideB0 = rowMajB ? ldB : 1;
    const auto strideB1 = rowMajB ? 1 : ldB;
    const auto strideC0 = rowMajC ? ldC : 1;
    const auto strideC1 = rowMajC ? 1 : ldC;

    std::cout << "alpha: " << alpha << ", beta: " << beta << '\n';

    if(datatype == HIP_R_32F)
    {
        printResult<hipblasLtFloat>(dA,
                                    dB,
                                    dC,
                                    shapeA.first,
                                    shapeA.second,
                                    shapeB.first,
                                    shapeB.second,
                                    m,
                                    n,
                                    strideA0,
                                    strideA1,
                                    strideB0,
                                    strideB1,
                                    strideC0,
                                    strideC1,
                                    rowMajA,
                                    rowMajB,
                                    rowMajC);
    }
    else if(datatype == HIP_R_16F)
    {
        printResult<hipblasLtHalf>(dA,
                                   dB,
                                   dC,
                                   shapeA.first,
                                   shapeA.second,
                                   shapeB.first,
                                   shapeB.second,
                                   m,
                                   n,
                                   strideA0,
                                   strideA1,
                                   strideB0,
                                   strideB1,
                                   strideC0,
                                   strideC1,
                                   rowMajA,
                                   rowMajB,
                                   rowMajC);
    }
    else if(datatype == HIP_R_16BF)
    {
        printResult<hipblasLtBfloat16>(dA,
                                       dB,
                                       dC,
                                       shapeA.first,
                                       shapeA.second,
                                       shapeB.first,
                                       shapeB.second,
                                       m,
                                       n,
                                       strideA0,
                                       strideA1,
                                       strideB0,
                                       strideB1,
                                       strideC0,
                                       strideC1,
                                       rowMajA,
                                       rowMajB,
                                       rowMajC);
    }
    else if(datatype == HIP_R_8I)
    {
        printResult<int8_t>(dA,
                            dB,
                            dC,
                            shapeA.first,
                            shapeA.second,
                            shapeB.first,
                            shapeB.second,
                            m,
                            n,
                            strideA0,
                            strideA1,
                            strideB0,
                            strideB1,
                            strideC0,
                            strideC1,
                            rowMajA,
                            rowMajB,
                            rowMajC);
    }
    else if(datatype == HIP_R_32I)
    {
        printResult<int32_t>(dA,
                             dB,
                             dC,
                             shapeA.first,
                             shapeA.second,
                             shapeB.first,
                             shapeB.second,
                             m,
                             n,
                             strideA0,
                             strideA1,
                             strideB0,
                             strideB1,
                             strideC0,
                             strideC1,
                             rowMajA,
                             rowMajB,
                             rowMajC);
    }

releaseResource:
    hipblasLtErr = hipblasLtMatrixTransformDescDestroy(desc);
    hipblasLtErr = hipblasLtDestroy(handle);
    hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutA);
    hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutB);
    hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutC);
    return EXIT_SUCCESS;
}
