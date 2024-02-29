/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
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
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt_init.hpp>
#include <iostream>
#include <numeric>

#ifndef CHECK_HIPBLASLT_ERROR
#define CHECK_HIP_ERROR(error)                    \
    if(error != hipSuccess)                       \
    {                                             \
        fprintf(stderr,                           \
                "Hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error),         \
                error,                            \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif

#ifndef CHECK_HIPBLASLT_ERROR
#define CHECK_HIPBLASLT_ERROR(error)                                                      \
    if(error != HIPBLAS_STATUS_SUCCESS)                                                   \
    {                                                                                     \
        fprintf(stderr, "hipBLASLt error(Err=%d) at %s:%d\n", error, __FILE__, __LINE__); \
        fprintf(stderr, "\n");                                                            \
        exit(EXIT_FAILURE);                                                               \
    }
#endif

extern "C" __global__ void flush_icache()
{
    asm __volatile__("s_icache_inv \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t" ::
                         :);
}

class Layer
{
public:
    // User input
    int   m;
    int   n;
    int   k;
    int   batch;
    float alpha;
    float beta;

    hipblaslt_ext::GemmProblemType problem;
    hipblaslt_ext::GemmEpilogue    epilogue;

    // Internal data
    int64_t            ws_size;
    void*              ws   = NULL;
    void*              a    = NULL;
    void*              b    = NULL;
    void*              c    = NULL;
    void*              d    = NULL;
    void*              bias = NULL;
    std::vector<void*> d_a, d_b, d_c, d_d, d_bias;

    // Internal related
    int algo_index = -1;
    int block_count;
    void setData(int                         m,
                 int                         n,
                 int                         k,
                 int                         batch,
                 float                       alpha,
                 float                       beta,
                 hipblasOperation_t          op_a,
                 hipblasOperation_t          op_b,
                 hipblasltDatatype_t         type_a,
                 hipblasltDatatype_t         type_b,
                 hipblasltDatatype_t         type_c,
                 hipblasltDatatype_t         type_d,
                 hipblasLtComputeType_t      type_compute,
                 hipblaslt_ext::GemmEpilogue epilogue)
    {
        this->m                    = m;
        this->n                    = n;
        this->k                    = k;
        this->batch                = batch;
        this->alpha                = alpha;
        this->beta                 = beta;
        this->problem.op_a         = op_a;
        this->problem.op_b         = op_b;
        this->problem.type_a       = type_a;
        this->problem.type_b       = type_b;
        this->problem.type_c       = type_c;
        this->problem.type_d       = type_d;
        this->problem.type_compute = type_compute;
        this->epilogue             = epilogue;
    }
    void initBlock(int block)
    {
        block_count = block;
        for(int b = 0; b < block_count; b++)
        {
            d_a.push_back(NULL);
            d_b.push_back(NULL);
            d_c.push_back(NULL);
            d_d.push_back(NULL);
            d_bias.push_back(NULL);
        }
    }
    ~Layer()
    {
        if(a)
            CHECK_HIP_ERROR(hipFree(a));
        if(b)
            CHECK_HIP_ERROR(hipFree(b));
        if(c)
            CHECK_HIP_ERROR(hipFree(c));
        if(d)
            CHECK_HIP_ERROR(hipFree(d));
        if(bias)
            CHECK_HIP_ERROR(hipFree(bias));
        if(ws)
            CHECK_HIP_ERROR(hipFree(ws));
        for(int b = 0; b < block_count; b++)
        {
            if(d_a[b])
                CHECK_HIP_ERROR(hipFree(d_a[b]));
            if(d_b[b])
                CHECK_HIP_ERROR(hipFree(d_b[b]));
            if(d_c[b])
                CHECK_HIP_ERROR(hipFree(d_c[b]));
            if(d_d[b])
                CHECK_HIP_ERROR(hipFree(d_d[b]));
            if(d_bias[b])
                CHECK_HIP_ERROR(hipFree(d_bias[b]));
        }
    }
};
int32_t type2Size(hipblasltDatatype_t type)
{
    switch(type)
    {
    case hipblasltDatatype_t::HIPBLASLT_R_8F_E4M3:
    case hipblasltDatatype_t::HIPBLASLT_R_8F_E5M2:
        return sizeof(float) / 4;
    case hipblasltDatatype_t::HIPBLASLT_R_32F:
        return sizeof(float);
    case hipblasltDatatype_t::HIPBLASLT_R_16F:
        return sizeof(float) / 2;
    default:
        return 0;
    }
    return 0;
}
void initData(
    hipblasltDatatype_t type, void* data, int m, int n, int lda, int stride, int batch_count)
{
    switch(type)
    {
    case hipblasltDatatype_t::HIPBLASLT_R_8F_E4M3:
    {
        hipblaslt_init_cos<hipblaslt_f8>((hipblaslt_f8*)data, m, n, lda, stride, batch_count);
    }
    break;
    case hipblasltDatatype_t::HIPBLASLT_R_16F:
    {
        hipblaslt_init_cos<hipblasLtHalf>((hipblasLtHalf*)data, m, n, lda, stride, batch_count);
    }
    break;
    default:
        exit(1);
    }
    return;
}

void initAndCopy(void**              data,
                 std::vector<void*>& d_data,
                 int                 m,
                 int                 n,
                 int                 batch,
                 hipblasltDatatype_t type,
                 int                 block_count)
{
    int64_t size = m * n * batch * type2Size(type);
    CHECK_HIP_ERROR(hipHostMalloc(data, size));
    initData(type, *data, m, n, m, m * n, batch);
    for(int block = 0; block < block_count; block++)
    {
        CHECK_HIP_ERROR(hipMalloc(&d_data[block], size));
        CHECK_HIP_ERROR(hipMemcpy(d_data[block], *data, size, hipMemcpyHostToDevice));
    }
}

void initNoCopy(void**              data,
                std::vector<void*>& d_data,
                int                 m,
                int                 n,
                int                 batch,
                hipblasltDatatype_t type,
                int                 block_count)
{
    int64_t size = m * n * batch * type2Size(type);
    CHECK_HIP_ERROR(hipHostMalloc(data, size));
    for(int block = 0; block < block_count; block++)
        CHECK_HIP_ERROR(hipMalloc(&d_data[block], size));
}

int main(int argc, char* argv[])
{
    bool     flush              = true;
    int32_t  gemm_count         = 1;
    uint32_t rotating           = 512 * 1024 * 1024;
    int32_t  cold_iters         = 100;
    int32_t  iters              = 1000;
    int64_t  max_workspace_size = 32 * 1024 * 1024;

    std::vector<int64_t> m, n, k, batch_count, solution_index;
    std::vector<hipblasltDatatype_t>    a_type, b_type, d_type;
    std::vector<bool>    enable_bias;
    if(argc >= 2)
    {
        for(int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];

            if((arg.at(0) == '-') || ((arg.at(0) == '-') && (arg.at(1) == '-')))
            {
                if(arg == "-i")
                {
                    iters = atoi(argv[++i]);
                }
                else if(arg == "-j")
                {
                    cold_iters = atoi(argv[++i]);
                }
                else if(arg == "--gemm_count")
                {
                    gemm_count = atoi(argv[++i]);
                }
                else if(arg == "--flush")
                {
                    flush = atoi(argv[++i]);
                }
                else if((arg == "-m") && (i + 1 < argc))
                {
                    m.push_back(atoi(argv[++i]));
                }
                else if((arg == "-n") && (i + 1 < argc))
                {
                    n.push_back(atoi(argv[++i]));
                }
                else if((arg == "-k") && (i + 1 < argc))
                {
                    k.push_back(atoi(argv[++i]));
                }
                else if((arg == "--batch_count") && (i + 1 < argc))
                {
                    batch_count.push_back(atoi(argv[++i]));
                }
                else if((arg == "--bias") && (i + 1 < argc))
                {
                    enable_bias.push_back(atoi(argv[++i]));
                }
                else if((arg == "--solution_index") && (i + 1 < argc))
                {
                    solution_index.push_back(atoi(argv[++i]));
                }
                else if((arg == "--a_type") && (i + 1 < argc))
                {
                    ++i;
                    if(strncmp(argv[i], "f32_r", 5) == 0)
                    {
                        a_type.push_back(HIPBLASLT_R_32F);
                    }
                    else if(strncmp(argv[i], "f16_r", 5) == 0)
                    {
                        a_type.push_back(HIPBLASLT_R_16F);
                    }
                    else if(strncmp(argv[i], "f8_r", 4) == 0)
                    {
                        a_type.push_back(HIPBLASLT_R_8F_E4M3);
                    }
                    else
                    {
                        std::cerr << "error with " << arg << std::endl;
                        std::cerr << "do not recognize value " << argv[i];
                        return EXIT_FAILURE;
                    }
                }
                else if((arg == "--b_type") && (i + 1 < argc))
                {
                    ++i;
                    if(strncmp(argv[i], "f32_r", 5) == 0)
                    {
                        b_type.push_back(HIPBLASLT_R_32F);
                    }
                    else if(strncmp(argv[i], "f16_r", 5) == 0)
                    {
                        b_type.push_back(HIPBLASLT_R_16F);
                    }
                    else if(strncmp(argv[i], "f8_r", 4) == 0)
                    {
                        b_type.push_back(HIPBLASLT_R_8F_E4M3);
                    }
                    else
                    {
                        std::cerr << "error with " << arg << std::endl;
                        std::cerr << "do not recognize value " << argv[i];
                        return EXIT_FAILURE;
                    }
                }
                else if((arg == "--d_type") && (i + 1 < argc))
                {
                    ++i;
                    if(strncmp(argv[i], "f32_r", 5) == 0)
                    {
                        d_type.push_back(HIPBLASLT_R_32F);
                    }
                    else if(strncmp(argv[i], "f16_r", 5) == 0)
                    {
                        d_type.push_back(HIPBLASLT_R_16F);
                    }
                    else if(strncmp(argv[i], "f8_r", 4) == 0)
                    {
                        d_type.push_back(HIPBLASLT_R_8F_E4M3);
                    }
                    else
                    {
                        std::cerr << "error with " << arg << std::endl;
                        std::cerr << "do not recognize value " << argv[i];
                        return EXIT_FAILURE;
                    }
                }
                else
                {
                    std::cerr << "error with " << arg << std::endl;
                    std::cerr << "do not recognize option" << std::endl << std::endl;
                    return EXIT_FAILURE;
                }
            }
            else
            {
                std::cerr << "error with " << arg << std::endl;
                std::cerr << "option must start with - or --" << std::endl << std::endl;
                return EXIT_FAILURE;
            }
        }
    }


    std::vector<Layer> layer(gemm_count);
    // Input data here, currently hardcoded
    // layer.setData
    hipblaslt_ext::GemmEpilogue epilogue;
    hipblaslt_ext::GemmEpilogue epilogue_bias;
    epilogue_bias.mode           = HIPBLASLT_EPILOGUE_BIAS;
    epilogue_bias.bias_data_type = HIPBLASLT_R_16F;
    for(size_t i = 0; i < layer.size(); i++)
    {
        layer[i].setData(m[i],
                        n[i],
                        k[i],
                        batch_count[i],
                        1.f,
                        0.f,
                        HIPBLAS_OP_N,
                        HIPBLAS_OP_N,
                        a_type[i],
                        b_type[i],
                        d_type[i],
                        d_type[i],
                        (a_type[i] != b_type[i])? HIPBLASLT_COMPUTE_F32_FAST_F16: HIPBLASLT_COMPUTE_F32,
                        (enable_bias[i])? epilogue_bias: epilogue);
        layer[i].algo_index = solution_index[i];
    }

    uint32_t totalRotatingSizeNeeded = 0;
    // Calculating block count
    int32_t max_iters   = max(cold_iters, iters);
    std::vector<int32_t> block_count;
    for(size_t i = 0; i < layer.size(); i++)
    {
        uint32_t size_c = 0, size_bias = 0;
        if(layer[i].beta != 0)
        {
            size_c = layer[i].m * layer[i].n * type2Size(layer[i].problem.type_c);
        }
        if(enable_bias[i])
        {
            size_bias = layer[i].m * type2Size(layer[i].problem.type_d);
        }
        totalRotatingSizeNeeded
            = layer[i].m * layer[i].n * type2Size(layer[i].problem.type_a)
               + layer[i].n * layer[i].k * type2Size(layer[i].problem.type_b) + size_c
               + layer[i].m * layer[i].n * type2Size(layer[i].problem.type_d) + size_bias;
        block_count.push_back(max(1, min(max_iters, ceil((float)rotating / totalRotatingSizeNeeded))));
    }

    if(rotating > 0)
    {
        std::cout << "Rotating buffer " << (float)rotating / (1024 * 1024) << " MiB. " << std::endl;
    }

    hipStream_t       stream;
    hipblasLtHandle_t handle;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));
    CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));

    hipblaslt_ext::GemmPreference gemmPref;
    gemmPref.setMaxWorkspaceBytes(max_workspace_size);
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResults;
    std::vector<std::vector<hipblaslt_ext::Gemm>> gemms(layer.size());
    for(size_t i = 0; i < layer.size(); i++)
    {
        Layer& l = layer[i];
        l.initBlock(block_count[i]);
        if(l.problem.op_a == HIPBLAS_OP_N)
            initAndCopy(&l.a, l.d_a, l.m, l.k, l.batch, l.problem.type_a, block_count[i]);
        else
            initAndCopy(&l.a, l.d_a, l.k, l.m, l.batch, l.problem.type_a, block_count[i]);
        if(l.problem.op_b == HIPBLAS_OP_N)
            initAndCopy(&l.b, l.d_b, l.k, l.n, l.batch, l.problem.type_b, block_count[i]);
        else
            initAndCopy(&l.b, l.d_b, l.n, l.k, l.batch, l.problem.type_b, block_count[i]);
        if(enable_bias[i])
            initAndCopy(&l.bias, l.d_bias, l.m, 1, l.batch, l.problem.type_b, block_count[i]);
        initNoCopy(&l.c, l.d_c, l.m, l.n, l.batch, l.problem.type_c, block_count[i]);
        initNoCopy(&l.d, l.d_d, l.m, l.n, l.batch, l.problem.type_d, block_count[i]);
        for(int b = 0; b < block_count[i]; b++)
        {
            gemms[i].push_back(hipblaslt_ext::Gemm(handle,
                                                   l.problem.op_a,
                                                   l.problem.op_b,
                                                   l.problem.type_a,
                                                   l.problem.type_b,
                                                   l.problem.type_c,
                                                   l.problem.type_d,
                                                   l.problem.type_compute));

            hipblaslt_ext::GemmInputs inputs;
            inputs.a     = l.d_a[b];
            inputs.b     = l.d_b[b];
            inputs.c     = l.d_c[b];
            inputs.d     = l.d_d[b];
            inputs.alpha = &l.alpha;
            inputs.beta  = &l.beta;
            inputs.bias  = l.d_bias[b];
            gemms[i][b].setProblem(l.m, l.n, l.k, l.batch, l.epilogue, inputs);
        }

        if(l.algo_index != -1)
        {
            std::vector<int>                              algoIndex = {l.algo_index};
            std::vector<hipblasLtMatmulHeuristicResult_t> tmpResult;
            CHECK_HIPBLASLT_ERROR(hipblaslt_ext::getAlgosFromIndex(handle, algoIndex, tmpResult));

            size_t workspaceSizeInBytes = 0;
            if(gemms[i][0].isAlgoSupported(tmpResult[0].algo, workspaceSizeInBytes)
               != HIPBLAS_STATUS_SUCCESS)
            {
                std::cout << "Error invalid index for GEMM " << i+1 << "." << std::endl;
                exit(1);
            }
            heuristicResults.push_back(tmpResult[0]);
            l.ws_size = workspaceSizeInBytes;
            CHECK_HIP_ERROR(hipMalloc(&l.ws, workspaceSizeInBytes));
        }
        else
        {
            std::vector<hipblasLtMatmulHeuristicResult_t> tmpResult;
            tmpResult.clear();
            gemms[i][0].algoGetHeuristic(1, gemmPref, tmpResult);
            size_t workspaceSizeInBytes = 0;
            if(tmpResult.size() == 0)
            {
                std::cout << "No Solution found GEMM" << i+1 << "." << std::endl;
                exit(1);
            }
            if(gemms[i][0].isAlgoSupported(tmpResult[0].algo, workspaceSizeInBytes)
               != HIPBLAS_STATUS_SUCCESS)
            {
                std::cout << "Error invalid index for GEMM " << i+1 << "." << std::endl;
                exit(1);
            }
            heuristicResults.push_back(tmpResult[0]);
            l.ws_size = workspaceSizeInBytes;
            CHECK_HIP_ERROR(hipMalloc(&l.ws, workspaceSizeInBytes));
        }
    }
    for(size_t gemmIdx = 0; gemmIdx < layer.size(); gemmIdx++)
    {
        for(int b = 0; b < block_count[gemmIdx]; b++)
        {
            CHECK_HIPBLASLT_ERROR(
                gemms[gemmIdx][b].initialize(heuristicResults[gemmIdx].algo, layer[gemmIdx].ws));
        }
    }

    for(size_t gemmIdx = 0; gemmIdx < layer.size(); gemmIdx++)
    {
        auto kernelname = gemms[gemmIdx][0].getSolutionName();
        std::cout << kernelname << std::endl;
    }

    hipEvent_t event_gpu_time_start, event_gpu_time_end;
    CHECK_HIP_ERROR(hipEventCreate(&event_gpu_time_start));
    CHECK_HIP_ERROR(hipEventCreate(&event_gpu_time_end));
    std::cout << "Run cold iter " << cold_iters << ", hot iters " << iters << std::endl
              << "Gemm count " << layer.size() << std::endl;
    for(int i = 0; i < cold_iters; i++)
    {
        for(size_t gemmIdx = 0; gemmIdx < layer.size(); gemmIdx++)
        {
            static_cast<void>(gemms[gemmIdx][i % block_count[gemmIdx]].run(stream));
        }
    }
    CHECK_HIP_ERROR(hipStreamSynchronize(stream));

    float gpu_time_ms, flush_time_ms;
    hipDeviceProp_t deviceProps;
    CHECK_HIP_ERROR(hipGetDeviceProperties(&deviceProps, 0));
    std::cout << std::endl << "Run Gemm1~5 x iters" << std::endl;
    CHECK_HIP_ERROR(hipEventRecord(event_gpu_time_start, stream));
    for(int i = 0; i < iters; i++)
    {
        for(size_t gemmIdx = 0; gemmIdx < layer.size(); gemmIdx++)
        {
            static_cast<void>(gemms[gemmIdx][i % block_count[gemmIdx]].run(stream));
        }
        if(flush)
            hipLaunchKernelGGL(flush_icache, dim3(deviceProps.multiProcessorCount*60), dim3(64), 0, stream);
    }
    CHECK_HIP_ERROR(hipEventRecord(event_gpu_time_end, stream));
    CHECK_HIP_ERROR(hipEventSynchronize(event_gpu_time_end));
    CHECK_HIP_ERROR(hipEventElapsedTime(&gpu_time_ms, event_gpu_time_start, event_gpu_time_end));
    if(flush)
    {
        CHECK_HIP_ERROR(hipEventRecord(event_gpu_time_start, stream));
        for(int i = 0; i < iters; i++)
            hipLaunchKernelGGL(flush_icache, dim3(deviceProps.multiProcessorCount*60), dim3(64), 0, stream);
        CHECK_HIP_ERROR(hipEventRecord(event_gpu_time_end, stream));
        CHECK_HIP_ERROR(hipEventSynchronize(event_gpu_time_end));
        CHECK_HIP_ERROR(hipEventElapsedTime(&flush_time_ms, event_gpu_time_start, event_gpu_time_end));
        gpu_time_ms -= flush_time_ms;
        std::cout << "flush time: " << flush_time_ms*1000 << " us" << std::endl;
        std::cout << "flush avg time: " << flush_time_ms*1000/iters << " us" << std::endl;
    }
    auto gpu_time_used = gpu_time_ms * 1000; // ms to us
    std::cout << "Total time: " << gpu_time_used << " us" << std::endl
              << "Total Avg time: " << gpu_time_used / iters << " us" << std::endl;

    std::cout << std::endl << "Run (Gemm x iters) x (1~5)" << std::endl;
    CHECK_HIP_ERROR(hipEventRecord(event_gpu_time_start, stream));
    for(size_t gemmIdx = 0; gemmIdx < layer.size(); gemmIdx++)
    {
        for(int i = 0; i < iters; i++)
        {
            static_cast<void>(gemms[gemmIdx][i % block_count[gemmIdx]].run(stream));
            if(flush)
                hipLaunchKernelGGL(flush_icache, dim3(deviceProps.multiProcessorCount*60), dim3(64), 0, stream);
        }
    }
    CHECK_HIP_ERROR(hipEventRecord(event_gpu_time_end, stream));
    CHECK_HIP_ERROR(hipEventSynchronize(event_gpu_time_end));
    CHECK_HIP_ERROR(hipEventElapsedTime(&gpu_time_ms, event_gpu_time_start, event_gpu_time_end));
    if(flush)
    {
        CHECK_HIP_ERROR(hipEventRecord(event_gpu_time_start, stream));
        for(size_t gemmIdx = 0; gemmIdx < layer.size(); gemmIdx++)
            for(int i = 0; i < iters; i++)
                hipLaunchKernelGGL(flush_icache, dim3(deviceProps.multiProcessorCount*60), dim3(64), 0, stream);
        CHECK_HIP_ERROR(hipEventRecord(event_gpu_time_end, stream));
        CHECK_HIP_ERROR(hipEventSynchronize(event_gpu_time_end));
        CHECK_HIP_ERROR(hipEventElapsedTime(&flush_time_ms, event_gpu_time_start, event_gpu_time_end));
        gpu_time_ms -= flush_time_ms;
        std::cout << "flush time: " << flush_time_ms*1000 << " us" << std::endl;
        std::cout << "flush avg time: " << flush_time_ms*1000/layer.size()/iters << " us" << std::endl;
    }
    gpu_time_used = gpu_time_ms * 1000; // ms to us
    std::cout << "Total time: " << gpu_time_used << " us" << std::endl
              << "Total Avg time: " << gpu_time_used / iters << " us" << std::endl;

    CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));
    CHECK_HIP_ERROR(hipEventDestroy(event_gpu_time_start));
    CHECK_HIP_ERROR(hipEventDestroy(event_gpu_time_end));
    CHECK_HIP_ERROR(hipStreamDestroy(stream));
    return 0;
}