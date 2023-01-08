/*******************************************************************************
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include "utility.hpp"
#include <sys/types.h>
#include <unistd.h>
LoggerSingleton LoggerSingleton::gInstance;
std::ostream*   get_logger_os()
{
    LoggerSingleton& s = LoggerSingleton::getInstance();
    return s.log_os;
}

uint32_t get_logger_layer_mode()
{
    LoggerSingleton& s = LoggerSingleton::getInstance();
    return s.env_layer_mode;
}

std::string prefix(const char* layer, const char* caller)
{
    time_t now   = time(0);
    tm*    local = localtime(&now);

    std::string             format = "[%d-%02d-%02d %02d:%02d:%02d][HIPBLASLT][%lu][%s][%s]\0";
    std::unique_ptr<char[]> buf(new char[255]);
    std::sprintf(buf.get(),
                 format.c_str(),
                 1900 + local->tm_year,
                 1 + local->tm_mon,
                 local->tm_mday,
                 local->tm_hour,
                 local->tm_min,
                 local->tm_sec,
                 getpid(),
                 layer,
                 caller);
    return std::string(buf.get());
}

const char* hipblasDatatype_to_string(hipblasDatatype_t type)
{
    switch(type)
    {
    case HIPBLAS_R_16F:
        return "R_16F";
    case HIPBLAS_R_16B:
        return "R_16BF";
    case HIPBLAS_R_32F:
        return "R_32F";
    default:
        return "Invalid";
    }
}

const char* rocblaslt_compute_type_to_string(rocblaslt_compute_type type)
{
    switch(type)
    {
    case rocblaslt_compute_f32:
        return "COMPUTE_32F";
    default:
        return "Invalid";
    }
}

const char* rocblaslt_matrix_layout_attributes_to_string(rocblaslt_matrix_layout_attribute_ type)
{
    switch(type)
    {
    case ROCBLASLT_MATRIX_LAYOUT_BATCH_COUNT:
        return "MATRIX_LAYOUT_BATCH_COUNT";
    case ROCBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET:
        return "MATRIX_LAYOUT_STRIDED_BATCH_OFFSET";
    default:
        return "Invalid";
    }
}

const char* rocblaslt_matmul_desc_attributes_to_string(rocblaslt_matmul_desc_attributes type)
{
    switch(type)
    {
    case ROCBLASLT_MATMUL_DESC_TRANSA:
        return "MATMUL_DESC_TRANSA";
    case ROCBLASLT_MATMUL_DESC_TRANSB:
        return "MATMUL_DESC_TRANSB";
    case ROCBLASLT_MATMUL_DESC_EPILOGUE:
        return "MATMUL_DESC_EPILOGUE";
    case ROCBLASLT_MATMUL_DESC_BIAS_POINTER:
        return "MATMUL_DESC_BIAS_POINTER";
    case ROCBLASLT_MATMUL_DESC_BIAS_DATA_TYPE:
        return "MATMUL_DESC_BIAS_DATA_TYPE";
    case ROCBLASLT_MATMUL_DESC_D_SCALE_POINTER:
        return "MATMUL_DESC_D_SCALE_POINTER";
    default:
        return "Invalid";
    }
}
const char* hipblasOperation_to_string(hipblasOperation_t op)
{
    switch(op)
    {
    case HIPBLAS_OP_N:
        return "OP_N";
    case HIPBLAS_OP_T:
        return "OP_T";
    case HIPBLAS_OP_C:
        return "OP_C";
    default:
        return "Invalid";
    }
}

const char* rocblaslt_layer_mode2string(rocblaslt_layer_mode layer_mode)
{
    switch(layer_mode)
    {
    case rocblaslt_layer_mode_none:
        return "None";
    case rocblaslt_layer_mode_log_error:
        return "Error";
    case rocblaslt_layer_mode_log_trace:
        return "Trace";
    case rocblaslt_layer_mode_log_hints:
        return "Hints";
    case rocblaslt_layer_mode_log_info:
        return "Info";
    case rocblaslt_layer_mode_log_api:
        return "Api";
    default:
        return "Invalid";
    }
}

const char* rocblaslt_epilogue_to_string(rocblaslt_epilogue epilogue)
{
    switch(epilogue)
    {
    case ROCBLASLT_EPILOGUE_DEFAULT:
        return "EPILOGUE_DEFAULT";
    case ROCBLASLT_EPILOGUE_RELU:
        return "EPILOGUE_RELU";
    case ROCBLASLT_EPILOGUE_BIAS:
        return "EPILOGUE_BIAS";
    case ROCBLASLT_EPILOGUE_RELU_BIAS:
        return "EPILOGUE_RELU_BIAS";
    case ROCBLASLT_EPILOGUE_GELU:
        return "EPILOGUE_GELU";
    case ROCBLASLT_EPILOGUE_GELU_BIAS:
        return "EPILOGUE_GELU_BIAS";
    default:
        return "Invalid epilogue";
    }
}

std::string rocblaslt_matrix_layout_to_string(rocblaslt_matrix_layout mat)
{
    std::string             format = mat->batch_count <= 1
                                         ? "[type=%s rows=%d cols=%d ld=%d]\0"
                                         : "[type=%s rows=%d cols=%d ld=%d batch_count=%d batch_stride=%d]\0";
    std::unique_ptr<char[]> buf(new char[255]);
    if(mat->batch_count <= 1)
        std::sprintf(buf.get(),
                     format.c_str(),
                     hipblasDatatype_to_string(mat->type),
                     mat->m,
                     mat->n,
                     mat->ld);
    else
        std::sprintf(buf.get(),
                     format.c_str(),
                     hipblasDatatype_to_string(mat->type),
                     mat->m,
                     mat->n,
                     mat->ld,
                     mat->batch_count,
                     mat->batch_stride);
    return std::string(buf.get());
}
std::string rocblaslt_matmul_desc_to_string(rocblaslt_matmul_desc matmul_desc)
{
    std::string format = matmul_desc->bias_type == static_cast<hipblasDatatype_t>(-1)
                             ? "[computeType=%s scaleType=%s transA=%s transB=%s "
                               "epilogue=%s biasPointer=0x%x]\0"
                             : "[computeType=%s scaleType=%s transA=%s transB=%s "
                               "epilogue=%s biasPointer=0x%x biasType=%s]\0";

    std::unique_ptr<char[]> buf(new char[255]);

    if(matmul_desc->bias_type == static_cast<hipblasDatatype_t>(-1))
        std::sprintf(buf.get(),
                     format.c_str(),
                     rocblaslt_compute_type_to_string(matmul_desc->compute_type),
                     hipblasDatatype_to_string(matmul_desc->scale_type),
                     hipblasOperation_to_string(matmul_desc->op_A),
                     hipblasOperation_to_string(matmul_desc->op_B),
                     rocblaslt_epilogue_to_string(matmul_desc->epilogue),
                     matmul_desc->bias);
    else
        std::sprintf(buf.get(),
                     format.c_str(),
                     rocblaslt_compute_type_to_string(matmul_desc->compute_type),
                     hipblasDatatype_to_string(matmul_desc->scale_type),
                     hipblasOperation_to_string(matmul_desc->op_A),
                     hipblasOperation_to_string(matmul_desc->op_B),
                     rocblaslt_epilogue_to_string(matmul_desc->epilogue),
                     matmul_desc->bias,
                     hipblasDatatype_to_string(matmul_desc->bias_type));
    return std::string(buf.get());
}
