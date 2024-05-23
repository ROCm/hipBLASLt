.. meta::
   :description: A library that provides GEMM operations with flexible APIs and extends functionalities beyond the traditional BLAS library
   :keywords: hipBLASLt, ROCm, library, API, tool

.. _api-reference:

***********************
hipBLASLt API reference
***********************

hipblasLtCreate()
------------------------------------------
.. doxygenfunction:: hipblasLtCreate

hipblasLtDestroy()
------------------------------------------
.. doxygenfunction:: hipblasLtDestroy

hipblasLtMatrixLayoutCreate()
------------------------------------------
.. doxygenfunction:: hipblasLtMatrixLayoutCreate

hipblasLtMatrixLayoutDestroy()
------------------------------------------
.. doxygenfunction:: hipblasLtMatrixLayoutDestroy

hipblasLtMatrixLayoutSetAttribute()
------------------------------------------
.. doxygenfunction:: hipblasLtMatrixLayoutSetAttribute

hipblasLtMatrixLayoutGetAttribute()
------------------------------------------
.. doxygenfunction:: hipblasLtMatrixLayoutGetAttribute

hipblasLtMatmulDescCreate()
------------------------------------------
.. doxygenfunction:: hipblasLtMatmulDescCreate

hipblasLtMatmulDescDestroy()
------------------------------------------
.. doxygenfunction:: hipblasLtMatmulDescDestroy

hipblasLtMatmulDescSetAttribute()
------------------------------------------
.. doxygenfunction:: hipblasLtMatmulDescSetAttribute

hipblasLtMatmulDescGetAttribute()
------------------------------------------
.. doxygenfunction:: hipblasLtMatmulDescGetAttribute

hipblasLtMatmulPreferenceCreate()
------------------------------------------
.. doxygenfunction:: hipblasLtMatmulPreferenceCreate

hipblasLtMatmulPreferenceDestroy()
------------------------------------------
.. doxygenfunction:: hipblasLtMatmulPreferenceDestroy

hipblasLtMatmulPreferenceSetAttribute()
------------------------------------------
.. doxygenfunction:: hipblasLtMatmulPreferenceSetAttribute

hipblasLtMatmulPreferenceGetAttribute()
------------------------------------------
.. doxygenfunction:: hipblasLtMatmulPreferenceGetAttribute

.. _hipblasltmatmulalgogetheuristic:

hipblasLtMatmulAlgoGetHeuristic()
------------------------------------------
.. doxygenfunction:: hipblasLtMatmulAlgoGetHeuristic

.. _hipblasltmatmul:

hipblasLtMatmul()
------------------------------------------
.. doxygenfunction:: hipblasLtMatmul

Datatypes Supported:

hipblasLtMatmul supports the following computeType, scaleType, Atype/Btype, Ctype/Dtype and Bias Type:

======================= =================== =============== ===============
computeType             scaleType/Bias Type Atype/Btype     Ctype/Dtype
======================= =================== =============== ===============
HIPBLASLT_COMPUTE_F32   HIPBLASLT_R_32F     HIPBLASLT_R_32F HIPBLASLT_R_32F
HIPBLASLT_COMPUTE_F32   HIPBLASLT_R_32F     HIPBLASLT_R_16F HIPBLASLT_R_16F
HIPBLASLT_COMPUTE_F32   HIPBLASLT_R_32F     HIPBLASLT_R_16F HIPBLASLT_R_32F
HIPBLASLT_COMPUTE_F32   HIPBLASLT_R_32F     HIPBLASLT_R_16B HIPBLASLT_R_16B
======================= =================== =============== ===============

hipblasLtMatrixTransformDescCreate()
------------------------------------------
.. doxygenfunction:: hipblasLtMatrixTransformDescCreate

hipblasLtMatrixTransformDescDestroy()
------------------------------------------
.. doxygenfunction:: hipblasLtMatrixTransformDescDestroy

hipblasLtMatrixTransformDescSetAttribute()
------------------------------------------
.. doxygenfunction:: hipblasLtMatrixTransformDescSetAttribute

hipblasLtMatrixTransformDescGetAttribute()
------------------------------------------
.. doxygenfunction:: hipblasLtMatrixTransformDescGetAttribute

hipblasLtMatrixTransform()
------------------------------------------
.. doxygenfunction:: hipblasLtMatrixTransform

hipblasLtMatrixTransform supports the following Atype/Btype/Ctype and scaleType:

======================= ===================
Atype/Btype/Ctype       scaleType
======================= ===================
HIP_R_32F               HIP_R_32F
HIP_R_16F               HIP_R_32F/HIP_R_16F
HIP_R_16BF              HIP_R_32F
HIP_R_8I                HIP_R_32F
HIP_R_32I               HIP_R_32F
======================= ===================
