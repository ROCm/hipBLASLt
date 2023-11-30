***********************
hipBLASLt API Reference
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

======================= =================== ============= ==============
computeType             scaleType/Bias Type Atype/Btype   Ctype/Dtype
======================= =================== ============= ==============
HIPBLAS_COMPUTE_32F     HIP_R_32F           HIP_R_32F     HIP_R_32F
HIPBLAS_COMPUTE_32F     HIP_R_32F           HIP_R_16F     HIP_R_16F
HIPBLAS_COMPUTE_32F     HIP_R_32F           HIP_R_16F     HIP_R_32F
HIPBLAS_COMPUTE_32F     HIP_R_32F           HIP_R_16BF    HIP_R_16BF
======================= =================== ============= ==============
