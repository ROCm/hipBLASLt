*****************************
hipBLASLt Datatypes Reference
*****************************

hipblasLtEpilogue_t
-------------------
.. doxygenenum:: hipblasLtEpilogue_t

hipblasLtHandle_t
-------------------
.. doxygentypedef:: hipblasLtHandle_t

hipblasLtMatmulAlgo_t
---------------------
.. doxygenstruct:: hipblasLtMatmulAlgo_t

hipblasLtMatmulDesc_t
---------------------
.. doxygentypedef:: hipblasLtMatmulDesc_t

hipblasLtMatmulDescAttributes_t
-------------------------------
.. doxygenenum:: hipblasLtMatmulDescAttributes_t

hipblasLtMatmulHeuristicResult_t
--------------------------------
.. doxygenstruct:: hipblasLtMatmulHeuristicResult_t

hipblasLtMatmulPreference_t
----------------------------
.. doxygentypedef:: hipblasLtMatmulPreference_t

hipblasLtMatmulPreferenceAttributes_t
-------------------------------------
.. doxygenenum:: hipblasLtMatmulPreferenceAttributes_t

hipblasLtMatrixLayout_t
-----------------------
.. doxygentypedef:: hipblasLtMatrixLayout_t

hipblasLtMatrixLayoutAttribute_t
--------------------------------
.. doxygenenum:: hipblasLtMatrixLayoutAttribute_t

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
HIPBLASLT_COMPUTE_F32   HIP_R_32F           HIP_R_32F     HIP_R_32F
HIPBLASLT_COMPUTE_F32   HIP_R_32F           HIP_R_16F     HIP_R_16F
HIPBLASLT_COMPUTE_F32   HIP_R_32F           HIP_R_16BF    HIP_R_16BF
======================= =================== ============= ==============
