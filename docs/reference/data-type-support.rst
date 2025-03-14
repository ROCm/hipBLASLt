.. meta::
   :description: hipBLASLt library data type support
   :keywords: hipBLASLt, ROCm, data type support

.. _data-type-support:

******************************************
Data type support
******************************************

This topic lists the supported data types for the hipBLASLt GEMM operation, 
which is performed by :ref:`hipblasltmatmul`. Here is the equation:

.. math::

 D = Activation(alpha \cdot op(A) \cdot op(B) + beta \cdot op(C) + bias)

Where :math:`op( )` refers to in-place operations, such as transpose and
non-transpose, and :math:`alpha` and :math:`beta` are scalars.

.. note:: 
  
  The ``__hip_fp8_e4m3_fnuz`` and ``__hip_fp8_e5m2_fnuz`` data types are only
  supported on the gfx94x platform.

For more information about data type support for the other ROCm libraries, see 
:doc:`Data types and precision support page <rocm:reference/precision-support>`. 

.. list-table:: Supported data types
  :header-rows: 1
  :name: supported-data-types

  *
    - A data type
    - B data type
    - C data type
    - D (Output) data type
    - Compute(Scale) data type

  *
    - ``float``
    - ``float``
    - ``float``
    - ``float``
    - ``float``

  *
    - ``half``
    - ``half``
    - ``half``
    - ``half``
    - ``float``

  *
    - ``half``
    - ``half``
    - ``half``
    - ``float``
    - ``float``


  *
    - ``bfloat16``
    - ``bfloat16``
    - ``bfloat16``
    - ``bfloat16``
    - ``float``

  *
    - ``__hip_fp8_e4m3_fnuz``
    - ``__hip_fp8_e4m3_fnuz``
    - ``float``
    - ``float``
    - ``float``

  *
    - ``__hip_fp8_e4m3_fnuz``
    - ``__hip_fp8_e4m3_fnuz``
    - ``half``
    - ``half``
    - ``float``

  *
    - ``__hip_fp8_e4m3_fnuz``
    - ``__hip_fp8_e4m3_fnuz``
    - ``bfloat16``
    - ``bfloat16``
    - ``float``

  *
    - ``__hip_fp8_e4m3_fnuz``
    - ``__hip_fp8_e4m3_fnuz``
    - ``__hip_fp8_e4m3_fnuz``
    - ``__hip_fp8_e4m3_fnuz``
    - ``float``

  *
    - ``__hip_fp8_e4m3_fnuz``
    - ``__hip_fp8_e4m3_fnuz``
    - ``__hip_fp8_e5m2_fnuz``
    - ``__hip_fp8_e5m2_fnuz``
    - ``float``

  *
    - ``__hip_fp8_e5m2_fnuz``
    - ``__hip_fp8_e5m2_fnuz``
    - ``float``
    - ``float``
    - ``float``

  *
    - ``__hip_fp8_e5m2_fnuz``
    - ``__hip_fp8_e5m2_fnuz``
    - ``half``
    - ``half``
    - ``float``

  *
    - ``__hip_fp8_e5m2_fnuz``
    - ``__hip_fp8_e5m2_fnuz``
    - ``bfloat16``
    - ``bfloat16``
    - ``float``

  *
    - ``__hip_fp8_e5m2_fnuz``
    - ``__hip_fp8_e5m2_fnuz``
    - ``__hip_fp8_e4m3_fnuz``
    - ``__hip_fp8_e4m3_fnuz``
    - ``float``

  *
    - ``__hip_fp8_e5m2_fnuz``
    - ``__hip_fp8_e5m2_fnuz``
    - ``__hip_fp8_e5m2_fnuz``
    - ``__hip_fp8_e5m2_fnuz``
    - ``float``

  *
    - ``int8_t``
    - ``int8_t``
    - ``int8_t``
    - ``int8_t``
    - ``int32_t``
