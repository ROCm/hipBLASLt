============================
Clients
============================

There is one client executables that can be used with hipBLASLt.

1. example_hipblaslt_preference

This client can be built by following the instructions in the `Building and Installing hipBLASLt github page <https://github.com/ROCmSoftwarePlatform/hipBLAS/blob/develop/docs/source/install.rst>`_ . After building the hipBLASLt clients, they can be found in the directory ``hipBLASLt/build/release/clients/staging``.

The next section will cover a brief explanation and the usage of each hipBLASLt client.

example_hipblaslt_preference
============================
example_hipblaslt_preference is used to measure performance and to verify the correctness of hipBLASLt functions.

It has a command line interface. For more information:

.. code-block:: bash

   ./example_hipblaslt_preference --help

For example, to measure performance of fp32 gemm:

.. code-block:: bash

   ./example_hipblaslt_preference --datatype fp32 --trans_a N --trans_b N -m 4096 -n 4096 -k 4096 --alpha 1 --beta 1

On a mi210 machine the above command outputs a performance of 13509 Gflops below:

.. code-block:: bash

   transAB, M, N, K, lda, ldb, ldc, stride_a, stride_b, stride_c, batch_count, alpha, beta, bias, activationType, ms, tflops
   NN, 4096, 4096, 4096, 4096, 4096, 4096, 16777216, 16777216, 16777216, 1, 1, 1, 0, none, 10.173825, 13.509074

The user can copy and change the above command. For example, to change the datatype to IEEE-16 bit and the size to 2048:

.. code-block:: bash

   ./example_hipblaslt_preference --datatype fp16 --trans_a N --trans_b N -m 2048 -n 2048 -k 2048 --alpha 1 --beta 1

Note that example_hipblaslt_preference also has the flag ``-V`` for correctness checks.
