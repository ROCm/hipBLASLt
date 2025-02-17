################################################################################
#
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

import itertools
import math
import os.path
import subprocess
import sys
import time
from collections import OrderedDict
from copy import deepcopy
from typing import Dict

from Tensile import __version__

from .Architectures import isaToGfx, SUPPORTED_ISA
from .Types import IsaVersion, IsaInfo
from .Utilities import locateExe, versionIsCompatible, print1, print2, printExit, printWarning, \
     verbosity
from .ValidParameters import validParameters

startTime = time.time()

globalParameters = OrderedDict()
globalParameters["MinimumRequiredVersion"] = (
    "0.0.0"  # which version of tensile is required to handle all the features required by this configuration file
)
globalParameters["PerformanceMetric"] = (
    "DeviceEfficiency"  # performance metric for benchmarking; one of {DeviceEfficiency, CUEfficiency}
)
globalParameters["ClientLogLevel"] = (
    3  # the log level of client. 0=Error, 1=Terse, 2=Verbose, 3=Debug (Aligned with ResultReporter.hpp)
)
# benchmarking
globalParameters["KernelTime"] = False  # T=use device timers, F=use host timers
globalParameters["PreciseKernelTime"] = (
    True  # T=On hip, use the timestamps for kernel start and stop rather than separate events.  Can provide more accurate kernel timing.  For GlobalSplitU kernels, recommend disabling this to provide consistent
)
# timing between GSU / non-GSU kernels
#globalParameters["CodeFromFiles"] = (
#    True  # if False byte arrays will be generated during Benchmarking phase as before
#)
globalParameters["PinClocks"] = False  # T=pin gpu clocks and fan, F=don't
globalParameters["HardwareMonitor"] = (
    True  # False: disable benchmarking client monitoring clocks using rocm-smi.
)
globalParameters["MinFlopsPerSync"] = (
    1  # Minimum number of flops per sync to increase stability for small problems
)
globalParameters["NumBenchmarks"] = (
    1  # how many benchmark data points to collect per problem/solution
)
globalParameters["SyncsPerBenchmark"] = (
    1  # how iterations of the stream synchronization for-loop to do per benchmark data point
)
globalParameters["EnqueuesPerSync"] = 1  # how many solution enqueues to perform per synchronization
globalParameters["MaxEnqueuesPerSync"] = -1  # max solution enqueues to perform per synchronization
globalParameters["SleepPercent"] = (
    300  # how long to sleep after every data point: 25 means 25% of solution time. Sleeping lets gpu cool down more.
)
globalParameters["SkipSlowSolutionRatio"] = 0.0  # Skip slow solution during warm-up stage.
# The valid range of this ratio is (0.0 ~ 1.0), and 0.0 means no skipping.
# Skip condition:  warm-up time * ratio > current best sol's warm-up time
# Suggestion:
#     Small size :  0.5
#     Medium size: 0.75
#     Large size :  0.9

# validation
globalParameters["NumElementsToValidate"] = (
    128  # number of elements to validate, 128 will be evenly spaced out (with prime number stride) across C tensor
)
globalParameters["NumElementsToValidateWinner"] = (
    0  # number of elements to validate in LibraryClient stage, the exact number to be validated is max(NumElementsToValidate,NumElementsToValidateWinner)
)
globalParameters["BoundsCheck"] = 0  # Bounds check
# 1: Perform bounds check to find out of bounds reads/writes.  NumElementsToValidate must be -1.
# 2: Perform bounds check by front side guard page
# 3: Perform bounds check by back side guard page
# 4: Perform bounds check by both back and front side guard page

globalParameters["ValidationMaxToPrint"] = 4  # maximum number of mismatches to print
globalParameters["ValidationPrintValids"] = False  # print matches too
# steps
globalParameters["ForceRedoBenchmarkProblems"] = (
    True  # if False and benchmarking already complete, then benchmarking will be skipped when tensile is re-run
)
globalParameters["ForceRedoLibraryLogic"] = (
    True  # if False and library logic already analyzed, then library logic will be skipped when tensile is re-run
)
globalParameters["ForceRedoLibraryClient"] = (
    True  # if False and library client already built, then building library client will be skipped when tensile is re-run
)

globalParameters["ShowProgressBar"] = (
    True  # if False and library client already built, then building library client will be skipped when tensile is re-run
)
globalParameters["SolutionSelectionAlg"] = (
    1  # algorithm to determine which solutions to keep. 0=removeLeastImportantSolutions, 1=keepWinnerSolutions (faster)
)
globalParameters["GenerateSourcesAndExit"] = False  # Exit after kernel source generation.
globalParameters["ExitOnFails"] = (
    1  # 1: Exit after benchmark run if failures detected.  2: Exit during benchmark run.
)
globalParameters["CpuThreads"] = (
    -1
)  # How many CPU threads to use for kernel generation.  0=no threading, -1 == nproc, N=min(nproc,N).  TODO - 0 sometimes fails with a kernel name error?  0 does not check error codes correctly
globalParameters["NumWarmups"] = 0

# even if error occurs in kernel generation (ie due to resource overflow),
# generate the kernel source anyway.  Tensile will also attempt to run
# the kernel.  Useful to examine and debug overflow errors.
# globalParameters["ForceGenerateKernel"] = 0

########################################
# less common
########################################
globalParameters["CMakeBuildType"] = (
    "Release"  # whether benchmark clients and library client should be release or debug
)
#globalParameters["PrintSolutionRejectionReason"] = (
#    False  # when a solution is marked as invalid, print why
#)
globalParameters["LogicFormat"] = "yaml"  # set library backend (yaml, or json)
globalParameters["LibraryFormat"] = "yaml"  # set library backend (yaml, or msgpack)

# True/False: CSV will/won't export WinnerGFlops, WinnerTimeUS, WinnerIdx, WinnerName.
# TODO - if no side-effect, we can set default to True. This can make analyzing "LibraryLogic" (AddFromCSV) faster
globalParameters["CSVExportWinner"] = False

# (When NumBenchmarks > 1). True: CSV will merge the rows of same Problem-ID. False: Each problem will write out "NumBenchmarks" rows
#   In old client - No effect, since in old client, CSV file only exports the last benchmark, somehow is not correct because the previous benchmarks are discarded
#   In new client - csv file exports "NumBenchmarks" rows for every problem. This also make the later analyzing slower
#                   Set this to "True" can merge the rows for same problem, hence can reduce the csv file size and speed up the later analyzing
# TODO - if side-effect, we can set default to True. This can make "getResults()" / "AddFromCSV()" faster
globalParameters["CSVMergeSameProblemID"] = False

# how to initialize tensor data
# serial-in-u will use a sequence that increments in the K dimension
# This is a predictable patterns that can be checked as the kernel runs to detect
# when the wrong data is being used.
# trig_float initializes with the sin function to have non-zero values in the mantissa
# and exponent. It cannot be used for int8 or int32. Need to use tensileAlmostEqual
# not tensileEqual for checking the result.
# See ClientWriter.py, the DataInitName(Enum) for a list of initialization patterns
#       - Problem-Independent: 0=0, 1=1, 2=2, 3=rand, 4=Nan, 5=Infinity, 6=BadInput(Nan), 7=BadOutput(Inf), 16=RandomNarrow,
#                              21=RandomNegPosLimited(-128~128 or -1~1), 23~26=Ind Cos/Sin Abs or Not
#       - Problem-dependent: 8=SerialID, 9=SerialDim0, 10=SerialDim1, 11=Identity, 12~15= Cos/Sin, Abs or Not
#       For A, B, C, D: All the InitMode (0~16) can be used
#       For Alpha/Beta: Only problem-independent init (0~7, 16, 23~26) can be used,
#                       problem-dependent init (8~15) would cause a exception (Invalid InitMode) in New Client
globalParameters["DataInitTypeAB"] = 3
globalParameters["DataInitTypeA"] = -1
globalParameters["DataInitTypeB"] = -1
globalParameters["DataInitTypeC"] = 3
globalParameters["DataInitTypeD"] = 0
globalParameters["DataInitTypeE"] = 0
globalParameters["DataInitTypeAlpha"] = 2
globalParameters["DataInitTypeBeta"] = 2
globalParameters["DataInitTypeBias"] = 3
globalParameters["DataInitTypeScaleA"] = 2
globalParameters["DataInitTypeScaleB"] = 2
globalParameters["DataInitTypeScaleC"] = 2
globalParameters["DataInitTypeScaleD"] = 2
globalParameters["DataInitTypeScaleAlphaVec"] = 3
globalParameters["DataInitValueActivationArgs"] = [2.0, 2.0]
globalParameters["CEqualD"] = (
    False  # Set to true if testing for the case where the pointer to C is the same as D.
)
# When this parameter is set to 0, the Tensile client will use srand(time(NULL)).
# If not 0 the Tensile client will use srand(seed).
globalParameters["DataInitSeed"] = 0
globalParameters["PruneSparseMode"] = (
    0  # Prune mode for Sparse Matrix: 0=random, 1=XX00, 2=X0X0, 3=0XX0, 4=X00X, 5=0X0X, 6=00XX
)

# build parameters
globalParameters["CMakeCXXFlags"] = ""  # pass flags to cmake
globalParameters["CMakeCFlags"] = ""  # pass flags to cmake
#globalParameters["DebugKernel"] = (
#    False  # assembly only, kernel gets buffer for debug "printing"; kernel writes data to memory, gets coppied to host and printed
#)
globalParameters["AsanBuild"] = False  # build with asan
#globalParameters["SaveTemps"] = False  # Generate intermediate results of hip kernels
globalParameters["KeepBuildTmp"] = False  # If true, do not remove artifacts in build_tmp

# debug for assembly
#globalParameters["SplitGSU"] = False  # Split GSU kernel into GSU1 and GSUM

# Tensor printing controls:
globalParameters["PrintTensorA"] = 0  # Print TensorA after initialization
globalParameters["PrintTensorB"] = 0  # Print TensorB after initialization
globalParameters["PrintTensorC"] = (
    0  # Print TensorC.  0x1=after init; 0x2=after copy-back; 0x3=both
)
globalParameters["PrintTensorD"] = (
    0  # Print TensorD.  0x1=after init; 0x2=after copy-back; 0x3=both
)
globalParameters["PrintTensorRef"] = (
    0  # Print reference tensor.  0x1=after init; 0x2=after copy-back; 0x3=both
)
globalParameters["PrintTensorBias"] = 0  # Print TensorBias after initialization
globalParameters["PrintTensorAmaxD"] = 0  # Print AmaxD after validation
#globalParameters["PrintIndexAssignments"] = 0  # Print the tensor index assignment info
globalParameters["PrintWinnersOnly"] = False  # Only print the solutions which become the fastest
globalParameters["PrintCodeCommands"] = (
    False  # print the commands used to generate the code objects (asm,link,hip-clang, etc)
)
globalParameters["DumpTensors"] = (
    False  # If True, dump tensors to binary files instead of printing them.
)

# If PrintMax* is greater than the dimension, the middle elements will be replaced with "..."


# device selection
globalParameters["Platform"] = 0  # select opencl platform

# shouldn't need to change
globalParameters["ClientExecutionLockPath"] = (
    None  # Path for a file lock to ensure only one client is executed at once.  filelock module is required if this is enabled.
)
globalParameters["LibraryUpdateFile"] = (
    ""  # File name for writing indices and speeds suitable for updating an existing library logic file
)
globalParameters["LibraryUpdateComment"] = (
    False  # Include solution name as a comment in the library update file
)

# internal, i.e., gets set during startup
globalParameters["AMDGPUArchPath"] = None  # /opt/rocm/llvm/bin/amdgpu-arch
globalParameters["ROCmAgentEnumeratorPath"] = None  # /opt/rocm/bin/rocm_agent_enumerator
globalParameters["ROCmSMIPath"] = None  # /opt/rocm/bin/rocm-smi
globalParameters["HipClangVersion"] = "0.0.0"

# default runtime is selected based on operating system, user can override
if os.name == "nt":
    globalParameters["RuntimeLanguage"] = "HIP"
else:
    globalParameters["RuntimeLanguage"] = "HIP"

globalParameters["CodeObjectVersion"] = "4"

# perf model
globalParameters["PerfModelL2ReadHits"] = 0.0
globalParameters["PerfModelL2WriteHits"] = 0.15
globalParameters["PerfModelL2ReadBwMul"] = 2
globalParameters["PerfModelReadEfficiency"] = 0.85

# limitation for training
globalParameters["MaxWorkspaceSize"] = 128 * 1024 * 1024  # max workspace for training (128MB)
#globalParameters["MinKForGSU"] = 32  # min K size to use GlobalSplitU algorithm (only for HPA now)

# control if a solution is run for a given problem
globalParameters["GranularityThreshold"] = 0.0

globalParameters["PristineOnGPU"] = (
    True  # use Pristine memory on Tensile trainning verification or not
)

globalParameters["SeparateArchitectures"] = (
    False  # write Tensile library metadata to separate files for each architecture
)

globalParameters["LazyLibraryLoading"] = (
    False  # Load library and code object files when needed instead of at startup
)

globalParameters["EnableMarker"] = False  # Enable Tensile markers

globalParameters["UseUserArgs"] = False

globalParameters["RotatingBufferSize"] = 0  # Size in MB
globalParameters["RotatingMode"] = (
    0  # Default is 0, allocated in order A0B0C0D0..ANBNCNDN. 1 is in order A0 pad B0 pad .... AN pad BN pad.
)
# Mode 0 requires memcpy everytime when the problem changes to reset the data, but mode 1 doesn't.

globalParameters["BuildIdKind"] = "sha1"
globalParameters["AsmDebug"] = (
    False  # Set to True to keep debug information for compiled code objects
)

globalParameters["UseEffLike"] = True  # Set to False to use winnerGFlops as the performance metric

# Save a copy - since pytest doesn't re-run this initialization code and YAML files can override global settings - odd things can happen
defaultGlobalParameters = deepcopy(globalParameters)


################################################################################
# Tensile internal parameters
################################################################################
# These parameters are not adjustable by the config yamls. They change with the
# generator versions
internalParameters = {
    # Each universal kernel will generate one PostGSU(GlobalSplitUPGR) kernel
    "GlobalSplitUPGR": 16
}

# These parameters are used in ContractionSolutions for user arguments support.
defaultInternalSupportParams = {
    "KernArgsVersion": 2,
    # Information about user input internal kernel argument support
    # Change this to False if the CustomKernel does not support.
    "SupportUserGSU": True,
    # This is a little different from GSU because GSU is already a parameter,
    # but WGM is not.
    "SupportCustomWGM": True,
    "SupportCustomStaggerU": True,
    # Use GG as G's backend
    "UseUniversalArgs": True,
}





# same parameter for all solution b/c depends only on compiler
defaultBenchmarkCommonParameters = [
    {"InnerUnroll": [1]},
    {"KernelLanguage": ["Assembly"]},
    {"LdsPadA": [-1]},
    {"LdsPadB": [-1]},
    {"LdsPadMetadata": [0]},
    {"LdsBlockSizePerPadA": [-1]},
    {"LdsBlockSizePerPadB": [-1]},
    {"LdsBlockSizePerPadMetadata": [0]},
    {"TransposeLDS": [-1]},
    {"MaxOccupancy": [40]},
    {"VectorWidthA": [-1]},
    {"VectorWidthB": [-1]},
    {"VectorStore": [-1]},
    {"StoreVectorWidth": [-1]},
    {"GlobalReadVectorWidthA": [-1]},
    {"GlobalReadVectorWidthB": [-1]},
    {"LocalReadVectorWidth": [-1]},
    {"WaveSeparateGlobalReadA": [0]},
    {"WaveSeparateGlobalReadB": [0]},
    {"WaveSeparateGlobalReadMetadata": [0]},
    {"UnrollLoopSwapGlobalReadOrder": [0]},
    {"PrefetchGlobalRead": [1]},
    {"PrefetchLocalRead": [1]},
    {"ClusterLocalRead": [1]},
    {"SuppressNoLoadLoop": [False]},
    {"ExpandPointerSwap": [True]},
    {"ScheduleGlobalRead": [1]},
    {"ScheduleLocalWrite": [1]},
    {"ScheduleIterAlg": [3]},
    {"GlobalReadPerMfma": [1]},
    {"LocalWritePerMfma": [-1]},
    {"InterleaveAlpha": [0]},
    {"OptNoLoadLoop": [1]},
    {"BufferLoad": [True]},
    {"BufferStore": [True]},
    {"DirectToVgprA": [False]},
    {"DirectToVgprB": [False]},
    {"DirectToVgprSparseMetadata": [False]},
    {"DirectToLds": [False]},
    {"UseSgprForGRO": [-1]},
    {"UseInstOffsetForGRO": [0]},
    {"AssertSummationElementMultiple": [1]},
    {"AssertFree0ElementMultiple": [1]},
    {"AssertFree1ElementMultiple": [1]},
    {"AssertAIGreaterThanEqual": [-1]},
    {"AssertAILessThanEqual": [-1]},
    {"StaggerU": [32]},  # recommend [0,32]
    {"StaggerUStride": [256]},  # recommend 256 for V10,V20
    {"StaggerUMapping": [0]},  # recommend [0,1]
    {"MagicDivAlg": [2]},
    {"GlobalSplitU": [1]},
    {"GlobalSplitUAlgorithm": ["MultipleBuffer"]},
    {"GlobalSplitUCoalesced": [False]},
    {"GlobalSplitUWorkGroupMappingRoundRobin": [False]},
    {"Use64bShadowLimit": [1]},
    {"NumLoadsCoalescedA": [1]},
    {"NumLoadsCoalescedB": [1]},
    {"WorkGroup": [[16, 16, 1]]},
    {"WorkGroupMapping": [8]},
    {"WorkGroupMappingXCC": [1]},
    {"WorkGroupMappingXCCGroup": [-1]},
    {"ThreadTile": [[4, 4]]},
    {"WavefrontSize": [64]},
    {"MatrixInstruction": [[]]},
    {"1LDSBuffer": [0]},
    {"DepthU": [-1]},
    {"NonTemporalE": [0]},
    {"NonTemporalD": [0]},
    {"NonTemporalC": [0]},
    {"NonTemporalA": [0]},
    {"NonTemporalB": [0]},
    {"NonTemporalWS": [0]},
    {"NonTemporalMetadata": [0]},
    {"NonTemporal": [-1]},
    {"PreloadKernArgs": [True]},
    {"CustomKernelName": [""]},
    {"NoReject": [False]},
    {"StoreRemapVectorWidth": [0]},
    {"SourceSwap": [False]},
    {"StorePriorityOpt": [False]},
    {"NumElementsPerBatchStore": [0]},
    {"StoreSyncOpt": [0]},
    {"GroupLoadStore": [False]},
    {"MIArchVgpr": [False]},
    {"StreamK": [0]},
    {"StreamKAtomic": [0]},
    {"StreamKXCCMapping": [0]},
    {"DebugStreamK": [0]},
    {"ActivationFused": [True]},
    {"ActivationFuncCall": [True]},
    {"ActivationAlt": [False]},
    {"WorkGroupReduction": [False]},
    {"ConvertAfterDS": [False]},
    {"ForceDisableShadowInit": [False]},
]

# dictionary of defaults comprised of default option for each parameter
defaultSolution = {}
for paramDict in defaultBenchmarkCommonParameters:
    for key, value in paramDict.items():
        defaultSolution[key] = value[0]
# other non-benchmark options for solutions

################################################################################
# Default Problem Type
################################################################################
defaultProblemType = {
    # =GEMM uses TransposeA,B parameters and makes the problem type more readable for users
    # =TensorContraction  requires specifying
    "OperationType": "GEMM",  # GEMM, TensorContraction, ConvolutionForward, ConvolutionBackwardData, ConvolutionBackwardWeights
    "DataType": 0,  # data types can specified by a variety of ways, such as "s", as listed in SolutionStructs.py::DataType
    "DataTypeA": 0,  # A data type can specified by a variety of ways, such as "s", as listed in SolutionStructs.py::DataType
    "DataTypeB": 0,  # B data type can specified by a variety of ways, such as "s", as listed in SolutionStructs.py::DataType
    "DataTypeE": 0,  # E data type can specified by a variety of ways, such as "s", as listed in SolutionStructs.py::DataType
    "DataTypeAmaxD": 0,  # AmaxD data type can specified by a variety of ways, such as "s", as listed in SolutionStructs.py::DataType
    "DestDataType": 0,  # destination data types can specified by a variety of ways, such as "s", as listed in SolutionStructs.py::DataType
    "ComputeDataType": 0,  # compute data types can specified by a variety of ways, such as "s", as listed in SolutionStructs.py::DataType
    "F32XdlMathOp": 0,  # reducing intermediate precision from f32 to a specific type, such as "x", as listed in SolutionStructs.py::DataType.
    # in:f32, intermediate:xf32, out:f32. f32 = xf32(f32) * xf32(f32)
    "UseBeta": True,  # =True use beta parameter (asm will check for B=0 and optimize the write for that), =False don't use beta parameter
    "UseE": False,  # =True use output E to output gemm results before activation
    "Gradient": False,  # =True set globalWriteElements to gradient mode
    "UseBias": 0,  # =1 support bias vector on M direction, =2 support bias vector on N direction, =3 support bias vector on both M,N direction
    "BiasSrc": "D",  # This parameter is used in gradient + bias. Support A, B, D.
    "UseScaleAB": "",  # Support "", "Scalar", and "Vector"
    "UseScaleCD": False,  # =True use scaleC, scaleD
    "UseScaleAlphaVec": 0,  # =1 support alpha vector on M direction, =2 support bias vector on N direction, =3 support alpha vector on both M,N direction
    "HighPrecisionAccumulate": False,  # f32 += f16*f16
    "SilentHighPrecisionAccumulate": False,  # Keep kernel names the same for HPA mode.  Useful for testing.
    "Sparse": 0,  # 4:2 Structured Sparse A Matrix, 0=Non Sparse, 1=Sparse Matrix A, 2=Sparse Matrix B
    "ComplexConjugateA": False,  # complex data should be conjugated for "C" transpose case
    "ComplexConjugateB": False,
    "StochasticRounding": False,  # By default, IEEE RNE rounding
    # for OperationType == GEMM
    "TransposeA": False,  # =True means transA="T" or "C", =False means transA = "N"
    "TransposeB": True,
    "Batched": False,  # add batching dimension
    "StridedBatched": True,  # use to select general batch or strided batch
    "GroupedGemm": False,  # use to select general batch or strided batch
    # for OperationType == TensorContraction
    # - Indices < NumIndicesC are Free or Batch indices and appear in C and D
    # - Indices which appear in both A and B, and are < NumIndicesC are batch.  A and B must have same number of batch indices.
    # - Indices which appear in both A and B, and are >= NumIndicesC are summation. A and B must have same number of summation indices.
    # - Indices which appear in A or B (but not both), are Free.  A and B may have different numbers of free indices.
    # - Summation loops are nested from smallest index number to largest, with the largest summation index as the 'unroll' loop.
    # - Memory order of C and D matrices is always 0..NumIndicesC-1, with 0 as the fastest-moving.
    #   - By choosing index assignments the output can be 'transposed'.  For example if IA=[1,2] IB=[0,2] then 0 is the coalesced dim for C/D.
    #   - Likewise batch index may be assigned between two free indices to control the output order, ie to write in CNHW format.
    #   - For example : IA=[0,1,3] IB=[2,1,3].  0,2 are free indices;  1 is batch.
    "IndexAssignmentsA": [0, 2],
    "IndexAssignmentsB": [1, 2],
    "NumIndicesC": 2,
    # use initial strides for AB.
    # This has some performance impact for the increased flexibility:
    #   - Additional strides will be passed into the kernel and will occupy SGPR registers
    #   - GlobalReadWidth must be 1 (since elements are not guaranteed to be adjacent in memory)
    "UseInitialStridesAB": False,
    # use initial strides for CD.
    # This has some performance impact for the increased flexibility:
    #   - Additional strides will be passed into the kernel and will occupy SGPR registers
    #   - Additional multiply on the store address path
    #   -VectorStore must be 0.  If VectorStore is -1, it will be silently set to 0 internally.
    "UseInitialStridesCD": False,
    "AllowNoFreeDims": False,  # allow A or B to specify no free dims
    # (if false, A and B must have at least one free dim)
    # (if true, A and B must have at least one free or batch dim)
    # SetConstStride* sets the specified stride in the problem.
    # These no longer generate predicates - see AssertStrideEqualA/B below
    # List of pairs of [index, constValue].
    # Index is a member of the global index assignments (not an offset into IndexAssignmentsA/B)
    # EX: SetConstStrideA: [ [3, 1], [2, 4] ] sets
    #     strideA for index3 to constant '1' and stride for index2 to constant '4'.
    "SetConstStrideA": [],
    "SetConstStrideB": [],
    "SetConstStrideBias": [],
    # Summation dimension indices
    "MirrorDimsA": [],
    "MirrorDimsB": [],
    "MirrorDimsMetadata": [],
    # for LD description
    "NumIndicesLD": 4,
    "IndexAssignmentsLD": [3, 4, 5, 6],  # order is LDD, LDC, LDA, LDB
    # Tile aware solution selection
    "TileAwareSelection": False,
    # Activation
    "Activation": False,
    "ActivationNoGuard": False,
    # AmaxD
    "OutputAmaxD": False,
    # For kernels putting arguments in workspaces instead of kernel arguments, they can choose to support user arguments input instead.
    "SupportUserArgs": True,
    "SwizzleTensorA": False,
    "SwizzleTensorB": False,
}

defaultProblemSizes = [{"Range": [[2880], 0, 0]}]
defaultBenchmarkFinalProblemSizes = [{"Range": [[64, 64, 64, 512], 0, 0]}]
defaultBatchedProblemSizes = [{"Range": [[2880], 0, [1], 0]}]
defaultBatchedBenchmarkFinalProblemSizes = [{"Range": [[64, 64, 64, 512], 0, [1], 0]}]


defaultSolutionSummationSizes = [32, 64, 96, 128, 256, 512, 1024, 2048, 4096, 8192, 16192]


################################################################################
# Default Analysis Parameters
################################################################################
defaultAnalysisParameters = {
    "ScheduleName": "Tensile",
    "DeviceNames": "fallback",
    "ArchitectureName": "gfx000",
    "LibraryType": "GridBased",
    "SolutionImportanceMin": 0.01,  # = 0.01=1% total time saved by keeping this solution
}


################################################################################
# Is query version compatible with current version
# a yaml file is compatible with tensile if
# tensile.major == yaml.major and tensile.minor.step > yaml.minor.step
################################################################################
def restoreDefaultGlobalParameters():
    """
    Restores `globalParameters` back to defaults.
    """
    global globalParameters
    global defaultGlobalParameters
    # Can't just assign globalParameters = deepcopy(defaultGlobalParameters) because that would
    # result in dangling references, specifically in Tensile.Tensile().
    globalParameters.clear()
    for key, value in deepcopy(defaultGlobalParameters).items():
        globalParameters[key] = value


# hopefully the isaInfoMap keys only contain isas we plan to build and not all
def printCapabilitiesTable(isaInfoMap: Dict[str, IsaInfo]): 
    """
    Prints a capability table for the given parameters and ISA information map.

    Args:
        supportedIsas: The ISAs to show in the table.
        isaInfoMap: The ISA information map containing assembler and architecture capabilities.
    """

    def printTable(rows):
        rows = [[str(cell) for cell in row] for row in rows]
        colWidths = [max(len(cell) for cell in col) for col in zip(*rows)]

        for row in rows:
            print(" ".join(cell.ljust(width) for cell, width in zip(row, colWidths)))

    def capRow(isaInfoMap, cap, capType):
        return [cap] + [
            "1" if cap in getattr(info, capType) and getattr(info, capType)[cap] else "-"
            for info in isaInfoMap.values()
        ]

    gfxs = list(map(isaToGfx, isaInfoMap.keys()))
    headerRow = ["Capability"] + gfxs

    allAsmCaps = sorted(
        set(itertools.chain(*[info.asmCaps for info in isaInfoMap.values()])),
        key=lambda k: (k.split("_")[-1], k),
    )
    asmCapRows = [capRow(isaInfoMap, cap, "asmCaps") for cap in allAsmCaps]

    allArchCaps = sorted(set(itertools.chain(*[info.archCaps for info in isaInfoMap.values()])))
    archCapRows = [capRow(isaInfoMap, cap, "archCaps") for cap in allArchCaps]

    printTable([headerRow] + asmCapRows + archCapRows)


def assignGlobalParameters(config, isaInfoMap: Dict[IsaVersion, IsaInfo], cxxCompiler=None):
    """
    Assign Global Parameters
    Each global parameter has a default parameter, and the user
    can override them, those overridings happen here
    """

    global globalParameters, SUPPORTED_ISA

    # Minimum Required Version
    if "MinimumRequiredVersion" in config:
        if not versionIsCompatible(config["MinimumRequiredVersion"]):
            printExit(
                "Config file requires version=%s is not compatible with current Tensile version=%s"
                % (config["MinimumRequiredVersion"], __version__)
            )

    # User-specified global parameters
    print2("GlobalParameters:")
    for key in globalParameters:
        defaultValue = globalParameters[key]
        if key in config:
            configValue = config[key]
            if configValue == defaultValue:
                print2(" %24s: %8s (same)" % (key, configValue))
            else:
                print2(" %24s: %8s (overriden)" % (key, configValue))
        else:
            print2(" %24s: %8s (unspecified)" % (key, defaultValue))

    globalParameters["ROCmPath"] = "/opt/rocm"
    if "ROCM_PATH" in os.environ:
        globalParameters["ROCmPath"] = os.environ.get("ROCM_PATH")
    if "TENSILE_ROCM_PATH" in os.environ:
        globalParameters["ROCmPath"] = os.environ.get("TENSILE_ROCM_PATH")
    if os.name == "nt" and "HIP_DIR" in os.environ:
        globalParameters["ROCmPath"] = os.environ.get("HIP_DIR")  # windows has no ROCM
    globalParameters["CmakeCxxCompiler"] = None
    if "CMAKE_CXX_COMPILER" in os.environ:
        globalParameters["CmakeCxxCompiler"] = os.environ.get("CMAKE_CXX_COMPILER")
    if "CMAKE_C_COMPILER" in os.environ:
        globalParameters["CmakeCCompiler"] = os.environ.get("CMAKE_C_COMPILER")

    globalParameters["ROCmBinPath"] = os.path.join(globalParameters["ROCmPath"], "bin")

    # ROCm AMD GPU Arch Path
    # ROCm Agent Enumerator Path
    if os.name == "nt":
        globalParameters["AMDGPUArchPath"] = locateExe(
            globalParameters["ROCmBinPath"], "hipinfo.exe"
        )
        globalParameters["ROCmAgentEnumeratorPath"] = locateExe(
            globalParameters["ROCmBinPath"], "hipinfo.exe"
        )
    else:
        globalParameters["AMDGPUArchPath"] = locateExe(
            globalParameters["ROCmPath"], "llvm/bin/amdgpu-arch"
        )
        globalParameters["ROCmAgentEnumeratorPath"] = locateExe(
            globalParameters["ROCmBinPath"], "rocm_agent_enumerator"
        )

    globalParameters["ROCmSMIPath"] = locateExe(globalParameters["ROCmBinPath"], "rocm-smi")
    globalParameters["ROCmLdPath"] = locateExe(
        os.path.join(globalParameters["ROCmPath"], "llvm/bin"), "ld.lld"
    )

    if "AMDGPUArchPath" in config:
        globalParameters["AMDGPUArchPath"] = config["AMDGPUArchPath"]

    if "AsanBuild" in config:
        globalParameters["AsanBuild"] = config["AsanBuild"]

    if "KeepBuildTmp" in config:
        globalParameters["KeepBuildTmp"] = config["KeepBuildTmp"]

    if "CodeObjectVersion" in config:
        globalParameters["CodeObjectVersion"] = config["CodeObjectVersion"]

    if verbosity >= 1:
        printCapabilitiesTable(isaInfoMap)

    isaList = list(isaInfoMap.keys())
    validParameters["ISA"] = [IsaVersion(0, 0, 0), *isaList]

    # For ubuntu platforms, call dpkg to grep the version of hip-clang.  This check is platform specific, and in the future
    # additional support for yum, dnf zypper may need to be added.  On these other platforms, the default version of
    # '0.0.0' will persist

    # Due to platform.linux_distribution() being deprecated, just try to run dpkg regardless.
    # The alternative would be to install the `distro` package.
    # See https://docs.python.org/3.7/library/platform.html#platform.linux_distribution

    # The following try except block computes the hipcc version
    # TODO: hipcc is deprecated, this block should be removed.
    try:
        compiler = "hipcc"
        output = subprocess.run(
            [compiler, "--version"], check=True, stdout=subprocess.PIPE
        ).stdout.decode()

        for line in output.split("\n"):
            if "HIP version" in line:
                globalParameters["HipClangVersion"] = line.split()[2]
                print1("# Found hipcc version " + globalParameters["HipClangVersion"])

    except (subprocess.CalledProcessError, OSError) as e:
        printWarning("Error: {} running {} {} ".format("hipcc", "--version", e))

    # The following keys may be present in the config, but are not (or no longer) global parameters.
    ignoreKeys = [
        "Architecture",
        "ShortNames",
        "PrintLevel",
        "Device",
        "UseCompression",
        "CxxCompiler",
        "CCompiler",
        "OffloadBundler",
        "Assembler",
        "LogicPath",
        "LogicFilter",
        "OutputPath",
        "Experimental",
        "GenSolTable",
    ]
    for key in config:
        if key in ignoreKeys:
            continue
        value = config[key]
        if key not in globalParameters:
            printWarning("Global parameter %s = %s unrecognised." % (key, value))
        globalParameters[key] = value


def setupRestoreClocks():
    import atexit

    def restoreClocks():
        if globalParameters["PinClocks"]:
            rsmi = globalParameters["ROCmSMIPath"]
            subprocess.call([rsmi, "-d", "0", "--resetclocks"])
            subprocess.call([rsmi, "-d", "0", "--setfan", "50"])

    atexit.register(restoreClocks)


setupRestoreClocks()


def assignParameterWithDefault(destinationDictionary, key, sourceDictionary, defaultDictionary):
    if key in sourceDictionary:
        destinationDictionary[key] = deepcopy(sourceDictionary[key])
    else:
        destinationDictionary[key] = deepcopy(defaultDictionary[key])
