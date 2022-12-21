################################################################################
#
# Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

from . import __version__
from . import Parallel
from .TensileInstructions import getGfxName, TensileInstructions
from collections import OrderedDict
from copy import deepcopy


import math
import os.path
import subprocess
import sys
import time

startTime = time.time()

ParallelMap = Parallel.ParallelMap

# print level
# 0 - user wants no printing
# 1 - user wants limited prints
# 2 - user wants full prints

################################################################################
# Global Parameters
################################################################################
globalParameters = OrderedDict()
workingDirectoryStack = []

########################################
# common
########################################
globalParameters["MinimumRequiredVersion"] = "0.0.0" # which version of tensile is required to handle all the features required by this configuration file
globalParameters["PerformanceMetric"] = "DeviceEfficiency" # performance metric for benchmarking; one of {DeviceEfficiency, CUEfficiency}
globalParameters["PrintLevel"] = 1                # how much info to print in generator. 0=none, 1=standard, 2=verbose
globalParameters["ClientLogLevel"] = 3            # the log level of client. 0=Error, 1=Terse, 2=Verbose, 3=Debug (Aligned with ResultReporter.hpp)
# benchmarking
globalParameters["KernelTime"] = False            # T=use device timers, F=use host timers
globalParameters["PreciseKernelTime"] = True      # T=On hip, use the timestamps for kernel start and stop rather than separate events.  Can provide more accurate kernel timing.  For GlobalSplitU kernels, recommend disabling this to provide consistent
# timing between GSU / non-GSU kernels
globalParameters["CodeFromFiles"] = True          # if False byte arrays will be generated during Benchmarking phase as before
globalParameters["SortProblems"] = False          # sort problems by size; else use order in YAML file
globalParameters["PinClocks"] = False             # T=pin gpu clocks and fan, F=don't
globalParameters["HardwareMonitor"] = True        # False: disable benchmarking client monitoring clocks using rocm-smi.
globalParameters["MinFlopsPerSync"] = 1           # Minimum number of flops per sync to increase stability for small problems
globalParameters["NumBenchmarks"] = 1             # how many benchmark data points to collect per problem/solution
globalParameters["SyncsPerBenchmark"] = 1         # how iterations of the stream synchronization for-loop to do per benchmark data point
globalParameters["EnqueuesPerSync"] = 1           # how many solution enqueues to perform per synchronization
globalParameters["MaxEnqueuesPerSync"] = -1       # max solution enqueues to perform per synchronization
globalParameters["SleepPercent"] = 300            # how long to sleep after every data point: 25 means 25% of solution time. Sleeping lets gpu cool down more.
# cProfile
globalParameters["Profiler"] = 0                  # Enable profiler. 0=off, 1=cProfile. This will set CpuThreads to 1.
# validation
globalParameters["NumElementsToValidate"] = 128   # number of elements to validate, 128 will be evenly spaced out (with prime number stride) across C tensor
globalParameters["BoundsCheck"] = 0   # Bounds check
#1: Perform bounds check to find out of bounds reads/writes.  NumElementsToValidate must be -1.
#2: Perform bounds check by front side guard page
#3: Perform bounds check by back side guard page
#4: Perform bounds check by both back and front side guard page

globalParameters["ValidationMaxToPrint"] = 4      # maximum number of mismatches to print
globalParameters["ValidationPrintValids"] = False # print matches too
# steps
globalParameters["ForceRedoBenchmarkProblems"] = True # if False and benchmarking already complete, then benchmarking will be skipped when tensile is re-run
globalParameters["ForceRedoLibraryLogic"] = True      # if False and library logic already analyzed, then library logic will be skipped when tensile is re-run
globalParameters["ForceRedoLibraryClient"] = True     # if False and library client already built, then building library client will be skipped when tensile is re-run

globalParameters["ShowProgressBar"] = True     # if False and library client already built, then building library client will be skipped when tensile is re-run
globalParameters["SolutionSelectionAlg"] = 1          # algorithm to detetermine which solutions to keep. 0=removeLeastImportantSolutions, 1=keepWinnerSolutions (faster)
globalParameters["ExpandRanges"] = True          # expand ranges into exact configs before writing logic file.  False ignores ranges.
globalParameters["ExitAfterKernelGen"] = False     # Exit after generating kernels
globalParameters["GenerateSourcesAndExit"] = False # Exit after kernel source generation.
globalParameters["ShowProgressBar"] = True     # if False and library client already built, then building library client will be skipped when tensile is re-run
globalParameters["WavefrontWidth"] = 64     # if False and library client already built, then building library client will be skipped when tensile is re-run
globalParameters["ExitOnFails"] = 1     # 1: Exit after benchmark run if failures detected.  2: Exit during benchmark run.
globalParameters["CpuThreads"] = -1  # How many CPU threads to use for kernel generation.  0=no threading, -1 == nproc, N=min(nproc,N).  TODO - 0 sometimes fails with a kernel name error?  0 does not check error codes correctly
# FROM MERGE
#globalParameters["CpuThreads"] = -4         # How many CPU threads to use for kernel generation.  0=no threading, <0 == nproc*abs(CpuThreads), N=min(nproc,N)

# even if error occurs in kernel generation (ie due to resource overflow),
# generate the kernel source anyway.  Tensile will also attempt to run
# the kernel.  Useful to examine and debug overflow errors.
globalParameters["ForceGenerateKernel"] = 0

########################################
# optimization knob controls
########################################

globalParameters["UnrollLoopEfficiencyEnable"] = False   # if True split(S) MAC&LDS in each unroll iteration into n smaller groups..

########################################
# less common
########################################
globalParameters["CMakeBuildType"] = "Release"            # whether benchmark clients and library client should be release or debug
globalParameters["PrintSolutionRejectionReason"] = False  # when a solution is marked as invalid, print why
globalParameters["LibraryFormat"] = "yaml"                # set library backend (either yaml or msgpack)
globalParameters["EmbedLibrary"] = None                   # whether library should be embedded or not

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
#       - Problem-Independent: 0=0, 1=1, 2=2, 3=rand, 4=Nan, 5=Infinity, 6=BadInput(Nan), 7=BadOutput(Inf), 16=RandomNarrow
#       - Problem-dependent: 8=SerialID, 9=SerialDim0, 10=SerialDim1, 11=Identity, 12~15= Cos/Sin, Abs or Not
#       For A, B, C, D: All the InitMode (0~16) can be used
#       For Alpha/Beta: Only problem-independent init (0~7, 16) can be used,
#                       problem-dependent init (8~15) would cause a exception (Invalid InitMode) in New Client
globalParameters["DataInitTypeAB"] = 3
globalParameters["DataInitTypeA"] = -1
globalParameters["DataInitTypeB"] = -1
globalParameters["DataInitTypeC"]  = 3
globalParameters["DataInitTypeD"]  = 0
globalParameters["DataInitTypeAlpha"] = 2
globalParameters["DataInitTypeBeta"] = 2
globalParameters["DataInitTypeBias"] = 3
globalParameters["DataInitTypeScaleD"] = 3
globalParameters["DataInitValueActivationArgs"] = [2.0, 2.0]
globalParameters["CEqualD"] = False               # Set to true if testing for the case where the pointer to C is the same as D.
globalParameters["BufferOffsetA"] = 0             # data offset of buffer A
globalParameters["BufferOffsetB"] = 0             # data offset of buffer B
globalParameters["BufferOffsetC"] = 0             # data offset of buffer C
globalParameters["BufferOffsetD"] = 0             # data offset of buffer D

# build parameters
globalParameters["CMakeCXXFlags"] = ""            # pass flags to cmake
globalParameters["CMakeCFlags"] = ""              # pass flags to cmake
globalParameters["DebugKernel"] = False           # assembly only, kernel gets buffer for debug "printing"; kernel writes data to memory, gets coppied to host and printed
globalParameters["LibraryPrintDebug"] = False     # solutions will print enqueue info when enqueueing a kernel

# debug for assembly
globalParameters["EnableAsserts"] = False         # Enable assembly debug assert
globalParameters["EnableDebugA"] = False          # Enable / Disable CheckValue1A
globalParameters["EnableDebugB"] = False          # Enable / Disable CheckValue1B
globalParameters["EnableDebugC"] = False          # Enable / Disable CheckValueC
globalParameters["ExpectedValueC"] = 16.0         # Expected C Value when CheckValueC, debug for Alpha*A*B
globalParameters["ForceCExpectedValue"] = False   # Force C to "DebugExpectedValueC", debug for global write

# Tensor printing controls:
globalParameters["PrintTensorA"] = 0          # Print TensorA after initialization
globalParameters["PrintTensorB"] = 0          # Print TensorB after initialization
globalParameters["PrintTensorC"] = 0          # Print TensorC.  0x1=after init; 0x2=after copy-back; 0x3=both
globalParameters["PrintTensorD"] = 0          # Print TensorD.  0x1=after init; 0x2=after copy-back; 0x3=both
globalParameters["PrintTensorRef"] = 0          # Print reference tensor.  0x1=after init; 0x2=after copy-back; 0x3=both
globalParameters["PrintIndexAssignments"] = 0      # Print the tensor index assignment info
globalParameters["PrintWinnersOnly"] = False      # Only print the solutions which become the fastest
globalParameters["PrintCodeCommands"] = False  # print the commands used to generate the code objects (asm,link,hip-clang, etc)
globalParameters["DumpTensors"] = False        # If True, dump tensors to binary files instead of printing them.

# If PrintMax* is greater than the dimension, the middle elements will be repaced with "..."


# device selection
globalParameters["Platform"] = 0                  # select opencl platform
globalParameters["Device"] = 0                    # select hip device or opencl device within platform

# shouldn't need to change
globalParameters["DeviceLDS"] = 65536             # LDS bytes per CU, for computing occupancy
globalParameters["MaxLDS"] = 65536                # max LDS a kernel should attempt to use
globalParameters["MaxDepthU"] = 256               # max DepthU value to allow
globalParameters["ShortNames"] = False            # on windows kernel names can get too long; =True will convert solution/kernel names to serial ids
globalParameters["MergeFiles"] = True             # F=store every solution and kernel in separate file; T=store all solutions in single file
globalParameters["NumMergedFiles"] = 1            # The number of files that kernels should be split between when merging

globalParameters["MaxFileName"] = 64              # If a file name would be longer than this, shorten it with a hash.
globalParameters["SupportedISA"] = [(8,0,3), (9,0,0), (9,0,6), (9,0,8), (9,0,10), (10,1,0), (10,1,1), (10,1,2), (10,3,0), (11,0,0), (11,0,1), (11,0,2)] # assembly kernels writer supports these architectures

globalParameters["GenerateManifestAndExit"] = False               # Output manifest file with list of expected library objects and exit
globalParameters["NewClient"] = 2                                 # Old client deprecated: NewClient must be set to 2.
globalParameters["ClientBuildPath"] = "0_Build"                   # subdirectory for host code build directory
globalParameters["BenchmarkProblemsPath"] = "1_BenchmarkProblems" # subdirectory for benchmarking phases
globalParameters["BenchmarkDataPath"] = "2_BenchmarkData"         # subdirectory for storing final benchmarking data
globalParameters["LibraryLogicPath"] = "3_LibraryLogic"           # subdirectory for library logic produced by analysis
globalParameters["LibraryClientPath"] = "4_LibraryClient"         # subdirectory for building example library client
globalParameters["ClientExecutionLockPath"] = None                # Path for a file lock to ensure only one client is executed at once.  filelock module is required if this is enabled.
globalParameters["LibraryUpdateFile"] = ""                        # File name for writing indices and speeds suitable for updating an existing library logic file
globalParameters["LibraryUpdateComment"] = False                  # Include solution name as a comment in the library update file

# internal, i.e., gets set during startup
globalParameters["CurrentISA"] = (0,0,0)
globalParameters["ROCmAgentEnumeratorPath"] = None      # /opt/rocm/bin/rocm_agent_enumerator
globalParameters["ROCmSMIPath"] = None                  # /opt/rocm/bin/rocm-smi
globalParameters["AssemblerPath"] = None                # /opt/rocm/hip/bin/hipcc
globalParameters["WorkingPath"] = os.getcwd()           # path where tensile called from
globalParameters["IndexChars"] =  "IJKLMNOPQRSTUVWXYZ"  # which characters to use for C[ij]=Sum[k] A[ik]*B[jk]
globalParameters["ScriptPath"] = os.path.dirname(os.path.realpath(__file__))            # path to Tensile/Tensile.py
globalParameters["SourcePath"] = os.path.join(globalParameters["ScriptPath"], "Source") # path to Tensile/Source/
globalParameters["HipClangVersion"] = "0.0.0"

# default runtime is selected based on operating system, user can override
if os.name == "nt":
  globalParameters["RuntimeLanguage"] = "HIP" #"OCL"
else:
  globalParameters["RuntimeLanguage"] = "HIP"

globalParameters["CodeObjectVersion"] = "V3"
globalParameters["CxxCompiler"] = "hipcc"
globalParameters["Architecture"] = "all"

# might be deprecated
globalParameters["EnableHalf"] = False
globalParameters["ClientArgs"] = ""
globalParameters["PackageLibrary"] = False

# perf model
globalParameters["PerfModelL2ReadHits"] = 0.0
globalParameters["PerfModelL2WriteHits"] = 0.15
globalParameters["PerfModelL2ReadBwMul"] = 2
globalParameters["PerfModelReadEfficiency"] = 0.85

# limitation for training
globalParameters["MaxWorkspaceSize"] = 32 * 1024 * 1024 # max workspace for training (32M)
globalParameters["MinKForGSU"] = 256 # min K size to use GlobalSplitU algorithm (only for HPA now)

# control if a solution is run for a given problem
globalParameters["GranularityThreshold"] = 0.0

# directory where custom kernels are located
globalParameters["CustomKernelDirectory"] = os.path.join(os.path.dirname(os.path.realpath(__file__)), "CustomKernels")

globalParameters["PristineOnGPU"] = True # use Pristine memory on Tensile trainning verification or not

globalParameters["SeparateArchitectures"] = False # write Tensile library metadata to separate files for each architecture

globalParameters["LazyLibraryLoading"] = False # Load library and code object files when needed instead of at startup

# Save a copy - since pytest doesn't re-run this initialization code and YAML files can override global settings - odd things can happen
defaultGlobalParameters = deepcopy(globalParameters)

# Translate GPU targets to filter filenames in Tensile_LOGIC directory
architectureMap = {
  'all':'_','gfx000':'none', 'gfx803':'r9nano', 'gfx900':'vega10',
  'gfx906':'vega20', 'gfx906:xnack+':'vega20', 'gfx906:xnack-':'vega20',
  'gfx908':'arcturus','gfx908:xnack+':'arcturus', 'gfx908:xnack-':'arcturus',
  'gfx90a':'aldebaran', 'gfx90a:xnack+':'aldebaran', 'gfx90a:xnack-':'aldebaran',
  'gfx1010':'navi10', 'gfx1011':'navi12', 'gfx1012':'navi14', 'gfx1030':'navi21',
  'gfx1100':'navi31', 'gfx1101':'navi32', 'gfx1102':'navi33'
}

def getArchitectureName(gfxName):
  if gfxName in architectureMap:
    return architectureMap[gfxName]
  else:
    for archKey in architectureMap:
      if gfxName in archKey:
        return architectureMap[archKey]
    return None

################################################################################
# Enumerate Valid Solution Parameters
################################################################################
validWorkGroups = []
for numThreads in range(64, 1025, 64):
  for nsg in [ 1, 2, 4, 8, 16, 32, 64, 96, 128, 256 ]:
    for sg0 in range(1, numThreads//nsg+1):
      sg1 = numThreads//nsg//sg0
      if sg0*sg1*nsg == numThreads:
          workGroup = [sg0, sg1, nsg]
          validWorkGroups.append(workGroup)

validThreadTileSides = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] + list(range(20, 256, 4))
validThreadTiles = []
for i in validThreadTileSides:
  for j in validThreadTileSides:
    validThreadTiles.append([i, j])

validActivationFormats = ('NCHW', 'NHWC', 'CNHW', 'NCDHW', 'NDHWC', 'CNDHW')
validWeightFormats = ('KCYX', "KYXC", "CKYX", "CYXK",  'KCZYX', 'CKZYX', 'CZYXK')
validMacroTileSides = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 6, 12, 24, 48, 96, 192, 384, 768 ]
validMacroTiles = []
validISA = [(0,0,0)]
validISA.extend(globalParameters["SupportedISA"])
depthUs = list(range(-16, 0))
depthUs.extend(list(range(2,512+1,1)))
for i in validMacroTileSides:
  for j in validMacroTileSides:
    validMacroTiles.append([i, j])

validMFMA = {}
validMFMA["H"] = [[32,32,4,2], [32,32,8,1], [16,16,4,4], [16,16,16,1], [4,4,4,16]]
validMFMA["S"] = [[32,32,1,2], [32,32,2,1], [16,16,1,4], [16,16,4,1], [4,4,1,16]]
validMFMA["B"] = [[32,32,2,2], [32,32,4,1], [16,16,2,4], [16,16,8,1], [4,4,2,16]]
validMFMA["D"] = [[16,16,4,1], [4,4,4,4]]
validMFMA["B1k"] = [[32,32,4,2], [32,32,8,1], [16,16,4,4], [16,16,16,1], [4,4,4,16]]
validMFMA["C"] = validMFMA["S"]
validMFMA["Z"] = validMFMA["D"]
validMFMA["I8"] = [[32,32,4,2], [32,32,8,1], [16,16,4,4], [16,16,16,1], [4,4,4,16]]
validTT = 16
validMFMA["_format9"] = []

for MFMA in [validMFMA["H"], validMFMA["S"], validMFMA["B"], validMFMA["D"]]:
  for MI in MFMA:
    for bm in range(int(math.log(MI[3],2))+1):
      for tt0 in range(1,validTT+1):
        for tt1 in range(1,validTT+1):
          for wave_m in range (3):
            for wave_n in range(3):
              validMFMA["_format9"].append([MI[0],MI[1],MI[2],MI[3],2**bm,tt0,tt1,2**wave_m, 2**wave_n])
validMatrixInstructions = [[], [-1]] + validMFMA["H"] + validMFMA["S"] + validMFMA["B"] + validMFMA["D"] + validMFMA["B1k"]
validMatrixInstructions = validMatrixInstructions + validMFMA["_format9"]

# The supported typed GEMM, each entry is (Ti, To, Tc).
# DataType (Ti)        = The data-type of the input matrices: A/B
# DestDataType (To)    = The data-type of the output matrices: C/D
# ComputeDataType (Tc) = The data-type of computaiton: alpha/beta:
# Cinternal: basically should == ComputeDataType

# Align the supported GEMM type with rocBLAS: [A/B/ C/D/ alpha/beta]
#   (rocblas/library/include/internal/rocblas_functions.h)
# GEMM (HPA=F, the data type of input, output, and computation are all the same.)
#   - HGEMM: [H/H/ H/H/ H/H]
#   - SGEMM: [S/S/ S/S/ S/S]
#   - DGEMM: [D/D/ D/D/ D/D]
#   - CGEMM: [C/C/ C/C/ C/C]
#   - ZGEMM: [Z/Z/ Z/Z/ Z/Z]
# GEMM_Ex: (HPA=T, Computation is in a higher precision data-type)
#   - GEMM_EX (HHS): [H/H/ H/H/ S/S]
#   - GEMM_EX (HSS): [H/H/ S/S/ S/S]
#   - GEMM_EX (BBS): [B/B/ B/B/ S/S]
#   - GEMM_EX (BSS): [B/B/ S/S/ S/S]
#   - GEMM_EX (I8II): [I8/I8/ I/I/ I/I]
#   - GEMM_EX (4xi8II): [4xi8/4xi8/ I/I/ I/I], tensile packs 4 i8 to 4xi8 with some restrictions
# This is used in SolutionStruct.py::checkIfSupportedGEMMType()
validGEMMTypes = [ ('H','H','H'), ('S','S','S'), ('D','D','D'), ('C','C','C'), ('Z','Z','Z'), \
                   ('H','H','S'), ('H','S','S'), \
                   ('B','B','S'), ('B','S','S'), \
                   ('I8','I','I'), ('4xi8','I','I'), ('I8','I8','I'), \
                   ('I8','I','S'), ('I8','I8','S')]

# All HPA types are listed here (HPA=T). The name of the library logic files for these types is:
# *_TiToTc_BH*.yaml where Ti, Tc, and To are the data types of A/B, C/D, and computation, respectively.
# The name of the library logic files for non-HPA (HPA=F) types is: *_TiB*.yaml.
HPATypes = [ ('H','S','S'), ('H','H','S'), ('B','B','S'), ('B','S','S'), ('I8','I','I'), ('4xi8','I','I'), ('I8','I','S'), ('I8','I8','S')]

validParameters = {
    # original global read to lds is interlace, [w0,w1,w2,w3,w0,w1,w2,w3,w0,w1,w2,w3,w0,w1,w2,w3]
    # when WaveSeparateGlobalRead is enabled, LDS is divided to number of waves part.
    # each wave load a block memory to lds,     [w0,w0,w0,w0,w1,w1,w1,w1,w2,w2,w2,w2,w3,w3,w3,w3]
    # -1 is selected by logic, 0 disable, 1 enable.
    "WaveSeparateGlobalReadA":    [ 0, 1 ],
    "WaveSeparateGlobalReadB":    [ 0, 1 ],

    # PrefetchGlobalRead = 1:
    # Requires 2X LDS space, and VGPRs for buffering data on way into LDS
    #   prefetch / double-buffer reads from global memory -> vgprs -> lds.
    #
    # PrefetchGlobalRead = 2:
    # Do another prefetch while writing data from vgpr to lds.
    #   prefetch / double-buffer reads from global memory -> vgprs --> lds.
    #                                                              |-> prefetch reads
    "PrefetchGlobalRead":         [ 0, 1, 2 ],

    # number of iteration prefetch local reads from lds to VGPRs buffer = PLR % LoopIter
    # number of VGPRs buffer = min(PLR+1,LoopIters)
    # LoopIters = DepthU / LocalSplitU
    # (LoopIters /= MatrixInstruction_K)
    # ex. MT64x128x16_MI32x32x4x2_PLR1, we'll have 4 LoopIters, prefetch read 1 iteration, with 2 VGPRs buffer (2=min(1+1,4))
    #     before loop:       plr[0]
    #           loop: iter0:plr[1] MAC_r[0], iter1:plr[0] MAC_r[1], iter2:plr[1] MAC_r[0], iter3:plr[0] MAC_r[1]
    #   no load loop: iter0:plr[1] MAC_r[0], iter1:plr[0] MAC_r[1], iter2:plr[1] MAC_r[0], iter3:       MAC_r[1]
    #
    # ex. MT64x128x16_MI32x32x4x2_PLR3, we'll have 4 LoopIters, prefetch read 3 iteration, with 4 VGPRs buffer (4=min(3+1,4))
    #     before loop:       plr[0] plr[1] plr[2]
    #           loop: iter0:plr[3] MAC_r[0], iter1:plr[0] MAC_r[1], iter2:plr[1] MAC_r[2], iter3:plr[2] MAC_r[3]
    #   no load loop: iter0:plr[3] MAC_r[0], iter1:       MAC_r[1], iter2:       MAC_r[2], iter3:       MAC_r[3]
    #
    # ex. MT64x128x16_MI32x32x4x2_PLR5, we'll have 4 LoopIters, prefetch read 5%4=1 iteration, with 4 VGPRs buffer (4=min(5+1,4))
    #     before loop:       plr[0]
    #           loop: iter0:plr[1] MAC_r[0], iter1:plr[2] MAC_r[1], iter2:plr[3] MAC_r[2], iter3:plr[0] MAC_r[3]
    #   no load loop: iter0:plr[1] MAC_r[0], iter1:plr[2] MAC_r[1], iter2:plr[3] MAC_r[2], iter3:       MAC_r[3]
    #
    # ex. MT64x128x16_MI32x32x4x2_PLR5_LRVW8, we'll have 4 LoopIters, prefetch read 5%4=1 iteration, with 4 VGPRs buffer (4=min(5+1,4)) , each read read 2 iterations
    #     before loop:       plr[0:1]
    #           loop: iter0:plr[2:3] MAC_r[0], iter1: MAC_r[1], iter2: MAC_r[2], iter3:plr[0:1] MAC_r[3]
    #   no load loop: iter0:plr[2:3] MAC_r[0], iter1: MAC_r[1], iter2: MAC_r[2], iter3:         MAC_r[3]
    #
    # ex. MT64x128x16_MI32x32x4x2_PLR7, we'll have 4 LoopIters, prefetch read 7%4=3 iteration, with 4 VGPRs buffer (=min(7+1,4)) --> Exactly the same as PLR3
    #     before loop:       plr[0]
    #           loop: iter0:plr[1] MAC_r[0], iter1:plr[2] MAC_r[1], iter2:plr[3] MAC_r[2], iter3:plr[0] MAC_r[3]
    #   no load loop: iter0:plr[1] MAC_r[0], iter1:plr[2] MAC_r[1], iter2:plr[3] MAC_r[2], iter3:       MAC_r[3]
    "PrefetchLocalRead":          list(range(128+1)),

    # We use double LDS buffer when PrefetchGlobalRead.
    # While it reads data from LDS[0]/[1], it prefetch global data and writes to LDS[1]/[0]
    # If we can make sure all data are read from LDS to register before writing data to LDS, we can use 1 LDS buffer to save LDS memory.
    # this can help to generate Kernel that LDS usage originally exceed MaxLDS if using double LDS buffer,
    # or help to increase Occupancy.
    #     1 means: Force to use 1 LDS Buffer even with PrefetchGlobalRead
    #    -1 means: generator will use 1 LDS buffer only when LDS exceed MaxLDS
    # Use case:
    #    SIA2: 1LDSBuffer is set to 1 natively
    #    SIA3: 1LDSBuffer works only when PGR=True
    # TODO: optimize scheduling to support more cases.
    "1LDSBuffer": [-1 ,0, 1],

    # Split the unroll summation into multiple sections and combine the sections
    # GSU applies only to the unroll summation dimension
    "GlobalSplitU":               list(range(1, 1024+1)),

    # choose how to do GlobalSplitU
    # 1: use atomic operation to accumlate on one buffer
    # 2: each GSU group write to each own buffer and accumulate by another kernel
    "GlobalSplitUAlgorithm":      ["SingleBuffer", "MultipleBuffer"],

    # don't create a whole copy of the Unroll loop with loads removed - instead
    # use buffer limits to suppress global loads and ignore unnecessary ds_reads
    "SuppressNoLoadLoop":         [False, True],

    # For PrefetchGlobalRead=1, create a second copy of the unroll loop with
    # the LDS pointer swaps expanded into inline constants for LDS read and write instructions
    # This eliminates 4 vector XOR instructions used for pointer swap
    "ExpandPointerSwap":          [False, True],

    # Schedule global reads and global read incrementsinto LocalRead iterations
    # Can reduce pressure on local read instruction dispatch queue
    # 0=perform global reads at start of instruction loop
    # 1=schedule into the local read instruction iterations
    "ScheduleGlobalRead":         [0, 1],

    # Schedule local writes into LocalRead iterations.
    # Can reduce pressure on local read instruction dispatch queue
    "ScheduleLocalWrite":         [0, 1],

    # Scheduling algorithm to use for each iteration:
    # 0 = minimal/no scheduling.  Global Read and increments, followed by local reads,
    # followed by local writes, followed by MACs
    "ScheduleIterAlg":            [0, 1, 2, 3],

    # For MatrixInstruction and SIA3, number of GlobalReadInstruction between mfma
    # the purpose of this parameter is to control density of global read instruction scheduling
    # Scheduling global read back to back can have better memory efficiency
    # However, when full of vmem FIFO, it will block other instruction to be issued
    # Range from 0.01 to 32
    #         0.1 means 1 GR per 10 mfma
    #           5 means 5 GR per 1 mfma
    "GlobalReadPerMfma":       [ i/100 for i in range(1,3200)],
    #
    # For MatrixInstruction and SIA3, number of LocalWriteInstruction between mfma
    # the purpose of this parameter is to control density of local write instruction scheduling
    # In PGR1, we want to schedule local write more denser, so we can have more
    #          latency to hide global read
    # In PGR2, since LW is followed by GR, every LW has same whole loop latecy
    #          to hide global read. We want to schedule LW less denser, can
    #          avoid full of vmem FIFO.
    # Range from 0.01 to 32
    #         0.1 means 1 LW per 10 mfma
    #           5 means 5 LW per 1 mfma
    # -1 will derived an optimized value internally
    # -2 will derived an optimized value and override LWPM silently (debug only, not recommended)
    "LocalWritePerMfma":       [ i/100 for i in range(1,3200)] + [ -1 ],

    # Interleave alpha scale calculation with beta loads and address calcs - rather
    # than as a separate block of instructions
    "InterleaveAlpha":             [0, 1],

    # Create a copy of NoLoadLoop which interleaves the stores with the final mac
    # calculation and may perform other optimizations
    # 0 = no interleave
    # 1 = interleave one stores after required macs have completed execution
    # 2 = interleave two stores after required macs have completed execution
    "OptNoLoadLoop":               [0, 1, 2],

    "BufferLoad":                 [ False, True ],
    "BufferStore":                [ False, True ],

    # Attempt to load directly from global memory into Vgpr.
    # Assembly only
    "DirectToVgprA":              [ False, True ],
    "DirectToVgprB":              [ False, True ],

    # Attempt to load directly from global memory into LDS.
    # Assembly only
    # Requires BufferLoad, assembler support for lds modifier on buffer
    # loads (checked automatically), GlobalVectorWidth=1 (this is hw
    # requirement) and A/B must not require any transpose.
    # DirectToLds reduces load latency and eliminates the
    # G2L registers used to stage data.  Also replaces the
    # local write offset with an SGPR.
    # For an 8x8 TT with PrefetchGlobalRead=1 this can save 33 VGPRs.
    #    - Requirements for DirectToLds=1:
    #      GlobalLoadVectorWidth = 1/2/4
    #      TransposeLDS = 1 for TLU=0 case
    # DirectToLds support for x2/x4 (1st part of async_copy() support)
    "DirectToLds":                [ False, True ],

    # Load options:
    # (GRO = Global Read Offset)
    # BufferLoad=0:
    #  = Use flat instructions with 64 bit GRO for each load
    #    + supports sizes up to 2^64
    #    - uses many VGPR for addressing
    #    - uses execmask+compares for edge detection
    #    - generates extra LDS traffic (could convert flat->global load)
    # BufferLoad=1:
    #  = Use buffer load instructions with 32-bit offset
    #    + Less VGPRS (32b offset vs 64-bit) needed for addressing
    #    + Uses hardware buffer limit for edge detection
    #    - Limited range - the bot-right corner of macro-tile (plus padding=GRVW
    #        for shift-pointer, if ShiftPtr is required) must be within 2^32.
    #      ShiftPtrPad = MayShift ? GRWV*BPE : 0
    #      For TLU=1: Unroll*StrideA1 + ShiftPtrPad <= 2^32
    #      For TLU=0: MT*StrideA1 + ShiftPtrPad <= 2^32
    #      These conditions should be checked using Assert - TODO
    #  = UseSgprForGRO=1:
    #    + Attempt to use SGPR for Global Read Offsets.
    #    + Use one VGPR base GRO + many SGPR GRO rather than many VGPR GRO.
    #    + Each SGPR stores an offset from base GlobalReadOffset+0.
    #    - Requirements for UseSgprForGRO=1:
    #      - BufferLoad=1
    #      - Use appropriate Assert*ElementMultiple or GRVW=1 to eliminate need for ShifPtr
    #        (UseSgprForGRO does not support ShiftPtr since ShiftPtr needs to potentially shift GRO)
    #  = KernelWriterAssembly also supports 64-bit 2D buffer size (see use64bPbcLimit)
    #    - Requires 4 instructions to move scalar limit and a couple SGPR
    #    - Enabled by default.  If the overhead matters we can add asserts/YAML parm to specialize
    #  = UseInstOffsetForGRO=1:
    #    + Attempt to use Instruction offset for Global Read Offsets.
    #    + This feature avoid updating m0 for subsequent GRO(s) for directToLds feature
    #    - Requirements for UseInstOffsetForGRO=1:
    #      - BufferLoad=1
    #      - DirectToLds=1

    #  converting m0 update from LocalWriteAddrSGpr using  is usually win
    # -1 attempt to use a hueristic to determine when the tile size will use too many SGPR and fall back to VGPR
    "UseInstOffsetForGRO":              [ -1, 0, 1],


    # Converting VGPR GRO into SGPR GRO is usually a win
    # However, the mode may exhaust all available SGPR, in particular for large unroll
    # -1 attempt to use a hueristic to determine when the tile size will use too many SGPR and fall back to VGPR
    "UseSgprForGRO":              [ -1, 0, 1],

    # Use a 64-bit shadow limit register to allow buffers larger than 2^32 bytes
    "Use64bShadowLimit":   [ True, False],

    # Assertion properties
    # These provide information or assertions that the problem size meets certain requirements
    # for sizes or alignments.  The kernel generator can use this information to produce
    # a kernel which uses those assertions to produce a faster kernel.
    #
    # If modifying or adding Assertions also change ProblemProperties class in TensileTypes.h

    # Kernel generator will assume that the summation size is some multiple of the element size
    # and uses this to optimize the kernel.
    # This can result in more efficient kernels, but requires runtime checking to ensure the specified
    # summation value meets the requirements.
    # (Recommended AF1EM value is 8 for half, 4 for single, 2 for double)
    #
    # Optimizations enabled by AssertSummationElementMultiple>1:
    #  - If >=2 for half:
    #     - Tail loop loads can be vectorized 2X to use dword
    #     - Enables asm kernels on V20
    #     - Can use DirectToLds for both unroll and tail loops
    #  - Tail loop can be unrolled up to InnerUnroll amount if AssertSummationElementMultiple%InnerUnroll==0
    #
    # 1 indicates no assertion (since all sizes are multiples of 1)
    "AssertSummationElementMultiple": [1,2,4,8,16,32,64],

    # Kernel generator will assume that the FreeIndex[0] size is some multiple of the element size
    # and uses this to optimize the kernel.
    # FreeIndex[0] is usually letter "I"
    # (Recommended AF0EM value is 8 for half, 4 for single, 2 for double)
    #
    # Optimizations enabled by AssertFree0ElementMultiple>1:
    # Load optimizations:
    #  - For TLU=1 matrix, if AF1WM>=GLVW then can enable UseSgprForGRO
    #      - Reduces registers used for address calculations
    #      - Removes address shift/unshift code
    #    - UseSgprForGRO will only be enabled if all matrices meet assertion requirements.
    #
    # Store Optimizations:
    #  - Can vectorize stores in edge tiles.  Vector width can be up to AF0EM.
    #   (since C matrix is always coalesced in Free0 index direction and this assertion guarantees the index element multiple)
    #
    # 1 indicates no assertion (since all sizes are multiples of 1)
    "AssertFree0ElementMultiple" : [1,2,4,8],

    # Kernel generator will assume that the FreeIndex[1] size is some multiple of the element size
    # and uses this to optimize the kernel.
    # FreeIndex[1] is usually letter "J"
    # (Recommended AF1EM value is 8 for half, 4 for single, 2 for double)

    # Optimizations enabled by AssertFree1ElementMultiple>1:
    #  - See above AssertFree0ElementMultiple "Load optimizations"

    # 1 indicates no assertion (since all sizes are multiples of 1)
    "AssertFree1ElementMultiple" : [1,2,4,8],

    # Stagger the start summation position of the tiles.
    # Elements from the summation dimension are loaded at offsets rather than all starting at 0.
    # StaggerU is the max 'clicks' of StaggerUStride bytes where each wg starts ; see StaggerUMapping
    # for how the specific stagger for a given wg is determined.
    #
    # The tile assignment C are same as with StaggerOffset=0 ; the difference is the
    # order that the summation elements are added.
    # GRO will wrap back to the row start when the edge is reached.
    #
    # This can be effective for TLU=0 style matrices where the K dimension is a large power-of-2.
    # In this case the start of each row of the tile is separated by an exact power-of-2
    # which causes poor dram, cache, and tlb behavior.  V20 has 16 channels each 256 bytes wide.

    # StaggerU adjusts the start position in the summation (aka 'U') dimension
    # to avoid these conflicts.  Both A and B matrix start at the adjusted position.
    # If >0 specifies the offset in multiples of the macro-tile "unroll" dim
    #  - Higher values will spread traffic to more channels but provide less L2 re-use.
    #  - StaggerU and WorkGroupMapping interact and should be tuned together -
    #    The WGM controls how tiles are assigned in C matrix, while StaggerU controls where those
    #    tiles start reading their summation dim parms.
    #  - StaggerU requires BufferLoad==1 and is silently ignored if BufferLoad==0
    "StaggerU":              [0,2,4,8,16,32,64],

    # Stride in bytes for each staggeru 'click'.
    # 256 is recommended since this is the width of memory channel (on gfx803,gfx900,gf906) - so
    # each click will start in a new memory channel and spread traffic among the 16 available channels.
    # For example StaggerUStride=256 and StaggerU=8 will use 8 unique starting points
    # in summation dimension, each offset by 256-bytes - provided the tensor dims are large
    # enough to support this.
    # StaggerUStride will be internally increased so it is an integer multiple of DepthU*BpeAB.
    # (the implementation requires this - the unroll iteration accesses data in steps of
    # DepthU*BPE
    "StaggerUStride":               [16,32,64,128,256,512,1024],

    # How the tile assignment (wg0, wg1, wg2) controls the initial StaggerU offset:
    # 0: Use wg0
    # 1: Use wg1
    # 2: Use wg2
    # 3: Use wgSerial, wgSerial = wg0 + wg1 * nwg0 + wg2 * (nwg0 * nwg1)
    # 4: Debug mode, offset each tile max allowed StaggerU.  This just moves hotspot
    #    to a different bank since all workgroups still start at same point.
    "StaggerUMapping":       [0,1,2,3,4],


    # 0=don't use magic div (source only)
    # 1=magic div alg #1.  Slightly faster but limited range (if magic number is 2^32)
    # 2=magic div alg#2.  Slightly slower but handles all unsigned ints up to 2^32
    "MagicDivAlg":       [0,1,2],

    # For Block Mapping type:
    # 0   : Use hardware-assigned wg number with no remapping.
    # N   : WG block width.  "Wrap" to a new wg1 "row" assignment after N WGs assigned in that row.
    # Tensor C always mapped with first free coord as fastest moving
    # (Elements in this dimension are sequential in memory.
    #
    # For 2D nonbatched Matrix this means index order is I, then J
    # For 2D batched Matrix this means index order is I, then J, then K.
    #
    # Then for 2D case:
    #   - If drawn in row-major format, I is the width and J is the height.
    #   - WGM determines dimensions of the box used to assign tiles from C
    #   - WGM is the height of the box (in the J dimension)
    #   - Given WGM, the box width (in I dim) is determined by number of CUs
    #   - The box always moves across matrixC in the fastest-moving "I" dim, then
    #     wraps to next J.  TODO - might be useful to change this?
    #
    # Examples for 2D matrix:
    # WGM=8:  on CU64 machine this is a square box
    # WGM=1:  Short/Fat - this will cover maximum width in I dimension of C.  This matches hardware assigned mapping.
    # WGM=64: Tall/Skinny - this will cover maximum width in J dimention of C.
    #
    # Formula for wgSerial:
    # wgSerial = wg0 + (wg1 % WorkGroupMapping) * nwg0
    "WorkGroupMapping":           list(range(0,1024+1)),  # change a workgroup's id so that the all the workgroups on the gpu at a time are hitting L2 cache the best
    "MaxOccupancy":               list(range(1, 40+1)),       # wg / CU; if cache thrashing is hurting performance, this allocates extra lds to artificially limit occupancy
    "WorkGroup":                  validWorkGroups,      # ( wg0 x wg1 x LocalSplitU ) dimensions of the workgroup which will operate on a tile and share lds

    #ThreadTile: ( tt0 x tt1 ) dimensions of the C tile that each thread works on,
    # TT=4 and VW=4 means a thread will work on a tight 4x4 tile of C, where VW=1 means the tile will work on 16 spread out values
    # Generally, the VW determines the consecutive a WI will work on, then it will skip ahead SG0*VW elements to get to the next row of VGPR inputs
    "ThreadTile":                 validThreadTiles,
    "MacroTile":                  validMacroTiles,      # MT0 = wg0*tt0, MT1 = wg1*tt1

    "WavefrontSize":              [32, 64],

    # MatrixInstruction: (M x N x K x B)
    # XDLOPS tile definition, only valid for gfx908, gfx90a
    # MxNxKxB specifies matrix instruction variants
    #  MxNxB determines the shape of the C tile each instruction worked on
    #      K determines the unroll depth
    # If empty, do not use these instructions
    #
    # Alternative format: (M x N x K x B x MIBlockM x WaveTileM x WaveTileN x WaveM x WaveN)
    # (Note: MxN means M-by-N in the following comments)
    # MIBlockM determines how many blocks along M dimension for multi-block MI variants. Concrete examples:
    #  - MI 16x16x1x4 (4-block variant) with MIBlockM=4 -> (16x16)*(4x1)=64x16 tile per instruction executed
    #  - MI 32x32x1x2 (2-block variant) with MIBlockM=1 -> (32x32)*(1x2)=32x64 tile per instruction executed
    # WaveTileM/N are dimensions of the C tile each wave works on, and is close to the concept of ThreadTile in classic VALU kernels
    #  - WT 4x1 -> each wave executes 4x1 matrix instructions on the C tile of total area (4*MITileM)x(1*MITileN)
    # WaveM/N are dimensions of waves spawned for one workgroup where each wave consists of 64 threads
    #  - Wave2x2 -> a total of 4 waves in one workgroup of shape 2x2
    # Putting it all together:
    #  - [32, 32, 1, 2,  1,  4, 1,  2, 2]
    #     ^^^^^^^^^^^^   ^   ^^^^   ^^^^
    #      MatrixInst  BlkM   WT    Wave
    #  - means (32x64) per MI * (4x1) per wave * (2x2) per workgroup = (32*4*2)x(64*1*2) = 256x128 macro tile
    # Tensile will ignore the parameters ThreadTile and WorkGroup when the alternative format is used
    "MatrixInstruction":          validMatrixInstructions,

    # StoreRemap: Optimize MatrixInstruction store patterns to enhance performance.
    #             MI output data between each threads are along N dims.
    #             But global memory is along M dim continous.
    #             That mean global write between each threads are not continous.
    #             Therefore, store performance for MI instruction is poor.
    # How StoreRemap works in final store stage:
    #             1. Put all thread output data into LDS.
    #             2. All thread read data from LDS along M dims.
    #                (match global Memory continous direction)
    #             3. All thread write out data into global memory.
    # 0:   Disable StoreRemap (default)
    # 1~8: Enable StoreRemap and set the global write vector width
    # Suggest optimum value: fp32 = [2,4], fp16 or bf16 = [4,8] (dwordx2 and dowrdx4)
    # -1:  Use dwordx2 if support SRVW, or set SRVW to 0
    "StoreRemapVectorWidth":      [-1,0,1,2,4,8],

    # SourceSwap: Optimizes MatrixInstruction store pattern by swapping mfma input order.
    "SourceSwap":                 [False, True],

    # Following parameters are designed for store scheduling.
    # (store stands for load from C (with beta) and store to C/D)
    #
    # we want to hide store behind unroll loop
    #   1. if we can launch 2 WorkGroups per CU (occupancy >= 2, large M/N)
    #   2. if there are remaining global memory bandwidth in unroll loop (compute bound kernel)
    #
    # we can hide store behind the other WG's loop by lowering priority of store
    #   priority of loop is the same as priority of store
    #     WG0: ???????????????\__
    #         |<-- loop --->|<-- store -->|end
    #
    #     WG1: ___________________________/????????????\__
    #         |<--------- loop ------------------->|<-- store -->|end
    #
    #   priority of loop is higher than priority of store
    #     WG0: ???????\____________________
    #         |<-- loop --->|<------ store ----->|end
    #
    #     WG1: _____________/?????\__________________
    #         |<------- loop -------->|<----- store ---->|end
    "StorePriorityOpt":           [False, True],
    #
    # If we issue store in short period of time, kernel will become from compute bound to memory bound
    # 0 means issue instructions as many as possible if VGPR available
    "NumElementsPerBatchStore":   list(range(-1, 256)),
    #
    # add sync after per batch store in order to store contiguous elements
    # add sleep after per batch store in order to distribute store over whole loops
    # NOTE: this parameter is highly depends on size_k
    # 0 means no sync and sleep
    "StoreSyncOpt":               list(range(0, 256)),
    #
    # There are index or address calculation between global instructions.
    # issue global instruction b2b has better performance
    "GroupLoadStore":             [False, True],

    # In order to remove the copying from Acc vgpr to Arch vgpr, only use Arch vgprs for v_mfma_xxx.
    # Only support for kernel whose totalVgpr counts less than 256 and gcn that has control bit ACC_CD.
    "MIArchVgpr":               [False, True],

    # alternate implementation for fp16 HPA MFMA
    "Fp16AltImpl": [False, True],

    # Controls desired width (#elements) for loads from global memory -> LDS.
    # and eliminates the pointer unshift logic
    # -1 : Set GlobalReadVectorWidth =  VectorWidth
    # NOTE: for input bpe=32, max GRVW is 4  (to fit dwordx4) (FP32), min GRVW is 1 (dword)
    #                 bpe=16, max GRVW is 8  (to fit dwordx4) (FP16), min GRVW is 2 (dword)
    #                 bpe=8,  max GRVW is 16 (to fit dwordx4) (INT8), min GRVW is 4 (dword)
    "GlobalReadVectorWidth":      [ -1, 1, 2, 3, 4, 6, 8, 16 ],

    # Controls desired width (#elements) for loads from LDS -> VGPR.
    # -1 : Set LocalReadVectorWidth =  VectorWidth
    #  1 cannot be used for half type.
    # used in combination with TransposeLDS=True
    # in TransposeLDS=1 case, use wider load to fetch elements in summation dimension from LDS
    # helps optimizing instruction scheduling between MFMA and nonMFMA instructions
    # NOTE: for input bpe=32, max LRVW is 4  (to fit ds_read_b128) (FP32)
    #                 bpe=16, max LRVW is 8  (to fit ds_read_b128) (FP16)
    #                 bpe=8,  max LRVW is 16 (to fit ds_read_b128) (INT8)

    "LocalReadVectorWidth":      [ -1, 1, 2, 4, 8, 16 ],

    # threads should read/write/operate on this many contiguous elements from the C matrix.
    # If VW=4 then thread0 will process 4 consec C elements, then thread1 next 4, etc.
    # If the ThreadTile is > VectorWidth then thread0 will next operate on the 4 elements in C at (4*NumThreads)
    # Typically the load vector width and store vector width are directly related to the VW.
    # The global load width is closely related to the width of local stores so
    # GlobalReadVectorWidth also controls local write width.
    # Local read width also matches since VectorWidth consec elements must be read
    # Typically matching 16 bytes is good choice since the stores will be optimally coalesced with 16 bytes/WI.
    # -1 means use the largest vector width up to 128 bits.
    # Using a VW too large which results in >16bytes/thread isn't supported
    # For MFMA non SourceSwap: this parameter didn't take effect
    # For MFMA SourceSwap: this parameter only take effect on A buffer for now
    "VectorWidth":                [ -1, 1, 2, 3, 4, 6, 8 ],

    # If 0, store 1 element per instruction.
    # If 1, store vector-width elements per instruction.
    # if -1, store vector-wide elements per instruction unless PBD would not generate a valid kernel
    "VectorStore":                    [-1, 0, 1],

    # Controls desired width (#elements) for stores from reg to global memory.
    # When MatrixInstruciton == None, derived parameter gwvw takes precedence.
    # -1 : Set StoreVectorWidth = VectorWidth
    "StoreVectorWidth":           [ -1, 1, 2, 3, 4, 6, 8 ],

    # when loading all the data from global into lds requires multiple load instructions, these parameters govern which
    # loads will pull which rectangle of data from global into lds
    # NLC=1 means one load along the coalesced dimension, which results in the most coalescing possible
    # NLC=-1 looks for the largest number of reads along the coalesced dimension which results in the least ammount of coalescing;
    # however in this case the stride between one load and another is a static value, therefore buffer loads only need one set of registers
    # whereas the =1 case has a stride which is a multiple of a kernel argument and therefore needs one address per load in the perpendicular dimension
    "NumLoadsCoalescedA":         list(range(-1, 64+1)),
    "NumLoadsCoalescedB":         list(range(-1, 64+1)),

    # DepthU, LocalSplitU (which is the 3rd number in WorkGroup), and LoopUnroll are closely related
    # LoopUnroll=4 means there are 4 subiterations within the loop, 4 actual iterations written in the code.
    # LocalSplit=2 means the workgroup is split up into 2 subgroups, and each subgroup is doing different parts of the summation.
    # subgroup0 does k=0-3, 8-11... and subgroup1 does k=4-7, 12-15...
    # So, each iteration through the summation loop, which has 4 actual subiterations, does 8 summation iterations, because each subgroup did 4;
    # and when data is read from global memory the threads read 8 elements along the summation dimension.
    # DepthU = LoopUnroll * LocalSplitU = 4*2 in this case
    # it made more sense for the user to directly control LocalSplitU and DepthU, then derrive afterwards LoopUnroll=DepthU/LocalSplitU
    # -1 : Only allow GLVW=1
    # -2 : Only allow max(GLVWA,GLVWB) < VW ?
    # -3 : Only allow min(GLVWA,GLVWB) < VW ?
    "DepthU":                     depthUs,

    # DepthULdsDivisor (Split LDS) determines how we pipeline the data from global memory to LDS
    # Instead of moving all in-flight data from the register buffer (G2L) to the LDS at once, we divide the G2L buffer into N portions and
    # write each portion of the G2L to LDS, read from LDS and do the actual matrix multiply-accumulate, before moving on to the portion and so on.
    # This helps cut down LDS usage by the value of the divisor. Helps increase CU occupancy or DepthU if kernel was previously LDS limited.
    #
    # The premise is to be able to fetch 256B (equivalent to 128 half's or 64 single's) in TN layout problems to maximize L2 utilization. This
    # was previously a problem for TN since it implies DepthU is large, and that leads to oversubscription of LDS.
    #
    # Preconditions:
    # ScheduleIterAlg=3, TransposeLDS=1, PGR=0/1 exlcuding 2, DirectToLds=0 (DirectToLds=0 because part of the data loaded *need* to reside in registers),
    # nRegs per load >= DepthULdsDivisor (since we artificially require at least 1 register per LDS write)
    #
    # Example: DepthULdsDivisor=2
    # v0, v1, v2, v3 | v0, v1, v2, v3 | ... ----> unroll dim
    # -----Thd 0----- -----Thd 1-----   ...
    # 1st subloop writes v0,v1 to LDS
    # 2nd subloop writes v2,v3 to LDS
    "DepthULdsDivisor":           [1, 2, 4],

    # integer ammount of padding to put into LDS, in 2016 this didn't seem to help performance, profilers were showing that channel conflicts weren't really hurting
    # performance so this has been deprecated and probably doesn't work
    # -1 means use same padding as the VectorWidth if TLU=0 else 0.  (Padding only helps when transpose is required)
    # With MatrixInstruciton: -1 means max(GRVW,MIInput) if TLU=0
    "LdsPadA":                     [ -1, 0, 1, 2, 3, 4, 8, 16, 32],
    "LdsPadB":                     [ -1, 0, 1, 2, 3, 4, 8, 16, 32],

    # Padding boundary for LDS. defines block-size for pad insertion. for every 'LdsBlockSizePerPad' bytes, LDS padding (pad value from LdsPad parameter)
    # is added (readOffset aware of the pad and adjusts offset value based on this parameter value).
    # Only support LdsBlockSizePerPad >= unrollDepth * BPE
    # 0 means disable LdsBlockSizePerPad,
    # -1 means round up to nearest power of 2 begin with 128
    "LdsBlockSizePerPad":          [-1, 0, 64, 128, 256, 512, 1024],

    # Transpose LDS format. Local store in Coalsced dimension , same as optimized global fetch dimension . applicable only in TLU=0 case for miSIMD(s)
    # TODO: No code for -1 ?
    "TransposeLDS":                [-1, 1, 0],

    # add gls or slc after global memory read/writes to change cacheing, not cacheing the writes is promising and improved performance a tiny bit
    # 1: glc, 2: slc, 3: glc+slc
    "NonTemporalD":               list(range(0,4)),
    "NonTemporalC":               list(range(0,4)),
    "NonTemporalA":               list(range(0,4)),
    "NonTemporalB":               list(range(0,4)),

    # Group together unroll iterations inside the unroll loop.
    # For example, InnerUnroll=2 will fetch LDS for two unroll iterations
    "InnerUnroll":                [1,2,4,8,16,32,64],

    # Kernels should be written in assembly or source
    # if assembly, ISA will determine architecture
    # if source, Runtime will determine language
    # later on, we'll relax this to inner kernel languages and outer kernel languages, such as inline asm embedded in ocl or in llvm
    "KernelLanguage":             [ "Assembly" ],
    "ISA":                        validISA,       # arch for assembly kernels

    # Name of the custom kernel located in globalParameters["CustomKernelDirectory"].
    # a custom kernel is a user written assembly kernel with its associated configuration parameters included in a custom.config section
    # inside the yaml block between the --- and ... markers.  These parameters are only used for information purposes, not kernel generation.
    # Ex:
    # custom.config:
    #   ProblemType:
    #     OperationType: GEMM
    #     etc...
    #   ThreadTile: [8, 8]
    #   etc...
    #
    # Custom kernels can be included in a BenchmarkProblemSizeGroup by having their name (without file extension) listed under the "CustomKernels"
    # category alongside InitialSolutionParameters, BenchmarkCommonParameters, etc...
    "CustomKernelName":            -1,

    # Will allow a kernel to be accepted even when checks determine it's not viable.
    # Intended for use with custom kernels which have confirmed to be correct
    "NoReject":                    [False, True],

    "MinVgprNumber":                list(range(0,256)),

    "MaxVgprNumber":                list(range(0,257)),

    # Debug use only.
    "ActivationFused":             [False, True],

    # True-  function call
    # False- inline
    "ActivationFuncCall":          [False, True]
    }


# same parameter for all solution b/c depends only on compiler
defaultBenchmarkCommonParameters = [
    {"InnerUnroll":               [ 1 ] },
    {"KernelLanguage":            [ "Assembly" ] },
    {"LdsPadA":                   [ 0 ] },
    {"LdsPadB":                   [ 0 ] },
    {"LdsBlockSizePerPad":        [ 0 ] },
    {"TransposeLDS":              [ 0 ] },
    {"MaxOccupancy":              [ 40 ] },
    {"VectorWidth":               [ -1 ] },
    {"VectorStore":               [ -1 ] },
    {"StoreVectorWidth":          [ -1 ] },
    {"GlobalReadVectorWidth":     [ -1 ] },
    {"LocalReadVectorWidth":      [ -1 ] },
    {"WaveSeparateGlobalReadA":   [ 0 ] },
    {"WaveSeparateGlobalReadB":   [ 0 ] },
    {"PrefetchGlobalRead":        [ 1 ] },
    {"PrefetchLocalRead":         [ 1 ] },
    {"SuppressNoLoadLoop":        [ False ]},
    {"ExpandPointerSwap":         [ True ]},

    {"ScheduleGlobalRead":        [ 1 ] },
    {"ScheduleLocalWrite":        [ 1 ] },
    {"ScheduleIterAlg":           [ 1 ] },

    {"GlobalReadPerMfma":         [ 1 ] },
    {"LocalWritePerMfma":         [ -1 ] },

    {"InterleaveAlpha":           [ 0 ] },
    {"OptNoLoadLoop":             [ 1 ] },

    {"BufferLoad":                [ True ] },
    {"BufferStore":               [ True ] },
    {"DirectToVgprA":             [ False ] },
    {"DirectToVgprB":             [ False ] },
    {"DirectToLds":               [ False ] },
    {"UseSgprForGRO":             [ -1 ] },
    {"UseInstOffsetForGRO":       [ 0 ] },
    {"AssertSummationElementMultiple": [ 1 ] },
    {"AssertFree0ElementMultiple": [ 1 ] },
    {"AssertFree1ElementMultiple": [ 1 ] },

    {"StaggerU":                  [ 32 ] },   # recommend [0,32]
    {"StaggerUStride":            [ 256 ] },  # recommend 256 for V10,V20
    {"StaggerUMapping":           [ 0 ] },    # recommend [0,1]
    {"MagicDivAlg":               [ 2 ] },
    {"GlobalSplitU":              [ 1 ] },
    {"GlobalSplitUAlgorithm":     [ "SingleBuffer" ] },
    {"Use64bShadowLimit":         [ 1 ] },
    {"NumLoadsCoalescedA":        [ 1 ] },
    {"NumLoadsCoalescedB":        [ 1 ] },
    {"WorkGroup":                 [ [16,16,1]] },
    {"WorkGroupMapping":          [ 8 ] },
    {"ThreadTile":                [ [4,4] ] },
    {"WavefrontSize":             [ 64 ]},
    {"MatrixInstruction":         [ [] ] },
    {"1LDSBuffer":                [ 0 ] },
    {"DepthU":                    [ -1 ] },
    {"DepthULdsDivisor":          [ 1 ] },
    {"NonTemporalD":              [ 0 ] },
    {"NonTemporalC":              [ 0 ] },
    {"NonTemporalA":              [ 0 ] },
    {"NonTemporalB":              [ 0 ] },
    {"CustomKernelName":          [ "" ] },
    {"NoReject":                  [ False ]},
    {"MinVgprNumber":             [0]},
    {"MaxVgprNumber":             [256]},
    {"StoreRemapVectorWidth":     [ 0 ] },
    {"SourceSwap":                [ False ] },
    {"StorePriorityOpt":          [ False ] },
    {"NumElementsPerBatchStore":  [ 0 ] },
    {"StoreSyncOpt":              [ 0 ] },
    {"GroupLoadStore":            [ False ] },
    {"MIArchVgpr":                [ False ] },
    {"ActivationFused":           [ True  ] },
    {"ActivationFuncCall":        [ True  ] }
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
    # =GEMM uses TransposeA,B paramters and makes the problem type more readeable for users
    # =TensorContraction  requires specifying
    "OperationType":            "GEMM",           # GEMM, TensorContraction, ConvolutionForward, ConvolutionBackwardData, ConvolutionBackwardWeights

    "DataType":                 0,                # data types can specified by a variety of ways, such as "s", as listed in SolutionStructs.py::DataType
    "DestDataType":             0,                # destination data types can specified by a variety of ways, such as "s", as listed in SolutionStructs.py::DataType
    "ComputeDataType":          0,                # compute data types can specified by a variety of ways, such as "s", as listed in SolutionStructs.py::DataType
    "UseBeta":                  True,             # =True use beta parameter (asm will check for B=0 and optimize the write for that), =False don't use beta parameter
    "UseBias":                  False,            # =True use bias vector
    "UseScaleD":                False,            # =True use scaleD vector
    "HighPrecisionAccumulate":  False,            # f32 += f16*f16
    "SilentHighPrecisionAccumulate": False,       # Keep kernel names the same for HPA mode.  Useful for testing.

    "ComplexConjugateA":        False,            # complex data should be conjugated for "C" transpose case
    "ComplexConjugateB":        False,

    # for OperationType == GEMM
    "TransposeA":               False,            # =True means transA="T" or "C", =False means transA = "N"
    "TransposeB":               True,
    "Batched":                  False,            # add batching dimension
    "StridedBatched":           True,             # use to select general batch or strided batch
    "GroupedGemm":              False,             # use to select general batch or strided batch

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
    "IndexAssignmentsA":        [0, 2],
    "IndexAssignmentsB":        [1, 2],
    "NumIndicesC":              2,

    # use initial strides for AB.
    # This has some performance impact for the increased flexibility:
    #   - Additional strides will be passed into the kernel and will occupy SGPR registers
    #   - GlobalReadWidth must be 1 (since elements are not guaranteed to be adjacent in memory)
    "UseInitialStridesAB":      False,

    # use initial strides for CD.
    # This has some performance impact for the increased flexibility:
    #   - Additional strides will be passed into the kernel and will occupy SGPR registers
    #   - Additional multiply on the store address path
    #   -VectorStore must be 0.  If VectorStore is -1, it will be silently set to 0 internally.
    "UseInitialStridesCD":      False,

    "AllowNoFreeDims":          False,  # allow A or B to specify no free dims
                                        # (if false, A and B must have at least one free dim)
                                        # (if true, A and B must have at least one free or batch dim)

    # SetConstStride* sets the specified stride in the problem.
    # These no longer generate predicates - see AssertStrideEqualA/B below
    # List of pairs of [index, constValue].
    # Index is a member of the global index assignments (not an offset into IndexAssignmentsA/B)
    # EX: SetConstStrideA: [ [3, 1], [2, 4] ] sets
    #     strideA for index3 to constant '1' and stride for index2 to constant '4'.
    "SetConstStrideA":          [],
    "SetConstStrideB":          [],

    # Summation dimension indices
    "MirrorDimsA":              [],
    "MirrorDimsB":              [],

    # for LD description
    "NumIndicesLD":             4,
    "IndexAssignmentsLD":       [3, 4, 5, 6],      # order is LDD, LDC, LDA, LDB

    # Tile aware solution selection
    "TileAwareSelection":       False,

    # FP16 Alternate Implementation
    "Fp16AltImpl":              False,

    # Activation
    "Activation":               False,
    "ActivationHPA":            False
    }

defaultProblemSizes = [{"Range": [ [2880], 0, 0 ]}]
defaultBenchmarkFinalProblemSizes = [{"Range": [
    [64, 64, 64, 512], 0, 0 ]}]
defaultBatchedProblemSizes = [{"Range": [ [2880], 0, [1], 0 ]}]
defaultBatchedBenchmarkFinalProblemSizes = [{"Range": [
    [64, 64, 64, 512], 0, [1], 0 ]}]


defaultSolutionSummationSizes = [32,64,96,128,256,512,1024,2048,4096,8192,16192]


################################################################################
# Default Analysis Parameters
################################################################################
defaultAnalysisParameters = {
    "ScheduleName":       "Tensile",
    "DeviceNames":  "fallback",
    "ArchitectureName": "gfx000",
    "SolutionImportanceMin":      0.01, # = 0.01=1% total time saved by keeping this solution
    }


################################################################################
# Searching Nested Lists / Dictionaries
# to see if keys exist and what their values are
################################################################################
# param name in structures?
def inListOfDictionaries(param, dictionaries):
  for dictionary in dictionaries:
    if param in dictionary:
      return True
  return False
def inListOfListOfDictionaries(param, dictionaries):
  for dictionaryList in dictionaries:
    if inListOfDictionaries(param, dictionaryList):
      return True
  return False
def inListOfLists(param, lists):
  for l in lists:
    if param in l:
      return True
  return False

# get param values from structures.
def hasParam( name, structure ):
  if isinstance(structure, list):
    for l in structure:
      if hasParam(name, l):
        return True
    return False
  elif isinstance(structure, dict):
    return name in structure
  else:
    return name == structure
    #printExit("structure %s is not list or dict" % structure)

def getParamValues( name, structure ):
  if isinstance(structure, list):
    for l in structure:
      param = getParamValues(name, l)
      if param != None:
        return param
    return None
  elif isinstance(structure, dict):
    if name in structure:
      return structure[name]
    else:
      return None
  else:
    printExit("structure %s is not list or dict" % structure)

################################################################################
# Print Debug
################################################################################
def print1(message):
  if globalParameters["PrintLevel"] >= 1:
    print(message)
    sys.stdout.flush()
def print2(message):
  if globalParameters["PrintLevel"] >= 2:
    print(message)
    sys.stdout.flush()

def printWarning(message):
  print("Tensile::WARNING: %s" % message)
  sys.stdout.flush()
def printExit(message):
  print("Tensile::FATAL: %s" % message)
  sys.stdout.flush()
  sys.exit(-1)

################################################################################
# Locate Executables
# rocm-smi, hip-clang, rocm_agent_enumerator, clang-offload-bundler
################################################################################
def isExe( filePath ):
  return os.path.isfile(filePath) and os.access(filePath, os.X_OK)
def locateExe( defaultPath, exeName ): # /opt/rocm/bin, hip-clang
  # look in defaultPath first
  exePath = os.path.join(defaultPath, exeName)
  if isExe(exePath):
    return exePath
  # look in PATH second
  for path in os.environ["PATH"].split(os.pathsep):
    exePath = os.path.join(path, exeName)
    if isExe(exePath):
      return exePath
  return None

def gfxArch(name):
    import re
    match = re.search(r'gfx([0-9a-fA-F]{3,})', name)
    if not match: return None

    ipart = match.group(1)

    step = int(ipart[-1], 16)
    ipart = ipart[:-1]

    minor = int(ipart[-1])
    ipart = ipart[:-1]

    major = int(ipart)

    rv = (major, minor, step)

    return rv

def detectGlobalCurrentISA():
  """
  Returns returncode if detection failure
  """
  global globalParameters

  if globalParameters["CurrentISA"] == (0,0,0) and globalParameters["ROCmAgentEnumeratorPath"]:
    process = subprocess.run([globalParameters["ROCmAgentEnumeratorPath"]], stdout=subprocess.PIPE)
    if os.name == "nt":
      line = ""
      for line_in in process.stdout.decode().splitlines():
        if 'gcnArchName' in line_in:
          line += line_in.split()[1]
          break # detemine if hipinfo will support multiple arch
      arch = gfxArch(line.strip())
      if arch is not None:
        if arch in globalParameters["SupportedISA"]:
          print1("# Detected local GPU with ISA: " + getGfxName(arch))
          globalParameters["CurrentISA"] = arch
    else:
      archList = []
      for line in process.stdout.decode().split("\n"):
        arch = gfxArch(line.strip())
        if arch is not None:
          if arch in globalParameters["SupportedISA"]:
            print1("# Detected local GPU with ISA: " + getGfxName(arch))
            archList.append(arch)
      if len(archList) > 0:
        globalParameters["CurrentISA"] = archList[globalParameters["Device"]]
    if (process.returncode):
      printWarning("%s exited with code %u" % (globalParameters["ROCmAgentEnumeratorPath"], process.returncode))
    return process.returncode
  return 0

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

def printTable(rows):
  rows = list([[str(cell) for cell in row] for row in rows])
  colWidths = list([max([len(cell) for cell in col]) for col in zip(*rows)])

  for row in rows:
    for (width, cell) in zip(colWidths, row):
      pad = ' ' * (width - len(cell))
      print(pad, cell, sep='', end=' ')
    print()

def printCapTable(parameters):
  import itertools
  archs = [(0,0,0)] + parameters["SupportedISA"]
  gfxNames = list(map(getGfxName, archs))

  headerRow = ['cap'] + gfxNames

  def capRow(caps, cap):
    return [cap] + [('1' if cap in caps[arch] and caps[arch][cap] else '0') for arch in archs]

  allAsmCaps = set(itertools.chain(*[caps.keys() for arch, caps in parameters["AsmCaps"].items()]))
  allAsmCaps = sorted(allAsmCaps, key=lambda k: (k.split("_")[-1], k))
  asmCapRows = [capRow(parameters["AsmCaps"], cap) for cap in allAsmCaps]

  allArchCaps = set(itertools.chain(*[caps.keys() for arch, caps in parameters["ArchCaps"].items()]))
  allArchCaps = sorted(allArchCaps)
  archCapRows = [capRow(parameters["ArchCaps"], cap) for cap in allArchCaps]

  printTable([headerRow] + asmCapRows + archCapRows)

def which(p):
    exes = [p+x for x in ['', '.exe', '.bat']]
    system_path = os.environ['PATH'].split(os.pathsep)
    if p == 'hipcc' and 'CMAKE_CXX_COMPILER' in os.environ and os.path.isfile(os.environ['CMAKE_CXX_COMPILER']):
        return os.environ['CMAKE_CXX_COMPILER']
    for dirname in system_path+[globalParameters["ROCmBinPath"]]:
        for exe in exes:
            candidate = os.path.join(os.path.expanduser(dirname), exe)
            if os.path.isfile(candidate):
                return candidate
    return None

################################################################################
################################################################################
def assignGlobalParameters( config ):
  """
  Assign Global Parameters
  Each global parameter has a default parameter, and the user
  can override them, those overridings happen here
  """

  global globalParameters

  # Minimum Required Version
  if "MinimumRequiredVersion" in config:
    if not versionIsCompatible(config["MinimumRequiredVersion"]):
      printExit("Config file requires version=%s is not compatible with current Tensile version=%s" \
          % (config["MinimumRequiredVersion"], __version__) )

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
    globalParameters["ROCmPath"] = os.environ.get("HIP_DIR") # windows has no ROCM
  globalParameters["CmakeCxxCompiler"] = None
  if "CMAKE_CXX_COMPILER" in os.environ:
    globalParameters["CmakeCxxCompiler"] = os.environ.get("CMAKE_CXX_COMPILER")

  globalParameters["ROCmBinPath"] = os.path.join(globalParameters["ROCmPath"], "bin")

  # ROCm Agent Enumerator Path
  if os.name == "nt":
    globalParameters["ROCmAgentEnumeratorPath"] = locateExe(globalParameters["ROCmBinPath"], "hipinfo.exe")
  else:
    globalParameters["ROCmAgentEnumeratorPath"] = locateExe(globalParameters["ROCmBinPath"], "rocm_agent_enumerator")

  if "CxxCompiler" in config:
    globalParameters["CxxCompiler"] = config["CxxCompiler"]

  if "TENSILE_ROCM_ASSEMBLER_PATH" in os.environ:
    globalParameters["AssemblerPath"] = os.environ.get("TENSILE_ROCM_ASSEMBLER_PATH")
  elif globalParameters["AssemblerPath"] is None and globalParameters["CxxCompiler"] == "hipcc":
    if os.name == "nt":
      globalParameters["AssemblerPath"] = locateExe(globalParameters["ROCmBinPath"], "clang++.exe")
    else:
      globalParameters["AssemblerPath"] = locateExe(os.path.join(globalParameters["ROCmPath"], "llvm/bin"), "clang++")

  globalParameters["ROCmSMIPath"] = locateExe(globalParameters["ROCmBinPath"], "rocm-smi")

  globalParameters["ExtractKernelPath"] = locateExe(os.path.join(globalParameters["ROCmPath"], "hip/bin"), "extractkernel")

  if "TENSILE_ROCM_OFFLOAD_BUNDLER_PATH" in os.environ:
    globalParameters["ClangOffloadBundlerPath"] = os.environ.get("TENSILE_ROCM_OFFLOAD_BUNDLER_PATH")
  else:
    if os.name == "nt":
      globalParameters["ClangOffloadBundlerPath"] = locateExe(globalParameters["ROCmBinPath"], "clang-offload-bundler.exe")
    else:
      globalParameters["ClangOffloadBundlerPath"] = locateExe(os.path.join(globalParameters["ROCmPath"], "llvm/bin"), "clang-offload-bundler")

  if "ROCmAgentEnumeratorPath" in config:
    globalParameters["ROCmAgentEnumeratorPath"] = config["ROCmAgentEnumeratorPath"]

  # read current gfx version
  returncode = detectGlobalCurrentISA()
  if globalParameters["CurrentISA"] == (0,0,0):
    printWarning("Did not detect SupportedISA: %s; cannot benchmark assembly kernels." % globalParameters["SupportedISA"])
  if returncode:
    if os.name == "nt":
      globalParameters["CurrentISA"] = (9,0,6)
      printWarning("Failed to detect ISA so forcing (gfx906) on windows")

  # TODO Remove this when rocm-smi supports gfx90a
  if globalParameters["CurrentISA"] == (9,0,10):
    printWarning("HardwareMonitor currently disabled for gfx90a")
    globalParameters["HardwareMonitor"] = False

  globalParameters["AsmCaps"] = {}
  globalParameters["ArchCaps"] = {}
  globalParameters["AsmBugs"] = {}

  for v in globalParameters["SupportedISA"] + [(0,0,0)]:
    ti = TensileInstructions()
    ti.init(v, globalParameters["AssemblerPath"], (globalParameters["PrintLevel"] >= 2))
    globalParameters["AsmCaps"][v] = ti.getAsmCaps()
    globalParameters["ArchCaps"][v] = ti.getArchCaps()
    globalParameters["AsmBugs"][v] = ti.getAsmBugs()

  if globalParameters["PrintLevel"] >= 1:
    printCapTable(globalParameters)

  globalParameters["SupportedISA"] = list([i for i in globalParameters["SupportedISA"] if globalParameters["AsmCaps"][i]["SupportedISA"]])

  validParameters["ISA"] = [(0,0,0), *globalParameters["SupportedISA"]]

  if "MergeFiles" in config and "NumMergedFiles" in config:
    if not config["MergeFiles"] and config["NumMergedFiles"] > 1:
      config["NumMergedFiles"] = 1
      printWarning("--num-merged-files and --no-merge-files specified, ignoring --num-merged-files")

  # For ubuntu platforms, call dpkg to grep the version of hip-clang.  This check is platform specific, and in the future
  # additional support for yum, dnf zypper may need to be added.  On these other platforms, the default version of
  # '0.0.0' will persist

  # Due to platform.linux_distribution() being deprecated, just try to run dpkg regardless.
  # The alternative would be to install the `distro` package.
  # See https://docs.python.org/3.7/library/platform.html#platform.linux_distribution
  try:
    if os.name == "nt":
      compileArgs = ['perl'] + [which('hipcc')] + ['--version']
      output = subprocess.run(compileArgs, check=True, stdout=subprocess.PIPE).stdout.decode()
    else:
      compiler = "hipcc"
      output = subprocess.run([compiler, "--version"], check=True, stdout=subprocess.PIPE).stdout.decode()

    for line in output.split('\n'):
      if 'HIP version' in line:
        globalParameters['HipClangVersion'] = line.split()[2]
        print1("# Found  hipcc version " + globalParameters['HipClangVersion'])

  except (subprocess.CalledProcessError, OSError) as e:
      printWarning("Error: {} running {} {} ".format('hipcc', '--version',  e))

  for key in config:
    value = config[key]
    if key not in globalParameters:
      printWarning("Global parameter %s = %s unrecognised." % ( key, value ))
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

################################################################################
# Assign Parameters
# populate dst with src[key] else give it the default/backup value
################################################################################
def assignParameterWithDefault(destinationDictionary, key, sourceDictionary, \
    defaultDictionary):
  if key in sourceDictionary:
    destinationDictionary[key] = deepcopy(sourceDictionary[key])
  else:
    destinationDictionary[key] = deepcopy(defaultDictionary[key])

################################################################################
# Push / Pop Working Path
# store a WorkingPath where to write files (like benchmark files)
################################################################################
def pushWorkingPath( foldername ):
  # Warning: this is not thread-safe, modifies the global WorkingPath!
  globalParameters["WorkingPath"] = \
      os.path.join(globalParameters["WorkingPath"], foldername )
  return ensurePath( globalParameters["WorkingPath"] )
def popWorkingPath():
  # Warning: this is not thread-safe, modifies the global WorkingPath!
  if len(workingDirectoryStack) == 0:
    globalParameters["WorkingPath"] = \
      os.path.split(globalParameters["WorkingPath"])[0]
  else:
    globalParameters["WorkingPath"] = workingDirectoryStack.pop()
def ensurePath(path):
  try:
    os.makedirs(path)
  except FileExistsError:
    pass
  except OSError:
    printExit("Failed to create directory \"%s\" " % (path) )
  return path
def setWorkingPath( fullPathName ):
  # Warning: this is not thread-safe, modifies the global WorkingPath!
  workingDirectoryStack.append(globalParameters["WorkingPath"])
  globalParameters["WorkingPath"] = ensurePath(fullPathName)


def roundUp(f):
  return (int)(math.ceil(f))

################################################################################
# Is query version compatible with current version
# a yaml file is compatible with tensile if
# tensile.major == yaml.major and tensile.minor.step > yaml.minor.step
################################################################################
def versionIsCompatible(queryVersionString):
  (qMajor, qMinor, qStep) = queryVersionString.split(".")
  (tMajor, tMinor, tStep) = __version__.split(".")

  # major version must match exactly
  if qMajor != tMajor:
    return False

  # minor.patch version must be >=
  if int(qMinor) > int(tMinor):
    return False
  if qMinor == tMinor:
    if int(qStep) > int(tStep):
      return False
  return True

def ClientExecutionLock():
  if not globalParameters["ClientExecutionLockPath"]:
    return open(os.devnull)

  import filelock
  return filelock.FileLock(globalParameters["ClientExecutionLockPath"])

# convert python list to C++ initializer style syntax
def listToInitializer(l):
  return "{" + ','.join(map(str, l)) + "}"

################################################################################
# Progress Bar Printing
# prints "||||" up to width
################################################################################
class ProgressBar:
  def __init__(self, maxValue, width=80):
    self.char = '|'
    self.maxValue = maxValue
    self.width = width
    self.maxTicks = self.width - 7


    self.priorValue = 0
    self.fraction = 0
    self.numTicks = 0
    self.createTime = time.time()

  def increment(self, value=1):
    self.update(self.priorValue+value)

  def update(self, value):
    currentFraction = 1.0 * value / self.maxValue
    currentNumTicks = int(currentFraction * self.maxTicks)
    if currentNumTicks > self.numTicks:
      self.numTicks = currentNumTicks
      self.fraction = currentFraction
      self.printStatus()
    self.priorValue = value

  def printStatus(self):
    sys.stdout.write("\r")
    sys.stdout.write("[%-*s] %3d%%" \
        % (self.maxTicks, self.char*self.numTicks, self.fraction*100) )
    if self.numTicks == self.maxTicks:
      stopTime = time.time()
      sys.stdout.write(" (%-.1f secs elapsed)\n"%(stopTime-self.createTime))
    sys.stdout.flush()

  def finish(self): pass

from copy import copy
class Backup:
  """RAII class to restore backed up fields from object"""
  fields = {}
  object = None
  def __init__(self, object, **fields):
    self.object = object
    for k, v in fields.items():
        self.fields[k] = copy(v)
  def __del__(self):
    for k, v in self.fields.items():
        setattr(self.object, k, v)

# Append copyrights to all files generated by tensile since they belong to Tensile intellectual property
CMakeHeader = """################################################################################
#
# Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

###################################################
# This file was generated by Tensile:             #
# https://github.com/ROCmSoftwarePlatform/Tensile #
###################################################


"""

CHeader = """/*******************************************************************************
* Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
* ies of the Software, and to permit persons to whom the Software is furnished
* to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
* PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
* FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
* COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
* IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
* CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*******************************************************************************/

/**************************************************
* This file was generated by Tensile:             *
* https://github.com/ROCmSoftwarePlatform/Tensile *
**************************************************/


"""

HR = "################################################################################"
