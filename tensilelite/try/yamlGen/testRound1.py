print("HELLORound1 try")

import pandas as pd

list = [
          "- [16, 16,16, 1,  1,  1, 1,  1,1 ] #MT16x16",
          "- [16, 16,16, 1,  1,  1, 1,  1,2 ] #1#MT16x32",
          "- [16, 16,16, 1,  1,  1, 2,  1,1 ] #1#MT16x32X",
          "- [16, 16,16, 1,  1,  1, 1,  1,4 ] #1##MT16x64",
          "- [16, 16,16, 1,  1,  1, 2,  1,2 ] #1##MT16x64?X",
          "- [16, 16,16, 1,  1,  1, 3,  1,2 ] #1#MT16x96",
          "- [16, 16,16, 1,  1,  1, 2,  1,4 ] #1#MT16x128",
          "- [16, 16,16, 1,  1,  1, 4,  1,2 ] #1#MT16x128?X",
          "- [16, 16,16, 1,  1,  1, 3,  1,4 ] #1#MT16x192",
          "- [16, 16,16, 1,  1,  1, 4,  1,4 ] #1#MT16x256??X",
          "- [16, 16,16, 1,  1,  1, 1,  2,1 ] #MT32x16V",
          "- [16, 16,16, 1,  1,  1, 1,  2,2 ] #1MT32x32V",
          "- [16, 16,16, 1,  1,  1, 2,  2,1 ] #1MT32x32??",
          "- [16, 16,16, 1,  1,  1, 3,  2,1 ] #2#MT32x48?",
          "- [16, 16,16, 1,  1,  1, 2,  2,2 ] #2#MT32x64",
          "- [16, 16,16, 1,  1,  2, 1,  1,4 ] #2#MT32x64?",
          "- [16, 16,16, 1,  1,  1, 4,  2,1 ] #2#MT32x64??",
          "- [16, 16,16, 1,  1,  1, 3,  2,2 ] #2#MT32x96?",
          "- [16, 16,16, 1,  1,  2, 2,  1,4 ] #2#MT32x128",
          "- [16, 16,16, 1,  1,  1, 4,  2,2 ] #2#MT32x128?",
          "- [16, 16,16, 1,  1,  2, 3,  1,4 ] #2#MT32x192",
          "- [16, 16,16, 1,  1,  3, 1,  1,2 ] #MT48x32",
          "- [16, 16,16, 1,  1,  3, 1,  1,4 ] #3#MT48x64",
          "- [16, 16,16, 1,  1,  1, 1,  4,1 ] #MT64x16",
          "- [16, 16,16, 1,  1,  2, 1,  2,2 ] #MT64x32",
          "- [16, 16,16, 1,  1,  2, 2,  2,1 ] #MT64x32?X",
          "- [16, 16,16, 1,  1,  1, 2,  4,1 ] #MT64x32??",
          "- [16, 16,16, 1,  1,  1, 3,  4,1 ] #MT64x48?",
          "- [16, 16,16, 1,  1,  2, 2,  2,2 ] #1MT64x64V",
          "- [16, 16,16, 1,  1,  1, 4,  4,1 ] #3MT64x64?",
          "- [16, 16,16, 1,  1,  4, 1,  1,4 ] #3MT64x64?",
          "- [16, 16,16, 1,  1,  2, 3,  2,2 ] #2#MT64x96 #22",
          "- [16, 16,16, 1,  1,  2, 4,  2,2 ] #2#MT64x128 #21",
          "- [16, 16,16, 1,  1,  4, 2,  1,4 ] #2#MT64x128? #21",
          "- [16, 16,16, 1,  1,  3, 1,  2,1 ] #MT96x16?",
          "- [16, 16,16, 1,  1,  3, 2,  2,2 ] #MT96x64 #22",
          "- [16, 16,16, 1,  1,  3, 1,  2,2 ] #MT96x32",
          "- [16, 16,16, 1,  1,  2, 1,  4,1 ] #MT128x16",
          "- [16, 16,16, 1,  1,  2, 2,  4,1 ] #MT128x32",
          "- [16, 16,16, 1,  1,  4, 1,  2,2 ] #MT128x32?",
          "- [16, 16,16, 1,  1,  2, 3,  4,1 ] #MT128x48V",
          "- [16, 16,16, 1,  1,  2, 4,  4,1 ] #MT128x64? #20",
          "- [16, 16,16, 1,  1,  4, 2,  2,2 ] #MT128x64? #20",
          "- [16, 16,16, 1,  1,  3, 1,  4,1 ] #MT192x16",
          "- [16, 16,16, 1,  1,  3, 2,  4,1 ] #MT192x32",
          "- [16, 16,16, 1,  1,  4, 1,  4,1 ] #MT256x16?",
]

for idx in range(1,23+1):
  # print(idx)
  file1 = open("/Victor/hipBLASLt/hipBLASLt/tensilelite/try/yamlGen/FP16_NN_MI250X_testBFVF_"+str(idx)+".yaml","w")

  file1.write(
"\
GlobalParameters:\n\
  MinimumRequiredVersion: 4.14.0\n\
  SleepPercent: 50\n\
  NumElementsToValidate: 0\n\
  DataInitTypeBeta: 0\n\
  DataInitTypeAlpha: 1\n\
  DataInitTypeBias: 0\n\
  NewClient: 2\n\
  CSVExportWinner: 1\n\
  CSVMergeSameProblemID: 1\n\
  Device: 4\n\
  # PrintSolutionRejectionReason: True\n\
  # SyncsPerBenchmark: 10\n\
  EnqueuesPerSync: 10\n\
  KernelTime: True\n\
  # ShortNames: True\n\
  # DataInitTypeScaleD: 0\n\
  NumWarmups: 10\n\
\n\
BenchmarkProblems:\n\
  ########################################\n\
  # NN - standard\n\
  ########################################\n\
  -\n\
    - # ProblemType\n\
      OperationType: GEMM\n\
      DataType: h\n\
      DestDataType: h\n\
      ComputeDataType: s\n\
      HighPrecisionAccumulate: True\n\
      TransposeA: 0\n\
      TransposeB: 0\n\
      # TransposeB: 1\n\
      UseBeta: True\n\
      Batched: True\n\
      # UseBias: True\n\
      # Activation: True\n\
      # ActivationHPA: True\n\
      # UseScaleD: True\n\
    - # BenchmarkProblemSizeGroup - Standard - All problem\n\
      InitialSolutionParameters:\n\
      BenchmarkCommonParameters:\n\
        - KernelLanguage: [\"Assembly\"]\n\
        # - EdgeType: [\"ShiftPtr\"]\n\
      ForkParameters:\n\
        - MatrixInstruction:\n\
          "+list[(idx-1)*2]+"\n\
          "+list[(idx-1)*2+1]+"\n\
        # - AssertFree0ElementMultiple: [4,8]\n\
        - AssertFree0ElementMultiple: [8]\n\
        - PrefetchGlobalRead: [2] #1\n\
        # - PrefetchLocalRead: [3,5]\n\
        # - PrefetchLocalRead: [1,3]\n\
        # - PrefetchLocalRead: [5,9]\n\
        - PrefetchLocalRead: [1,3,5,9]\n\
        # - DepthU: [64,96]\n\
        - DepthU: [16,32,64]\n\
        # - DepthU: [32]\n\
        - VectorWidth: [8]\n\
        # - VectorWidth: [1,2]\n\
        # - VectorWidth: [2,4,8]\n\
        # - VectorWidth: [-1,2,4,8]\n\
        # - GlobalReadVectorWidth: [8]\n\
        - GlobalReadVectorWidth: [-1,2,4,8]\n\
        # - GlobalReadVectorWidth: [4]\n\
        # - LocalReadVectorWidth: [4,8]\n\
        - LocalReadVectorWidth: [-1,2,4,8]\n\
        # - LocalReadVectorWidth: [4]\n\
        - ScheduleIterAlg: [3]\n\
        - InnerUnroll: [1]\n\
        - ExpandPointerSwap: [0]\n\
        - TransposeLDS: [1] #NN\n\
        # - TransposeLDS: [0] #NT\n\
        - LdsBlockSizePerPad: [-1]\n\
        - LdsPadA: [-1]\n\
        - LdsPadB: [-1]\n\
        # - StaggerUStride: [0]\n\
        # - StaggerUStride: [128,256]\n\
        # - StaggerU: [0,32]\n\
        # - StaggerU: [0,4,32]\n\
        # - WorkGroupMapping: [2]\n\
        - WorkGroupMapping: [1,4,8,16,32,64,110]\n\
        # - WorkGroupMapping: [1]\n\
        # - StaggerUMapping: [0,3]\n\
        # - WaveSeparateGlobalReadA: [1]\n\
        # - WaveSeparateGlobalReadA: [0,1]\n\
        # - WaveSeparateGlobalReadB: [0,1]\n\
        # - WaveSeparateGlobalReadB: [1]\n\
        # - MaxOccupancy: [40]\n\
        # - 1LDSBuffer: [0]\n\
        - 1LDSBuffer: [0,1]\n\
        # - 1LDSBuffer: [1]\n\
        # - GlobalSplitU: [8]\n\
        - GlobalSplitU: [1]\n\
        # - GlobalSplitU: [2,3,4,5,15,16]\n\
        # - GlobalSplitU: [6,8,9]\n\
        # - GlobalSplitU: [1]\n\
        # - GlobalSplitUAlgorithm: [\"MultipleBuffer\"]\n\
        - GlobalSplitUAlgorithm: [\"SingleBuffer\"]\n\
        - GlobalReadPerMfma: [1]\n\
        - LocalWritePerMfma: [-1]\n\
        # - StoreVectorWidth: [4]\n\
        # - StoreVectorWidth: [2,4]\n\
        # - StoreVectorWidth: [1]\n\
        # - SourceSwap: [0]\n\
        - SourceSwap: [0,1]\n\
        # - SourceSwap: [1]\n\
        # - NumElementsPerBatchStore: [2]\n\
        # - NumElementsPerBatchStore: [0, 2]\n\
        # - NumElementsPerBatchStore: [2]\n\
        - StorePriorityOpt: [1]\n\
        # - StorePriorityOpt: [0,1]X\n\
        # - NumLoadsCoalescedA: [1]\n\
        # - NumLoadsCoalescedA: [1,3]X\n\
        - StoreRemapVectorWidth: [0,-1]\n\
      BenchmarkJoinParameters:\n\
      BenchmarkFinalParameters:\n\
        - ProblemSizes:\n\
          - Exact: [320  , 8192, 1 , 320]\n\
          - Exact: [640  , 2048, 1 , 640]\n\
          - Exact: [320  , 8192, 1 , 1280]\n\
          - Exact: [1280 , 512 , 1 , 1280]\n\
          - Exact: [10240, 512 , 1 , 1280]\n\
          - Exact: [2560 , 8192, 1 , 320]\n\
          - Exact: [5120 , 2048, 1 , 640]\n\
          - Exact: [1280 , 512 , 1 , 5120]\n\
          - Exact: [640  , 2048, 1 , 2560]\n\
          - Exact: [320  , 154 , 1 , 768]\n\
          - Exact: [1280 , 154 , 1 , 768]\n\
          - Exact: [40   , 4096, 16, 4096]\n\
          - Exact: [80   , 1024, 16, 1024]\n\
          - Exact: [40   , 4096, 16, 77]\n\
          - Exact: [80   , 1024, 16, 77]\n\
          - Exact: [160  , 256 , 16, 256]\n\
        - ActivationArgs:\n\
          - [Enum: none]\n\
        - BiasTypeArgs: ['s']\n\
LibraryLogic:\n\
    ScheduleName: \"aldebaran\"\n\
    DeviceNames: [\"Device 0050\", \"Device 0051\", \"Device 0052\", \"Device 0054\", \"Device 0062\", \"Device 7400\", \"Device 740c\"]\n\
    ArchitectureName: \"gfx90a\"\n\
\n\
"
  )
