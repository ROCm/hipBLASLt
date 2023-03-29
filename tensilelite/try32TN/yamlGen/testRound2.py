print("HELLORound2\n")

import pandas as pd

def get_arg(arg):
    start = WinnerName.find(arg) + 1
    end = start + WinnerName[start:].find('_')
    if arg == '_WGM':
      end = len(WinnerName)
    value = WinnerName[start+len(arg)-1:end]
    # if arg == '_VW':
    #   end = start+len(arg)
    #   print("_VW",WinnerName[start+len(arg)-1:])
    # value = WinnerName[start+len(arg)-1:end]
    return value

def Convert(string):
    li = list(string.split(","))
    return li

problemnum = 5

idx=1
df = pd.read_csv('/Victor/hipBLASLt/hipBLASLt/tensilelite/try32TN/yamlGen/chcek.csv', encoding='gbk')
# print(list(df))
# print(df['kernelname'])
for problemidx in range(0, problemnum):
  print(problemidx)
  file1 = open("/Victor/hipBLASLt/hipBLASLt/tensilelite/try32TN/yamlGen/FP16_NN_MI250X_testBFVF_Round2_"+str(problemidx)+".yaml","w")
  WinnerName = df['kernelname'][problemidx]
  # WinnerName = " Cijk_Ailk_Bljk_HHS_BH_MT32x64x64_MI32x32x8x1_SN_1LDSB0_GRVW7_K1_LPB8_LRVW8_MIWT1_1_PLR10_SRVW4_SVW4_VW8_WG32_4_1_WGM16"#df[' WinnerName'][problemidx]

  # pos = -1 
  # while True:
  #     pos = text.find('x', pos + 1)
  #     if pos == -1:
  #         break
  #     do_something

  start = -1

  start = WinnerName.find('_MT')
  # print(WinnerName[start:])
  MT = get_arg('_MT')
  # print(MT)
  start_MT0 = 0
  end_MT0 = MT.find('x')
  MT0 = MT[start_MT0:end_MT0]
  # print(MT0)
  start_MT1 = end_MT0 + 1
  # print(MT[start_MT1:])
  end_MT1 = start_MT1 + MT[start_MT1:].find('x')
  MT1 = MT[start_MT1:end_MT1]
  # print(MT1)


  start = WinnerName.find('_MIWT') + 1
  end = start + WinnerName[start:].find('_')
  WT0 = WinnerName[start+len('_MIWT')-1:end]
  # print(WT0)
  start = end + 1
  # print(WinnerName[start:])
  end = start + WinnerName[start:].find('_')
  WT1 = WinnerName[start:end]
  # print(WT1)

  start = WinnerName.find('x')
  start = start+1 + WinnerName[start+1:].find('x')
  # start = WinnerName[start:].find('x', start + 1)
  # print(WinnerName[start:],"\n")
  end = start + WinnerName[start:].find('_')
  DepthU = WinnerName[start+len('x'):end]
  # print(DepthU,"\n")

  start = end+3
  end = start + WinnerName[start:].find('_')
  MI = WinnerName[start:end]
  # print(MI)
  end_MI_M = end
  start_MI_M = start
  end_MI_M = start_MI_M + WinnerName[start_MI_M:end_MI_M].find('x')
  MI_M = WinnerName[start_MI_M:end_MI_M]
  # print(MI_M)
  start = end_MI_M + 1
  # print(WinnerName[start:end])
  end_MI_N = end
  start_MI_N = start
  end_MI_N = start_MI_N + WinnerName[start_MI_N:end_MI_N].find('x')
  MI_N = WinnerName[start_MI_N:end_MI_N]
  # print(MI_N)
  start = end_MI_N + 1
  # print(WinnerName[start:end])
  end_MI_K = end
  start_MI_K = start
  end_MI_K = start_MI_K + WinnerName[start_MI_K:end_MI_K].find('x')
  MI_K = WinnerName[start_MI_K:end_MI_K]
  # print(MI_K)

  WG0 = (int(MT0)/int(WT0))/int(MI_M)
  WG1 = (int(MT1)/int(WT1))/int(MI_N)

  AssertFree0ElementMultiple = get_arg('_AF0EM')
  TransposeLDS = get_arg('_TLDS')
  PrefetchLocalRead = get_arg('_PLR')
  VectorWidth = get_arg('_VW')
  # if VectorWidth == 'ijk':
  #   VectorWidth = '-1,2,4,8'
  GlobalReadVectorWidth = get_arg('_GRVW')
  LocalReadVectorWidth = get_arg('_LRVW')
  # if LocalReadVectorWidth == 'k':
  #   LocalReadVectorWidth = '-1,2,4,8'
  WorkGroupMapping = get_arg('_WGM')
  # print(WorkGroupMapping,"\n")
  LDSBuffer = get_arg('1LDSB')
  StoreVectorWidth = get_arg('_SVW')
  SourceSwap = get_arg('_SS')
  # print(SourceSwap,"\n")
  # if SourceSwap == 'ijk':
  #   SourceSwap = 0
  # print(SourceSwap,"\n")  
  StoreRemapVectorWidth = get_arg('_SRVW')

  WaveSeparateGlobalReadA = get_arg('_WSGRA')
  WaveSeparateGlobalReadB = get_arg('_WSGRB')

  StaggerUStride = get_arg('_SUS')
  print("StaggerUStride",StaggerUStride)

  StaggerU = get_arg('_SU')
  print("StaggerU",StaggerU)

  problemSize = df['problemsize'][problemidx]
  print(problemSize)

  listproblemSize = Convert(problemSize)
  print(listproblemSize)

  yaml = "\n\
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
        TransposeA: 1\n\
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
              - ["+str(MI_M)+", "+str(MI_N)+", "+str(MI_K)+", 1,  1, "+str(WT0)+", "+str(WT1)+", "+str(WG0)+", "+str(WG1)+"]\n\
            # - [32, 32, 8, 1,  1,  1, 1,  1,1 ] #MT16x16\n\
            # - [32, 32, 8, 1,  1,  1, 1,  1,2 ] #1#MT16x32\n\
            # - [32, 32, 8, 1,  1,  1, 2,  1,1 ] #1#MT16x32X\n\
  #           - V[16, 16,16, 1,  1,  1, 3,  1,1 ] #1MT16x48X\n\
  #           #V- [16, 16,16, 1,  1,  1, 4,  1,1 ] #1MT16x64X\n\
            # - [32, 32, 8, 1,  1,  1, 1,  1,4 ] #1##MT16x64\n\
            # - [16, 16,16, 1,  1,  1, 2,  1,2 ] #1##MT16x64?X\n\
            # - [16, 16,16, 1,  1,  1, 3,  1,2 ] #1#MT16x96\n\
            # - [16, 16,16, 1,  1,  1, 2,  1,4 ] #1#MT16x128\n\
            # - [16, 16,16, 1,  1,  1, 4,  1,2 ] #1#MT16x128?X\n\
            # - [16, 16,16, 1,  1,  1, 3,  1,4 ] #1#MT16x192\n\
            # - [16, 16,16, 1,  1,  1, 4,  1,4 ] #1#MT16x256??X\n\
            # - [16, 16,16, 1,  1,  1, 1,  2,1 ] #MT32x16V\n\
            # - [16, 16,16, 1,  1,  1, 1,  2,2 ] #1MT32x32V\n\
            # - [16, 16,16, 1,  1,  1, 2,  2,1 ] #1MT32x32??\n\
            # - [16, 16,16, 1,  1,  1, 3,  2,1 ] #2#MT32x48?\n\
  #           - V[16, 16,16, 1,  1,  2, 3,  1,1 ] #MT32x48V\n\
            # - [16, 16,16, 1,  1,  1, 2,  2,2 ] #2#MT32x64\n\
            # - [16, 16,16, 1,  1,  2, 1,  1,4 ] #2#MT32x64?\n\
            # - [16, 16,16, 1,  1,  1, 4,  2,1 ] #2#MT32x64??\n\
  #           - V[16, 16,16, 1,  1,  2, 3,  1,2 ] #2#MT32x96\n\
            # - [16, 16,16, 1,  1,  1, 3,  2,2 ] #2#MT32x96?\n\
  #           - V[16, 16,16, 1,  1,  2, 4,  1,4 ] #MT32x96?\n\
            # - [16, 16,16, 1,  1,  2, 2,  1,4 ] #2#MT32x128\n\
            # - [16, 16,16, 1,  1,  1, 4,  2,2 ] #2#MT32x128?\n\
  #           - V[16, 16,16, 1,  1,  2, 3,  1,4 ] #2#MT32x192\n\
  #           - V[16, 16,16, 1,  1,  1, 6,  2,2 ] #2#MT32x192?\n\
  #           - V[16, 16,16, 1,  1,  3, 1,  1,1 ] #MT48x16X\n\
  #          - [32, 32, 8, 1,  1,  3, 1,  1,2 ] #MT48x32\n\
  #           - V[16, 16,16, 1,  1,  3, 2,  1,1 ] #MT48x32X\n\
  #           - V[16, 16,16, 1,  1,  3, 2,  1,2 ] #MT48x64\n\
  #          - [32, 32, 8, 1,  1,  3, 1,  1,4 ] #3#MT48x64\n\
  #           - V[16, 16,16, 1,  1,  3, 3,  1,2 ] #MT48x96?X\n\
  #           - V[16, 16,16, 1,  1,  3, 2,  1,4 ] #MT48x128\n\
  #           - V[16, 16,16, 1,  1,  3, 4,  1,2 ] #MT48x128?X\n\
  #           - [16, 16,16, 1,  1,  1, 1,  4,1 ] #MT64x16\n\
  #           - [16, 16,16, 1,  1,  2, 1,  2,2 ] #MT64x32\n\
  #           - [16, 16,16, 1,  1,  2, 2,  2,1 ] #MT64x32?X\n\
  #           - [16, 16,16, 1,  1,  1, 2,  4,1 ] #MT64x32??\n\
  #           - V[16, 16,16, 1,  1,  2, 3,  2,1 ] #MT64x48V\n\
  #           - [16, 16,16, 1,  1,  1, 3,  4,1 ] #MT64x48?\n\
  #           - [16, 16,16, 1,  1,  2, 2,  2,2 ] #1MT64x64V\n\
  #           - [16, 16,16, 1,  1,  1, 4,  4,1 ] #3MT64x64?\n\
  #           - [16, 16,16, 1,  1,  4, 1,  1,4 ] #3MT64x64?\n\
  #           - V[16, 16,16, 1,  1,  2, 3,  2,2 ] #2#MT64x96\n\
  #           - V[16, 16,16, 1,  1,  4, 3,  1,2 ] #2#MT64x96?X\n\
  #           - V[16, 16,16, 1,  1,  2, 4,  2,2 ] #2#MT64x128\n\
  #           - V[16, 16,16, 1,  1,  4, 2,  1,4 ] #2#MT64x128?\n\
  #           - V[16, 16,16, 1,  1,  4, 3,  1,4 ] #1MT64x192\n\
  #           - V[16, 16,16, 1,  1,  2, 6,  2,2 ] #1MT64x192?\n\
  #           - [16, 16,16, 1,  1,  3, 1,  2,1 ] #MT96x16?\n\
  #           - V[16, 16,16, 1,  1,  3, 2,  2,1 ] #MT96x32\n\
  #           - V[16, 16,16, 1,  1,  3, 2,  2,2 ] #MT96x64\n\
  #           - V[16, 16,16, 1,  1,  3, 3,  2,1 ] #MT96x48\n\
  #           - [16, 16,16, 1,  1,  3, 1,  2,2 ] #MT96x32\n\
  #           - V[16, 16,16, 1,  1,  3, 3,  2,2 ] #MT96x96\n\
  #           - [16, 16,16, 1,  1,  2, 1,  4,1 ] #MT128x16\n\
  #           - [16, 16,16, 1,  1,  2, 2,  4,1 ] #MT128x32\n\
  #           - [16, 16,16, 1,  1,  4, 1,  2,2 ] #MT128x32?\n\
  #           - V[16, 16,16, 1,  1,  2, 3,  4,1 ] #MT128x48V\n\
  #           - V[16, 16,16, 1,  1,  2, 4,  4,1 ] #MT128x64?\n\
  #           - V[16, 16,16, 1,  1,  4, 2,  2,2 ] #MT128x64?\n\
  #           - V[16, 16,16, 1,  1,  4, 3,  2,2 ] #MT128x96?\n\
  #           - [16, 16,16, 1,  1,  3, 1,  4,1 ] #MT192x16\n\
  #           - [16, 16,16, 1,  1,  3, 2,  4,1 ] #MT192x32\n\
  #           - [16, 16,16, 1,  1,  4, 1,  4,1 ] #MT256x16?\n\
  # ##############################################################################\n\
  #           - [16, 16,16, 1,  1,  3, 3,  4,1 ] #MT192x48?\n\
  #           - [16, 16,16, 1,  1,  2, 1,  1,1 ] #MT32x16?\n\
  #           - [16, 16,16, 1,  1,  2, 1,  1,2 ] #MT32x32?\n\
  #           - [16, 16,16, 1,  1,  2, 1,  2,1 ] #MT64x16?\n\
  #           - [16, 16,16, 1,  1,  2, 2,  1,1 ] #MT32x32?\n\
  #           - [16, 16,16, 1,  1,  2, 2,  1,2 ] #MT32x64?\n\
  #           - [16, 16,16, 1,  1,  2, 4,  1,1 ] #MT32x64?\n\
  #           - [16, 16,16, 1,  1,  2, 4,  1,2 ] #MT32x96?\n\
  #           - [16, 16,16, 1,  1,  2, 4,  2,1 ] #MT64x64?\n\
  #           - [16, 16,16, 1,  1,  3, 3,  1,1 ] #MT48x48?\n\
  #           - [16, 16,16, 1,  1,  3, 3,  1,4 ] #MT48x192?\n\
  #           - [16, 16,16, 1,  1,  3, 4,  1,1 ] #MT48x64?\n\
  #           - [16, 16,16, 1,  1,  3, 4,  1,4 ] #MT48x256?\n\
  #           - [16, 16,16, 1,  1,  3, 4,  2,1 ] #MT96x64?\n\
  #           - [16, 16,16, 1,  1,  3, 4,  2,2 ] #MT96x128?\n\
  #           - [16, 16,16, 1,  1,  3, 4,  4,1 ] #MT192x64?\n\
  #           - [16, 16,16, 1,  1,  4, 1,  1,1 ] #MT64x16?\n\
  #           - [16, 16,16, 1,  1,  4, 1,  1,2 ] #MT64x32?\n\
  #           - [16, 16,16, 1,  1,  4, 1,  2,1 ] #MT128x16?\n\
  #           - [16, 16,16, 1,  1,  4, 2,  1,1 ] #MT192x32?\n\
  #           - [16, 16,16, 1,  1,  4, 2,  1,2 ] #MT64x64?\n\
  #           - [16, 16,16, 1,  1,  4, 2,  2,1 ] #MT128x32?\n\
  #           - [16, 16,16, 1,  1,  4, 2,  4,1 ] #MT256x32?\n\
  #           - [16, 16,16, 1,  1,  4, 3,  1,1 ] #MT64x48?\n\
  #           - [16, 16,16, 1,  1,  4, 3,  2,1 ] #MT128x48?\n\
  #           - [16, 16,16, 1,  1,  4, 3,  4,1 ] #MT256x48?\n\
  #           - [16, 16,16, 1,  1,  4, 4,  1,1 ] #MT64x64?\n\
  #           - [16, 16,16, 1,  1,  4, 4,  1,2 ] #MT64x128?\n\
  #           - [16, 16,16, 1,  1,  4, 4,  1,4 ] #MT64x356?\n\
  #           - [16, 16,16, 1,  1,  4, 4,  2,1 ] #MT128x64?\n\
  #           - [16, 16,16, 1,  1,  4, 4,  2,2 ] #MT128x128?\n\
  #           - [16, 16,16, 1,  1,  4, 4,  4,1 ] #MT256x64?\n\
          # - AssertFree0ElementMultiple: [4,8]\n\
          - AssertFree0ElementMultiple: ["+str(AssertFree0ElementMultiple)+"]\n\
          - PrefetchGlobalRead: [2] #1\n\
          # - PrefetchLocalRead: [3,5]\n\
          - PrefetchLocalRead: [1,3,5,9]\n\
          # - PrefetchLocalRead: [5,9]\n\
          # - PrefetchLocalRead: ["+str(PrefetchLocalRead)+"]\n\
          # - DepthU: [64,96]\n\
          - DepthU: ["+str(DepthU)+"]\n\
          # - DepthU: [16,32,64]\n\
          # - VectorWidth: [4]\n\
          # - VectorWidth: [1,2]\n\
          # - VectorWidth: ["+str(VectorWidth)+"]\n\
          - VectorWidth: [-1,2,4,8]\n\
          # - GlobalReadVectorWidth: [8]\n\
          # - GlobalReadVectorWidth: ["+str(GlobalReadVectorWidth)+"]\n\
          - GlobalReadVectorWidth: [-1,2,4,8]\n\
          - LocalReadVectorWidth: [-1,2,4,8]\n\
          # - LocalReadVectorWidth: ["+str(LocalReadVectorWidth)+"]\n\
          # - LocalReadVectorWidth: [4]\n\
          - ScheduleIterAlg: [3]\n\
          - InnerUnroll: [1]\n\
          - ExpandPointerSwap: [0]\n\
          - TransposeLDS: ["+str(TransposeLDS)+"] #NN\n\
          # - TransposeLDS: [0] #NT\n\
          - LdsBlockSizePerPad: [-1]\n\
          - LdsPadA: [-1]\n\
          - LdsPadB: [-1]\n"

  if 1:#(StaggerUStride) == '0':
    yaml = yaml+"          - StaggerUStride: [128,256]\n"
  # elif (StaggerUStride) == 'jk':
  #   yaml = yaml+"          - StaggerUStride: [128,256]\n"
  else:
    yaml = yaml+"          - StaggerUStride: ["+str(StaggerUStride)+"]\n"

  if 0:#(StaggerU) != 'ijk':
    yaml = yaml+"          - StaggerU: ["+str(StaggerU)+"]\n"
  else:
    yaml = yaml+"          - StaggerU: [0,4,32]\n"

  yaml = yaml+"          - WorkGroupMapping: [1,4,8,16,32,64,110]\n\
          # - WorkGroupMapping: ["+str(WorkGroupMapping)+"]\n\
          # - WorkGroupMapping: [1]\n\
          # - StaggerUMapping: [0,3]\n\
          # - WaveSeparateGlobalReadA: ["+str(WaveSeparateGlobalReadA)+"]\n\
          # - WaveSeparateGlobalReadA: [0,1]\n\
          # - WaveSeparateGlobalReadB: [0,1]\n\
          # - WaveSeparateGlobalReadB: ["+str(WaveSeparateGlobalReadB)+"]\n\
          # - MaxOccupancy: [40]\n\
          # - 1LDSBuffer: [0, 1]\n\
          - 1LDSBuffer: ["+str(LDSBuffer)+"]\n\
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
          - StoreVectorWidth: [-1,2,4,8]\n\
          # - StoreVectorWidth: ["+str(StoreVectorWidth)+"]\n\
          # - StoreVectorWidth: [1]\n\
          - SourceSwap: [0, 1]\n\
          # - SourceSwap: ["+str(SourceSwap)+"]\n\
          # - SourceSwap: [1]\n\
          # - NumElementsPerBatchStore: [2]\n\
          # - NumElementsPerBatchStore: [0, 2]\n\
          # - NumElementsPerBatchStore: [2]\n\
          - StorePriorityOpt: [1]\n\
          # - StorePriorityOpt: [0,1]X\n\
          # - NumLoadsCoalescedA: [1]\n\
          # - NumLoadsCoalescedA: [1,3]X\n\
          # - StoreRemapVectorWidth: ["+str(StoreRemapVectorWidth)+"]\n\
          - StoreRemapVectorWidth: [0,-1]\n\
        BenchmarkJoinParameters:\n\
        BenchmarkFinalParameters:\n\
          - ProblemSizes:\n\
            - Exact: ["+str(listproblemSize[0])+", "+str(listproblemSize[1])+", "+str(listproblemSize[2])+", "+str(listproblemSize[3])+"]\n\
          - ActivationArgs:\n\
            - [Enum: none]\n\
          - BiasTypeArgs: ['s']\n\
  LibraryLogic:\n\
      ScheduleName: \"aldebaran\"\n\
      DeviceNames: [\"Device 0050\", \"Device 0051\", \"Device 0052\", \"Device 0054\", \"Device 0062\", \"Device 7400\", \"Device 740c\"]\n\
      ArchitectureName: \"gfx90a\"\n\
  \n\
  "
  file1.write(yaml)

