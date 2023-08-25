import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--prob', type=str, default = None)
parser.add_argument('--first', type=str, default = None)
parser.add_argument('--justPrechoose', type=str, default = None)
parser.add_argument('--homepath', type=str, default = None)
parser.add_argument('--MI16', type=str, default = None)
parser.add_argument('--MI32', type=str, default = None)
args = parser.parse_args()
print(args.prob)

dirpath = args.homepath+"tensilelite/try/yamlGen/"
print("HELLORound1 try")

from prechoose import prechoose

start = 0
# print(MT[start_MT1:])
end = start + args.prob[start:].find('_')
M = args.prob[start:end]
# print(start)
# print(end)
print("M",M)

start = end+1
# print(MT[start_MT1:])
end = start + args.prob[start:].find('_')
N = args.prob[start:end]
# print(start)
# print(end)
print("N",N)

start = end+1
# print(MT[start_MT1:])
end = start + args.prob[start:].find('_')
B = args.prob[start:end]
# print(start)
# print(end)
print("B",B)

start = end+1
# print(MT[start_MT1:])
# end = start + args.prob[start:].find('_')
K = args.prob[start:]
# print(start)
# print(end)
print("K",K)

problist = [
[int(M), int(N), int(B), int(K)]
]

print("args.MI16.MI32:",args.MI16,args.MI32)

prelist = prechoose(problist, args.MI16, args.MI32)

print("prelist",prelist)

list = [
        "- "+str(i[1])+""for i in prelist
]

GSUlist = [
        str(i[2])for i in prelist
]

# print("problist: ", problist)
# print("list: ", list)
# print("GSUlist: ", GSUlist)

import pandas as pd

# list = [
#           "- [16, 16,16, 1,  1,  1, 1,  1,1 ] #MT16x16",
#           "- [16, 16,16, 1,  1,  1, 1,  1,2 ] #1#MT16x32",
#           "- [16, 16,16, 1,  1,  1, 2,  1,1 ] #1#MT16x32X",
#           "- [16, 16,16, 1,  1,  1, 1,  1,4 ] #1##MT16x64",
#           "- [16, 16,16, 1,  1,  1, 2,  1,2 ] #1##MT16x64?X",
#           "- [16, 16,16, 1,  1,  1, 3,  1,2 ] #1#MT16x96",
#           "- [16, 16,16, 1,  1,  1, 2,  1,4 ] #1#MT16x128",
#           "- [16, 16,16, 1,  1,  1, 4,  1,2 ] #1#MT16x128?X",
#           "- [16, 16,16, 1,  1,  1, 3,  1,4 ] #1#MT16x192",
#           "- [16, 16,16, 1,  1,  1, 4,  1,4 ] #1#MT16x256??X",
#           "- [16, 16,16, 1,  1,  1, 1,  2,1 ] #MT32x16V",
#           "- [16, 16,16, 1,  1,  1, 1,  2,2 ] #1MT32x32V",
#           "- [16, 16,16, 1,  1,  1, 2,  2,1 ] #1MT32x32??",
#           "- [16, 16,16, 1,  1,  1, 3,  2,1 ] #2#MT32x48?",
#           "- [16, 16,16, 1,  1,  1, 2,  2,2 ] #2#MT32x64",
#           "- [16, 16,16, 1,  1,  2, 1,  1,4 ] #2#MT32x64?",
#           "- [16, 16,16, 1,  1,  1, 4,  2,1 ] #2#MT32x64??",
#           "- [16, 16,16, 1,  1,  1, 3,  2,2 ] #2#MT32x96?",
#           "- [16, 16,16, 1,  1,  2, 2,  1,4 ] #2#MT32x128",
#           "- [16, 16,16, 1,  1,  1, 4,  2,2 ] #2#MT32x128?",
#           "- [16, 16,16, 1,  1,  2, 3,  1,4 ] #2#MT32x192",
#           "- [16, 16,16, 1,  1,  3, 1,  1,2 ] #MT48x32",
#           "- [16, 16,16, 1,  1,  3, 1,  1,4 ] #3#MT48x64",
#           "- [16, 16,16, 1,  1,  1, 1,  4,1 ] #MT64x16",
#           "- [16, 16,16, 1,  1,  2, 1,  2,2 ] #MT64x32",
#           "- [16, 16,16, 1,  1,  2, 2,  2,1 ] #MT64x32?X",
#           "- [16, 16,16, 1,  1,  1, 2,  4,1 ] #MT64x32??",
#           "- [16, 16,16, 1,  1,  1, 3,  4,1 ] #MT64x48?",
#           "- [16, 16,16, 1,  1,  2, 2,  2,2 ] #1MT64x64V",
#           "- [16, 16,16, 1,  1,  1, 4,  4,1 ] #3MT64x64?",
#           "- [16, 16,16, 1,  1,  4, 1,  1,4 ] #3MT64x64?",
#           "- [16, 16,16, 1,  1,  2, 3,  2,2 ] #2#MT64x96 #22",
#           "- [16, 16,16, 1,  1,  2, 4,  2,2 ] #2#MT64x128 #21",
#           "- [16, 16,16, 1,  1,  4, 2,  1,4 ] #2#MT64x128? #21",
#           "- [16, 16,16, 1,  1,  3, 1,  2,1 ] #MT96x16?",
#           "- [16, 16,16, 1,  1,  3, 2,  2,2 ] #MT96x64 #22",
#           "- [16, 16,16, 1,  1,  3, 1,  2,2 ] #MT96x32",
#           "- [16, 16,16, 1,  1,  2, 1,  4,1 ] #MT128x16",
#           "- [16, 16,16, 1,  1,  2, 2,  4,1 ] #MT128x32",
#           "- [16, 16,16, 1,  1,  4, 1,  2,2 ] #MT128x32?",
#           "- [16, 16,16, 1,  1,  2, 3,  4,1 ] #MT128x48V",
#           "- [16, 16,16, 1,  1,  2, 4,  4,1 ] #MT128x64? #20",
#           "- [16, 16,16, 1,  1,  4, 2,  2,2 ] #MT128x64? #20",
#           "- [16, 16,16, 1,  1,  3, 1,  4,1 ] #MT192x16",
#           "- [16, 16,16, 1,  1,  3, 2,  4,1 ] #MT192x32",
#           "- [16, 16,16, 1,  1,  4, 1,  4,1 ] #MT256x16?",
# ]

print("args.justPrechoose:",args.justPrechoose)

if not int(args.justPrechoose):
  print("args.justPrechoose:",args.justPrechoose)
  import os

  for probidx in range(1,len(problist)+1):
    # for idx in range(1,int(len(list)/2)+1):
    for idx in range(1,int(int(args.first)/2)+1):
      # print(probidx)
      filename = dirpath+str(problist[probidx-1][0])+"_"+str(problist[probidx-1][1])+"_"+str(problist[probidx-1][2])+"_"+str(problist[probidx-1][3])+"/FP16_NN_MI250X_"+str(idx)+".yaml"
      dirname = os.path.dirname(filename)
      print("dirname: ", dirname)
      if not os.path.exists(dirname):
          os.makedirs(dirname)
      file1 = open(filename,"w")

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
      Device: 0\n\
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
          UseBias: True\n\
          Activation: True\n\
          ActivationHPA: True\n\
          UseScaleDVec: True\n\
          UseScaleAlphaVec: True\n\
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
            - DepthU: [16,32,64,128]\n\
            # - DepthU: [16,32,64]\n\
            # - VectorWidth: [-1,2,4,8]\n\
            # - VectorWidth: [-1]\n\
            # - GlobalReadVectorWidth: [-1,2,4,8]\n\
            - VectorWidthA: [-1]\n\
            - VectorWidthB: [-1]\n\
            - GlobalReadVectorWidthA: [-1]\n\
            - GlobalReadVectorWidthB: [-1]\n\
            # - LocalReadVectorWidth: [4,8]\n\
            - LocalReadVectorWidth: [-1,2,4,8]\n\
            # - LocalReadVectorWidth: [4]\n\
            - ScheduleIterAlg: [3]\n\
            - InnerUnroll: [1]\n\
            - ExpandPointerSwap: [0]\n\
            - TransposeLDS: [1] #NN\n\
            # - TransposeLDS: [0] #NT\n\
            - LdsBlockSizePerPadA: [-1]\n\
            - LdsBlockSizePerPadB: [-1]\n\
            - LdsPadA: [-1]\n\
            - LdsPadB: [-1]\n\
            # - StaggerUStride: [0]\n\
            # - StaggerUStride: [128,256]\n\
            # - StaggerU: [0,32]\n\
            # - StaggerU: [0,4,32]\n\
            # - WorkGroupMapping: [2]\n\
            # - WorkGroupMapping: [1,4,8,16,32,64,110]\n\
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
            # - GlobalSplitU: [1]\n\
            # - GlobalSplitU: [2,3,4,5,15,16]\n\
            # - GlobalSplitU: [6,8,9]\n\
            - GlobalSplitU: ["+GSUlist[(idx-1)*2]+", "+GSUlist[(idx-1)*2+1]+"]\n\
            - GlobalSplitUAlgorithm: [\"MultipleBuffer\"]\n\
            # - GlobalSplitUAlgorithm: [\"SingleBuffer\"]\n\
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
            - ClusterLocalRead: [1]\n\
          BenchmarkJoinParameters:\n\
          BenchmarkFinalParameters:\n\
            - ProblemSizes:\n\
              - Exact: "+str(problist[(probidx-1)])+"\n\
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
