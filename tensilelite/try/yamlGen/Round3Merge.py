print("HELLO")

import pandas as pd

# df = pd.read_csv('tensilelite/try/tunning_BFVF_1/2_BenchmarkData/Cijk_Ailk_Bljk_HHS_BH_00_CSVWinner.csv', encoding='gbk')


# print(df[' WinnerTimeUS'][0])

# dic = {}

# dic[0] = df[' WinnerTimeUS'][0]

# df = pd.read_csv('tensilelite/try/tunning_BFVF_2/2_BenchmarkData/Cijk_Ailk_Bljk_HHS_BH_00_CSVWinner.csv', encoding='gbk')
import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--prob', type=str, default = None)
parser.add_argument('--homepath', type=str, default = None)
args = parser.parse_args()
print(args.prob)
dirpath = args.homepath+"tensilelite/try/yamlGen/"+args.prob+"/"
# print(df[' WinnerTimeUS'][0])

# dic[1] = df[' WinnerTimeUS'][0]

# print(dic[1])

problemnum = 1

list = []
for problemidx in range(0, problemnum):
	dic = {}
	list.append(dic)

ProblemList = []
gflops = []
WinnerKernelList = []
WinnerKernelTime = []
# WinnerKernelList = [[] for _ in range(problemnum)]
# WinnerKernelTime = [[] for _ in range(problemnum)]

# print(WinnerKernelList)

idx = 1

for problemidx in range(0, problemnum):
	# print(problemidx)
	df = pd.read_csv(dirpath+'tunning_BFVF_Round3/tunning_BFVF_Round3_'+str(problemidx)+'/2_BenchmarkData/Cijk_Ailk_Bljk_HHS_BH_Bias_AH_SDV_SAV_00_CSVWinner.csv', encoding='gbk')
	ProblemList.append(str(df[' SizeI'][0])+", "+str(df[' SizeJ'][0])+", "+str(df[' SizeK'][0])+", "+str(df[' SizeL'][0]))
	gflops.append(str(df[' WinnerGFlops'][0]))
	WinnerKernelList.append(str(df[' WinnerName'][0]))
	WinnerKernelTime.append(str(df[' WinnerTimeUS'][0]))
	print(WinnerKernelList[problemidx][0], WinnerKernelTime[problemidx][0])

dfout = pd.DataFrame(columns=['problemsize', 'gflops', 'time', 'kernelname'])

for problemidx in range(0, problemnum):
	# print(list[problemidx])
	# print(problemidx, ProblemList[problemidx])
	# print(problemidx, gflops[problemidx])
	# print(WinnerKernelTime[problemidx], WinnerKernelList[problemidx])
	# print('\n')
	# dfout = dfout.append({	'problemsize':ProblemList[problemidx],
	# 				'gflops':gflops[problemidx],
	# 				'time':WinnerKernelTime[problemidx],
	# 				'kernelname':str(WinnerKernelList[problemidx])}, ignore_index=True)
	df1 = pd.DataFrame({'problemsize': [ProblemList[problemidx]],
                    'gflops': [gflops[problemidx]],
                    'time': [WinnerKernelTime[problemidx]],
                    'kernelname': [str(WinnerKernelList[problemidx])]},
                   columns=['problemsize', 'gflops', 'time', 'kernelname'])
	print(df1)
	dfout = pd.concat([dfout, df1], ignore_index=True)
	print(dfout)

print(dfout)
dfout.to_csv(dirpath+'chcekRound3.csv', encoding='gbk', index=False)
# max_key = max(list[1], key=lambda key: list[1][key])
# print(max_key)

# print(list)

