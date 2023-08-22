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
parser.add_argument('--first', type=str, default = None)
args = parser.parse_args()
print("args.prob",args.prob)
dirpath = "/hipBLASLt/tensilelite/try/yamlGen/"+args.prob+"/"
# print(df[' WinnerTimeUS'][0])

# dic[1] = df[' WinnerTimeUS'][0]

# print(dic[1])

problemnum = 1

list = []
for problemidx in range(0, problemnum):
	dic = {}
	list.append(dic)

ProblemList = []
WinnerKernelList = [[] for _ in range(problemnum)]
WinnerKernelTime = [[] for _ in range(problemnum)]

# print(WinnerKernelList)

idx = 1
df = pd.read_csv(dirpath+'tunning_BFVF_Round1/tunning_BFVF_Round1_'+str(idx)+'/2_BenchmarkData/Cijk_Ailk_Bljk_HHS_BH_00_CSVWinner.csv', encoding='gbk')
for problemidx in range(0, problemnum):
	ProblemList.append(str(df[' SizeI'][problemidx])+", "+str(df[' SizeJ'][problemidx])+", "+str(df[' SizeK'][problemidx])+", "+str(df[' SizeL'][problemidx]))
	
for idx in range(1,int(int(args.first)/2)+1):
	# print("args.first", args.first)
	df = pd.read_csv(dirpath+'tunning_BFVF_Round1/tunning_BFVF_Round1_'+str(idx)+'/2_BenchmarkData/Cijk_Ailk_Bljk_HHS_BH_00_CSVWinner.csv', encoding='gbk')
	for problemidx in range(0, problemnum):
		# print(df[' WinnerTimeUS'][problemidx])
		list[problemidx][idx] = df[' WinnerGFlops'][problemidx]
		WinnerKernelList[problemidx].append(str(df[' WinnerName'][problemidx]))
		WinnerKernelTime[problemidx].append(str(df[' WinnerTimeUS'][problemidx]))
		print(WinnerKernelList[problemidx][idx-1], WinnerKernelTime[problemidx][idx-1])

# for problemidx in range(0, problemnum):
# 	print(WinnerKernelList[problemidx])

dfout = pd.DataFrame(columns=['problemsize', 'gflops', 'time', 'kernelname'])

for problemidx in range(0, problemnum):
	# print(list[problemidx])
	print(problemidx, ProblemList[problemidx])
	max_key = max(list[problemidx], key=lambda key: list[problemidx][key])
	print(max_key, list[problemidx][max_key])
	print(WinnerKernelTime[problemidx][max_key-1], WinnerKernelList[problemidx][max_key-1])
	print('\n')
	# dfout = dfout.append({	'problemsize':ProblemList[problemidx],
	# 				'gflops':list[problemidx][max_key],
	# 				'time':WinnerKernelTime[problemidx][max_key-1],
	# 				'kernelname':str(WinnerKernelList[problemidx][max_key-1])}, ignore_index=True)

	df1 = pd.DataFrame({'problemsize': [ProblemList[problemidx]],
                    'gflops': [list[problemidx][max_key]],
                    'time': [WinnerKernelTime[problemidx][max_key-1]],
                    'kernelname': [str(WinnerKernelList[problemidx][max_key-1])]},
                   columns=['problemsize', 'gflops', 'time', 'kernelname'])
	print(df1)
	dfout = pd.concat([dfout, df1], ignore_index=True)
	print(dfout)

print(dfout)
dfout.to_csv(dirpath+'chcek.csv', encoding='gbk', index=False)
# max_key = max(list[1], key=lambda key: list[1][key])
# print(max_key)

# print(list)

