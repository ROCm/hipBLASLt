print("HELLO")
import math
from operator import itemgetter

from itertools import combinations
from itertools import product
from itertools import combinations_with_replacement
from itertools import permutations

WT = [[1,1],[1,2],[1,4],[2,1],[2,2],[4,1]]
# TT0 = list(permutations("1234",1))
# TT1 = list(permutations("1234",1))
TT0 = [i for i in range(1,5)]
TT1 = [1,2,3,4]
TTWT = list(product(TT0,TT1,WT))
print(TT0)
print(TTWT)
print(len(TTWT))

# for i in TTWT:
#     print(str(i))

CUs = 304
MinKForGSU = 128 #define at Common.py
WorkspaceCheck = 33554432

sizes = [
			[1104, 1, 1, 4608],
			[1104, 16, 1, 4608]
		]

MI = [
		# [16, 16,16, 1, 1, 1, 1, 1,1 ], 
		# [16, 16,16, 1, 1, 1, 2, 4,1 ],
		# [16, 16,16, 1, 1, 1, 1, 4,1 ]
	]

print(MI)

# for i in range(len(TTWT)):
#     # create a new list
#     lst = [16, 16, 16, 1, 1] + [TTWT[i][0]] + [TTWT[i][1]] + TTWT[i][2]
#     # append new list to nested list
#     MI.append(lst)

# for i in range(len(TTWT)):
#     # create a new list
#     lst = [32, 32, 8, 1, 1] + [TTWT[i][0]] + [TTWT[i][1]] + TTWT[i][2]
#     # append new list to nested list
#     MI.append(lst)

# print(MI)

for (i, item) in enumerate(MI, start=1):
    print(i, item)

# GSU = [1,2,3,4,5,6]
GSU = [i for i in range(1, CUs+1)]
# GSU = [1]

def prechoose(problist, MI16, MI32):
	print("MI16,MI32:",MI16,MI32)
	if int(MI16):
		for i in range(len(TTWT)):
		    # create a new list
		    lst = [16, 16, 16, 1, 1] + [TTWT[i][0]] + [TTWT[i][1]] + TTWT[i][2]
		    # append new list to nested list
		    MI.append(lst)

	if int(MI32):
		for i in range(len(TTWT)):
		    # create a new list
		    lst = [32, 32, 8, 1, 1] + [TTWT[i][0]] + [TTWT[i][1]] + TTWT[i][2]
		    # append new list to nested list
		    MI.append(lst)

	print(MI)

	global GSU
	print("len(sizes)", len(sizes))
	print("len(MI)", len(MI))
	# print("GSU", GSU)
	# print("problist", problist)

	idx = 0
	# for i in sizes:
	i = problist[0]

	if int(MI16):
		if 0: #(math.ceil(i[0]/(16*4))*math.ceil(i[1]/(16*4))/CUs) >= 1.0:
			noGSU = [1]
			GSU = noGSU
			print("no GSU", GSU, math.ceil(i[0]/(16*4)), math.ceil(i[1]/(16*4)))
			IFGSU = 0
		else:
			GSUTOP = int(i[3]/MinKForGSU)
			GSUTOPafterCheck = int(WorkspaceCheck/(i[0]*i[1]*4)) # *4 = bytes per elements
			if GSUTOP > GSUTOPafterCheck:
				GSUTOP = max(1, GSUTOPafterCheck)
			MinKGSU = [i for i in range(1, GSUTOP+1)]
			GSU = MinKGSU
			print("GSU", GSU, math.ceil(i[0]/(16*4)), math.ceil(i[1]/(16*4)))
			IFGSU = 1

	if int(MI32):
		if 0: #(math.ceil(i[0]/(32*4))*math.ceil(i[1]/(32*4))/CUs) >= 1.0:
			noGSU = [1]
			GSU = noGSU
			print("no GSU", GSU, math.ceil(i[0]/(32*4)), math.ceil(i[1]/(32*4)))
			IFGSU = 0
		else:
			GSUTOP = int(i[3]/MinKForGSU)
			GSUTOPafterCheck = int(WorkspaceCheck/(i[0]*i[1]*4)) # *4 = bytes per elements
			if GSUTOP > GSUTOPafterCheck:
				GSUTOP = max(1, GSUTOPafterCheck)
			MinKGSU = [i for i in range(1, GSUTOP+1)]
			GSU = MinKGSU
			print("GSU", GSU, math.ceil(i[0]/(32*4)), math.ceil(i[1]/(32*4)))
			IFGSU = 1

	listx = []
	# for solutionidx in range(0, len(sizes)*len(MI)*len(GSU)):
	for solutionidx in range(0, len(MI)*len(GSU)):
		# print("solutionidx", solutionidx)
		listin = []
		listx.append(listin)

	for j in MI:
		for k in GSU:
			# print("tiles0", (math.ceil(i[0]/(j[0]*j[5]*j[7]))))
			# print("tiles1", (math.ceil(i[1]/(j[1]*j[6]*j[8]))*k))
			# print("tiles", (math.ceil(i[0]/(j[0]*j[5]*j[7]))*math.ceil(i[1]/(j[1]*j[6]*j[8]))*k))
			# tiles_Cus = ((i[0]/(j[0]*j[5]*j[7]))*(i[1]/(j[1]*j[6]*j[8]))*k)/CUs
			if IFGSU and (j[5]*j[6] > 2): # because WT not good at GSU use mode
				k = 1
			tiles_Cus = (math.ceil(i[0]/(j[0]*j[5]*j[7]))*math.ceil(i[1]/(j[1]*j[6]*j[8]))*k)/CUs
			# print("tiles/CUs", tiles_Cus)
			tiles_Cus_absToOne = tiles_Cus

			if tiles_Cus_absToOne >= 1:
				while tiles_Cus_absToOne >= 1 and (IFGSU and tiles_Cus_absToOne <=4):
					tiles_Cus_absToOne = abs(1-tiles_Cus_absToOne)
				if tiles_Cus_absToOne > 0.5:
					tiles_Cus_absToOne = abs(1-tiles_Cus_absToOne)
			else:
				tiles_Cus_absToOne = abs(1-tiles_Cus_absToOne)
			Ceil_tiles_Cus = math.ceil(tiles_Cus)
			# print("ceil(tiles/CUs)", Ceil_tiles_Cus)
			guanality = tiles_Cus / Ceil_tiles_Cus
			# print("(tiles/CUs)/ceil(tiles/CUs)", guanality)
			density = (i[0]*i[1])/((j[0]*j[5]*j[7])*(j[1]*j[6]*j[8])*math.ceil(i[0]/(j[0]*j[5]*j[7]))*math.ceil(i[1]/(j[1]*j[6]*j[8])))
			# print("density", density)
			densityxguanality = density*guanality

			# print("tiles2", ((i[0]/(j[0]*j[5]*j[7]))*(i[1]/(j[1]*j[6]*j[8]))*k))
			tiles_Cus2 = ((i[0]/(j[0]*j[5]*j[7]))*(i[1]/(j[1]*j[6]*j[8]))*k)/CUs
			# print("tiles/CUs2", tiles_Cus2)
			Ceil_tiles_Cus2 = math.ceil(((i[0]/(j[0]*j[5]*j[7]))*(i[1]/(j[1]*j[6]*j[8]))*k)/CUs)
			# print("ceil(tiles/CUs)2", Ceil_tiles_Cus2)
			guanality2 = tiles_Cus2 / Ceil_tiles_Cus2
			# print("(tiles/CUs)/ceil(tiles/CUs)2", guanality2)
			listx[idx] = [i,j,k,densityxguanality, density, tiles_Cus_absToOne, tiles_Cus, guanality, guanality2]
			# print("list[idx]", listx[idx])
			idx = idx+1
	# print("listx", listx)

	while [] in listx:
		listx.remove([])
	# print("listx", listx)
	# for i in listx:
	# 	# print("Victor", i)
	# 	if i == []:
	# 		print("Victor", i)
	# 		listx.remove(i)

	# if IFGSU:
	if 0:
		listx.sort(key=itemgetter(3), reverse=True)
	else:
		listx.sort(key=itemgetter(5))
	# print("listx after sort", listx)

	for idx in listx:
		print(listx.index(idx), idx)

	return listx

# prechoose()

# print(list(map(lambda x: x[3], listx)))
# max_key = max(list, key=lambda key: list[][2])
# print("max_key", list[][2])

# print("tiles", (math.ceil(sizes[1][0]/(MI[1][0]*MI[1][5]*MI[1][7]))*math.ceil(sizes[1][1]/(MI[1][1]*MI[1][6]*MI[1][8]))*GSU[5]))

# import pandas as pd

# # df = pd.read_csv('tensilelite/try/tunning_BFVF_1/2_BenchmarkData/Cijk_Ailk_Bljk_HHS_BH_00_CSVWinner.csv', encoding='gbk')


# # print(df[' WinnerTimeUS'][0])

# # dic = {}

# # dic[0] = df[' WinnerTimeUS'][0]

# # df = pd.read_csv('tensilelite/try/tunning_BFVF_2/2_BenchmarkData/Cijk_Ailk_Bljk_HHS_BH_00_CSVWinner.csv', encoding='gbk')


# # print(df[' WinnerTimeUS'][0])

# # dic[1] = df[' WinnerTimeUS'][0]

# # print(dic[1])

# problemnum = 12

# list = []
# for problemidx in range(0, problemnum):
# 	dic = {}
# 	list.append(dic)

# ProblemList = []
# WinnerKernelList = [[] for _ in range(problemnum)]
# WinnerKernelTime = [[] for _ in range(problemnum)]

# # print(WinnerKernelList)

# idx = 1
# df = pd.read_csv('/home/victorwu/hipBLASLt/tensilelite/try/yamlGen/tunning_BFVF_Round1/tunning_BFVF_Round1_'+str(idx)+'/2_BenchmarkData/Cijk_Ailk_Bljk_HHS_BH_00_CSVWinner.csv', encoding='gbk')
# for problemidx in range(0, problemnum):
# 	ProblemList.append(str(df[' SizeI'][problemidx])+", "+str(df[' SizeJ'][problemidx])+", "+str(df[' SizeK'][problemidx])+", "+str(df[' SizeL'][problemidx]))
	
# for idx in range(1,23+1):
# 	df = pd.read_csv('/home/victorwu/hipBLASLt/tensilelite/try/yamlGen/tunning_BFVF_Round1/tunning_BFVF_Round1_'+str(idx)+'/2_BenchmarkData/Cijk_Ailk_Bljk_HHS_BH_00_CSVWinner.csv', encoding='gbk')
# 	for problemidx in range(0, problemnum):
# 		# print(df[' WinnerTimeUS'][problemidx])
# 		list[problemidx][idx] = df[' WinnerGFlops'][problemidx]
# 		WinnerKernelList[problemidx].append(str(df[' WinnerName'][problemidx]))
# 		WinnerKernelTime[problemidx].append(str(df[' WinnerTimeUS'][problemidx]))

# # for problemidx in range(0, problemnum):
# # 	print(WinnerKernelList[problemidx])

# dfout = pd.DataFrame(columns=['problemsize', 'gflops', 'time', 'kernelname'])

# for problemidx in range(0, problemnum):
# 	# print(list[problemidx])
# 	print(problemidx, ProblemList[problemidx])
# 	max_key = max(list[problemidx], key=lambda key: list[problemidx][key])
# 	print(max_key, list[problemidx][max_key])
# 	print(WinnerKernelTime[problemidx][max_key-1], WinnerKernelList[problemidx][max_key-1])
# 	print('\n')
# 	# dfout = dfout.append({	'problemsize':ProblemList[problemidx],
# 	# 				'gflops':list[problemidx][max_key],
# 	# 				'time':WinnerKernelTime[problemidx][max_key-1],
# 	# 				'kernelname':str(WinnerKernelList[problemidx][max_key-1])}, ignore_index=True)

# 	df1 = pd.DataFrame({'problemsize': [ProblemList[problemidx]],
#                     'gflops': [list[problemidx][max_key]],
#                     'time': [WinnerKernelTime[problemidx][max_key-1]],
#                     'kernelname': [str(WinnerKernelList[problemidx][max_key-1])]},
#                    columns=['problemsize', 'gflops', 'time', 'kernelname'])
# 	print(df1)
# 	dfout = pd.concat([dfout, df1], ignore_index=True)
# 	print(dfout)

# print(dfout)
# dfout.to_csv('/home/victorwu/hipBLASLt/tensilelite/try/yamlGen/chcek.csv', encoding='gbk', index=False)
# # max_key = max(list[1], key=lambda key: list[1][key])
# # print(max_key)

# # print(list)

