print("HELLO")

import pandas as pd


import re
import pickle

problemnum=23

arg = []
prob = []
startsection = ""
endsection = ""

problemidxoffset = 34+1

def Convert(problemidx, bytetype):
	global arg, prob, startsection, endsection

# for problemidx in range(0, problemnum):
	# file=open("tunning_BFVF_Round3/tunning_BFVF_Round3_"+str(idx)+"/3_LibraryLogic/aldebaran_Cijk_Ailk_Bljk_HHS_BH.yaml",'r')
	lines = None
	with open("/operator/hipBLASLt/tensilelite/try"+str(bytetype)+"/yamlGen/tunning_BFVF_Round3/tunning_BFVF_Round3_"+str(problemidx)+"/3_LibraryLogic/aldebaran_Cijk_Ailk_Bljk_HHS_BH.yaml","rt") as txtfile:
	    lines = txtfile.readlines()
	    
	berry_idx = [i for i, item in enumerate(lines) if re.search('1LDSBuffer', item)]

	if problemidx == 0:
		startsection = lines[:berry_idx[0]]
		print(startsection)
	result1 = lines[berry_idx[0]:lines.index("- [2, 3, 0, 1]\n")]

	sol_idx = [i for i, item in enumerate(result1) if re.search('SolutionIndex', item)]

	# print(lines[sol_idx[0]])
	result1[sol_idx[0]] = "    SolutionIndex: "+str(problemidx+problemidxoffset)+"\n"
	# print(result1[sol_idx[0]])
	if problemidx != 0:
		result1[0] = result1[0].replace('- - ', '  - ')
	arg = arg+result1


	berry_idx = [i for i, item in enumerate(lines) if re.search('1LDSBuffer', item)]

	result2 = lines[lines.index("- [2, 3, 0, 1]\n"):lines.index("- null\n")]

	if problemidx == 0:
		endsection = lines[lines.index("- null\n"):]
		print(endsection)

	# print(lines[sol_idx[0]])
	start = result2[2].find('[')
	# print(result2[2][:start+1])
	# print(result2[2][start+1:])
	result2[2] = result2[2][:start+1]+str(problemidx+problemidxoffset)+result2[2][start+2:]

	if problemidx != 0:
		result2.pop(0)

	if problemidx != 0:
		result2[0] = result2[0].replace('- - - ', '  - - ')
		# result2[0][0] = ' '
		# result2[0][1] = ' '
		# print(result2[0])
		# print(result2[0][1])
		# print(result2[0][2])
	prob = prob+result2
	# print(result2)




ProblemListMI32 = []
gflopsMI32 = []
WinnerKernelListMI32 = []
WinnerKernelTimeMI32 = []

for problemidx in range(0, problemnum):
	# print(problemidx)
	df = pd.read_csv('/operator/hipBLASLt/tensilelite/try32/yamlGen/tunning_BFVF_Round3/tunning_BFVF_Round3_'+str(problemidx)+'/2_BenchmarkData/Cijk_Ailk_Bljk_HHS_BH_00_CSVWinner.csv', encoding='gbk')
	ProblemListMI32.append(str(df[' SizeI'][0])+", "+str(df[' SizeJ'][0])+", "+str(df[' SizeK'][0])+", "+str(df[' SizeL'][0]))
	gflopsMI32.append(str(df[' WinnerGFlops'][0]))
	WinnerKernelListMI32.append(str(df[' WinnerName'][0]))
	WinnerKernelTimeMI32.append(str(df[' WinnerTimeUS'][0]))

dfout = pd.DataFrame(columns=['problemsize', 'gflops', 'time', 'kernelname'])

for problemidx in range(0, problemnum):
	# print(problemidx, ProblemListMI32[problemidx])
	# print(problemidx, gflopsMI32[problemidx])
	# print(WinnerKernelTimeMI32[problemidx], WinnerKernelListMI32[problemidx])
	# print('\n')
	# dfout = dfout.append({	'problemsize':ProblemListMI32[problemidx],
	# 				'gflops':gflopsMI32[problemidx],
	# 				'time':WinnerKernelTimeMI32[problemidx],
	# 				'kernelname':str(WinnerKernelListMI32[problemidx])}, ignore_index=True)
	df1 = pd.DataFrame({'problemsize': [ProblemListMI32[problemidx]],
                    'gflops': [gflopsMI32[problemidx]],
                    'time': [WinnerKernelTimeMI32[problemidx]],
                    'kernelname': [str(WinnerKernelListMI32[problemidx])]},
                   columns=['problemsize', 'gflops', 'time', 'kernelname'])
	print(df1)
	dfout = pd.concat([dfout, df1], ignore_index=True)
	print(dfout)

print(dfout)
dfout.to_csv('/operator/hipBLASLt/tensilelite/chcekRoundNNMI32.csv', encoding='gbk', index=False)

################################################################16

ProblemListMI16 = []
gflopsMI16 = []
WinnerKernelListMI16 = []
WinnerKernelTimeMI16 = []

for problemidx in range(0, problemnum):
	# print(problemidx)
	df = pd.read_csv('/operator/hipBLASLt/tensilelite/try/yamlGen/tunning_BFVF_Round3/tunning_BFVF_Round3_'+str(problemidx)+'/2_BenchmarkData/Cijk_Ailk_Bljk_HHS_BH_00_CSVWinner.csv', encoding='gbk')
	ProblemListMI16.append(str(df[' SizeI'][0])+", "+str(df[' SizeJ'][0])+", "+str(df[' SizeK'][0])+", "+str(df[' SizeL'][0]))
	gflopsMI16.append(str(df[' WinnerGFlops'][0]))
	WinnerKernelListMI16.append(str(df[' WinnerName'][0]))
	WinnerKernelTimeMI16.append(str(df[' WinnerTimeUS'][0]))

dfout = pd.DataFrame(columns=['problemsize', 'gflops', 'time', 'kernelname'])

for problemidx in range(0, problemnum):
	# print(problemidx, ProblemListMI16[problemidx])
	# print(problemidx, gflopsMI16[problemidx])
	# print(WinnerKernelTimeMI16[problemidx], WinnerKernelListMI16[problemidx])
	# print('\n')
	# dfout = dfout.append({	'problemsize':ProblemListMI16[problemidx],
	# 				'gflops':gflopsMI16[problemidx],
	# 				'time':WinnerKernelTimeMI16[problemidx],
	# 				'kernelname':str(WinnerKernelListMI16[problemidx])}, ignore_index=True)
	df1 = pd.DataFrame({'problemsize': [ProblemListMI16[problemidx]],
                    'gflops': [gflopsMI16[problemidx]],
                    'time': [WinnerKernelTimeMI16[problemidx]],
                    'kernelname': [str(WinnerKernelListMI16[problemidx])]},
                   columns=['problemsize', 'gflops', 'time', 'kernelname'])
	print(df1)
	dfout = pd.concat([dfout, df1], ignore_index=True)
	print(dfout)

print(dfout)
dfout.to_csv('/operator/hipBLASLt/tensilelite/chcekRoundNNMI16.csv', encoding='gbk', index=False)

################################################################Final

ProblemList = []
gflops = []
WinnerKernelList = []
WinnerKernelTime = []

for problemidx in range(0, problemnum):
	# print(problemidx)
	# print("32", WinnerKernelTimeMI32[problemidx], WinnerKernelTimeMI16[problemidx])
	if float(WinnerKernelTimeMI16[problemidx]) < float(WinnerKernelTimeMI32[problemidx]):
		# print("16")
		ProblemList.append(ProblemListMI16[problemidx])
		gflops.append(gflopsMI16[problemidx])
		WinnerKernelList.append(WinnerKernelListMI16[problemidx])
		WinnerKernelTime.append(WinnerKernelTimeMI16[problemidx])
		Convert(problemidx,"")
	else:
		# print("32")
		ProblemList.append(ProblemListMI32[problemidx])
		gflops.append(gflopsMI32[problemidx])
		WinnerKernelList.append(WinnerKernelListMI32[problemidx])
		WinnerKernelTime.append(WinnerKernelTimeMI32[problemidx])
		Convert(problemidx,"32")

with open(r'/operator/hipBLASLt/tensilelite/mergeALL.yaml', 'w') as fp:
    fp.write(''.join(startsection))
with open(r'/operator/hipBLASLt/tensilelite/mergeALL.yaml', 'a') as fp:
    fp.write(''.join(arg))
with open(r'/operator/hipBLASLt/tensilelite/mergeALL.yaml', 'a') as fp:
    fp.write(''.join(prob))
with open(r'/operator/hipBLASLt/tensilelite/mergeALL.yaml', 'a') as fp:
    fp.write(''.join(endsection))

dfout = pd.DataFrame(columns=['problemsize', 'gflops', 'time', 'kernelname'])

for problemidx in range(0, problemnum):
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
dfout.to_csv('/operator/hipBLASLt/tensilelite/chcekRoundNN.csv', encoding='gbk', index=False)