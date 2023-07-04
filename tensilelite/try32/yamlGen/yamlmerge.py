print("MERGE")

import re
import pickle

arg = []
prob = []

problemnum=23

problemidxoffset = 2+1

for problemidx in range(0, problemnum):
	# file=open("tunning_BFVF_Round2/tunning_BFVF_Round2_"+str(idx)+"/3_LibraryLogic/aldebaran_Cijk_Ailk_Bljk_HHS_BH.yaml",'r')
	lines = None
	with open("/operator/hipBLASLt/tensilelite/try32/yamlGen/tunning_BFVF_Round3/tunning_BFVF_Round3_"+str(problemidx)+"/3_LibraryLogic/aldebaran_Cijk_Ailk_Bljk_HHS_BH.yaml","rt") as txtfile:
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
with open(r'/operator/hipBLASLt/tensilelite/try32/yamlGen/merge.yaml', 'w') as fp:
    fp.write(''.join(startsection))
with open(r'/operator/hipBLASLt/tensilelite/try32/yamlGen/merge.yaml', 'a') as fp:
    fp.write(''.join(arg))
with open(r'/operator/hipBLASLt/tensilelite/try32/yamlGen/merge.yaml', 'a') as fp:
    fp.write(''.join(prob))
with open(r'/operator/hipBLASLt/tensilelite/try32/yamlGen/merge.yaml', 'a') as fp:
    fp.write(''.join(endsection))

# with open('listfile', 'wb') as fp:
#         pickle.dump(names, fp)