"""Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
   ies of the Software, and to permit persons to whom the Software is furnished
   to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
   PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
   FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
   COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
   IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
   CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import sys
from os import path
import yaml
import subprocess
import math
import os
from collections import defaultdict
import re
from pathlib import Path
from git_info import create_github_file
from specs import get_machine_specs

# Parameters for output csv file

trackedParamList = ['transA','transB','grouped_gemm','batch_count','m','n','k','alpha','lda','stride_a','beta','ldb','stride_b',
                    'ldc','stride_c','ldd','stride_d','a_type','b_type','c_type','d_type','compute_type','scaleA','scaleB','scaleC',
                    'scaleD','activation_type','bias_vector','bias_type','gflops','us']

                    # 'transA', 'transB', 'uplo', 'diag', 'side', 'M', 'N', 'K', 'KL', 'KU', 'alpha', 'alphai', 'beta', 'betai',
                    # 'incx', 'incy', 'lda', 'ldb', 'ldd', 'stride_x', 'stride_y', 'stride_a', 'stride_b', 'stride_c', 'stride_d',
                    # 'batch_count', 'algo', 'solution_index', 'flags', 'iters', 'cold_iters', 'pointer_mode', 'num_perfs', 'sample_num']
                    # Add output (eg. gflops) in code dependent on what we want to record (gflops vs. gbytes)

# outputIndex = trackedParamList.index('num_perfs')
outputIndex = trackedParamList.index('us')

#d is default dict or contains a superset of trackedParams
def extractTrackedParams(d):
    return [d[p] for p in trackedParamList]

def parseOutput(output):
    csvKeys = ''

    lineLists = []

    capturingValues = False
    for line in output.split('\n'):
        line = line.strip()
        if capturingValues:
            dd_output = defaultdict(str, zip(csvKeys, line.split(',')))
            dd = dd_output
            lineLists += [extractTrackedParams(dd)]
            capturingValues = False
        elif line.startswith('['):
            line = line.replace('hipblaslt-Gflops', 'gflops')
            csvKeys = line.split(':')[1].split(',')
            # print("KEYS=",csvKeys)
            capturingValues = True
    return csvKeys, lineLists

def addSample(csvLists, newSample, perf_queries):
    newList = []
    matched = False
    for row in csvLists:
        # if new sample has same input params as row
        if row[:outputIndex] == newSample[:outputIndex] and not matched:
            cur_num_samples = row[trackedParamList.index('sample_num')]
            for i in range(len(perf_queries)):
                # outputIndex + 2 puts us just past end of trackedParamList
                # add 2 * len(perf_queries) to get past mean/median results
                # add cur_samples * (i + 1) + i to insert at end of this perf_query output
                idx = outputIndex + 2 + 2 * len(perf_queries) + (cur_num_samples) * (i + 1) + i
                if newSample[trackedParamList.index(perf_queries[i])].strip() != '':
                    row.insert(idx, newSample[trackedParamList.index(perf_queries[i])])
                else:
                    row.insert(idx, '0')
            row[trackedParamList.index('sample_num')] = row[trackedParamList.index('sample_num')] + 1
            matched = True
        newList += [row]

    # if this is the first sample with this combination of input params, create new row
    if not matched:
        num_perfs = 0
        for perf in perf_queries:
            if newSample[trackedParamList.index(perf)].strip() == '':
                newSample[trackedParamList.index(perf)] = '0'
        newSample[trackedParamList.index('num_perfs')] = len(perf_queries)
        newSample[trackedParamList.index('sample_num')] = 1
        newList += [newSample]

    return newList

def calculateMean(row, param_idx = 0):
    # if we're recording gflops and gbytes for 3 samples,
    # the data looks as follows:
    # [gflops1, gflops2, gflops3, gbytes1, gbytes2, gbytes3] for example
    # the indexing here should support any # of samples, and any # of perf_queries recorded

    # outputIndex + 2 puts us just past end of trackedParamList
    # add 2 * len(perf_queries) to get past mean/median results
    # add cur_samples * i + i to get beginning of results for this perf_query output
    num_perfs = row[trackedParamList.index('num_perfs')]
    num_samples = int(row[trackedParamList.index('sample_num')])
    idx = outputIndex + 2 + 2 * num_perfs + (param_idx) * num_samples + param_idx
    l = [float(v) for v in row[idx:idx + num_samples]]
    return sum(l)/float(len(l))

def calculateMedian(row, param_idx = 0):
    num_perfs = row[trackedParamList.index('num_perfs')]
    num_samples = int(row[trackedParamList.index('sample_num')])
    idx = outputIndex + 2 + 2 * num_perfs + (param_idx) * num_samples + param_idx
    l = [float(v) for v in row[idx:idx + num_samples]]
    l.sort()
    return l[math.floor(len(l)/2)]

def exportBenchmarkCVS(benchCmd, problemsYaml, samples, outputFile):

    with open(problemsYaml, 'r') as file:
        problems = file.read()

    # csvLists = [trackedParamList]

    output = subprocess.check_output([benchCmd,
                                      '--yaml', problemsYaml])

    output = output.decode('utf-8')

    print(output)

    csvKeys, benchResults = parseOutput(output)

    # for i in range(0, int(samples)):
    #     print('Sample ({}/{})'.format(i+1, samples))
    #     output = subprocess.check_output([benchCmd,
    #                                       '--yaml', problemsYaml])

    #     output = output.decode('utf-8')

    #     print(output)

    #     benchResults = parseOutput(output, problems)

    #     for sample in benchResults:
    #         csvLists = addSample(csvLists, sample, perf_queries)

    # for problem in csvLists[1:]:
    #     for i in range(len(res_queries)):
    #         problem[trackedParamList.index('mean_' + res_queries[i])] = calculateMean(problem, i)
    #         problem[trackedParamList.index('median_' + res_queries[i])] = calculateMedian(problem, i)

    content = ''
    content += ','.join([str(key) for key in csvKeys])+'\n'

    for line in benchResults:
        content += ','.join([str(e) for e in line])+'\n'

    with open(outputFile, 'w') as f:
        print("Writing results to {}".format(outputFile))
        f.write(content)

def main(args):
    if len(args) < 4:
        print('Usage:\n\tpython3 bench_pts_data.py bench_executable path tag benchfile1 benchfile2...')
        return 0

    benchCmd  = args[0]
    dirName   = args[1]
    tag       = args[2]
    filenames = args[3:]

    # perf_queries: what we read from bench output
    # res_queries: what we write to .csv file.
    # perf_queries = ['hipblaslt-Gflops']
    # res_queries = ['gflops']

    # # append to trackedParamList our performance output of interest
    # global trackedParamList
    # for perf in res_queries:
    #     trackedParamList += ['mean_' + perf]
    #     trackedParamList += ['median_' + perf]
    # for perf in perf_queries:
    #     trackedParamList += [perf]

    for filename in filenames:
        print("==================================\n|| Running benchmarks from {}\n==================================".format(filename))
        filePrefix = Path(filename).stem
        subDirectory = os.path.join(dirName, "hipBLASLt_PTS_Benchmarks_"+filePrefix, tag)
        Path(subDirectory).mkdir(parents=True, exist_ok=True)

        outputName, _ = os.path.splitext(os.path.basename(filename))

        exportBenchmarkCVS(benchCmd,
                           filename,
                           10,
                           os.path.join(subDirectory, outputName+'_benchmark.csv'))

        # Will only be correct if script is run from directory of the git repo associated
        # with the rocblas-bench executable
        get_machine_specs(os.path.join(subDirectory, 'specs.txt'))
        create_github_file(os.path.join(subDirectory, 'hipBLASLt-commit-hash.txt'))

if __name__=='__main__':
    main(sys.argv[1:])
