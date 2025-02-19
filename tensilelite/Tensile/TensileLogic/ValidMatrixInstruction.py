################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

"""
ValidMatrixInstruction
---
Format: (M x N x K x B)
    XDLOPS tile definition, only valid for gfx908, gfx90a
    MxNxKxB specifies matrix instruction variants
    MxNxB determines the shape of the C tile each instruction worked on
        K determines the unroll depth

Alternative format: (M x N x K x B x MIBlockM x WaveTileM x WaveTileN x WaveM x WaveN)
    (Note: MxN means M-by-N in the following comments)
    MIBlockM determines how many blocks along M dimension for multi-block MI variants. Concrete examples:
    - MI 16x16x1x4 (4-block variant) with MIBlockM=4 -> (16x16)*(4x1)=64x16 tile per instruction executed
    - MI 32x32x1x2 (2-block variant) with MIBlockM=1 -> (32x32)*(1x2)=32x64 tile per instruction executed
    WaveTileM/N are dimensions of the C tile each wave works on, and is close to the concept of ThreadTile in classic VALU kernels
    - WT 4x1 -> each wave executes 4x1 matrix instructions on the C tile of total area (4*MITileM)x(1*MITileN)
    WaveM/N are dimensions of waves spawned for one workgroup where each wave consists of 64 threads
    - Wave2x2 -> a total of 4 waves in one workgroup of shape 2x2
    Putting it all together:
    - [32, 32, 1, 2,  1,  4, 1,  2, 2]
       ^^^^^^^^^^^^   ^   ^^^^   ^^^^
        MatrixInst   BlkM  WT    Wave
    - means (32x64) per MI * (4x1) per wave * (2x2) per workgroup = (32*4*2)x(64*1*2) = 256x128 macro tile
    Tensile will ignore the parameters ThreadTile and WorkGroup when the alternative format is used

Notes:
    - If empty, do not use these instructions
"""

import math

from Tensile.SolutionStructs import reject
from Tensile.TensileInstructions.DataType import DataType

from .Utilities import elineno

MI_KEY: str = "MatrixInstruction"
MI_ENABLED_KEY: str = "EnableMatrixInstruction"


validMFMA = {}
validMFMA["H"] = [[32, 32, 4, 2], [32, 32, 8, 1], [16, 16, 4, 4], [16, 16, 16, 1], [4, 4, 4, 16]]
validMFMA["S"] = [[32, 32, 1, 2], [32, 32, 2, 1], [16, 16, 1, 4], [16, 16, 4, 1], [4, 4, 1, 16]]
validMFMA["B"] = [[32, 32, 2, 2], [32, 32, 4, 1], [16, 16, 2, 4], [16, 16, 8, 1], [4, 4, 2, 16]]
validMFMA["4xi8"] = [
    [32, 32, 4, 2],
    [32, 32, 8, 1],
    [16, 16, 4, 4],
    [16, 16, 16, 1],
    [4, 4, 4, 16],
    [32, 32, 16, 1],
    [16, 16, 32, 1],
]
validMFMA["D"] = [[16, 16, 4, 1], [4, 4, 4, 4]]
validMFMA["B1k"] = [[32, 32, 4, 2], [32, 32, 8, 1], [16, 16, 4, 4], [16, 16, 16, 1], [4, 4, 4, 16]]
validMFMA["C"] = validMFMA["S"]
validMFMA["Z"] = validMFMA["D"]
validMFMA["I8"] = [
    [32, 32, 4, 2],
    [32, 32, 8, 1],
    [16, 16, 4, 4],
    [16, 16, 16, 1],
    [4, 4, 4, 16],
] + [[32, 32, 16, 1], [16, 16, 32, 1]]
validMFMA["X"] = [[32, 32, 4, 1], [16, 16, 8, 1]]
validMFMA["F8"] = [[32, 32, 16, 1], [16, 16, 32, 1]]
validMFMA["B8"] = validMFMA["F8"]
validMFMA["F8B8"] = validMFMA["F8"]
validMFMA["B8F8"] = validMFMA["F8"]
validMFMA["F8N"] = [[32, 32, 16, 1], [16, 16, 32, 1]]
validMFMA["B8N"] = validMFMA["F8N"]
validMFMA["F8B8N"] = validMFMA["F8N"]
validMFMA["B8F8N"] = validMFMA["F8N"]
validWMMA = [
    [16, 16, 16, 1],
]
validTT = 32
validMFMA["_format9"] = []

for MFMA in [
    validMFMA["H"],
    validMFMA["S"],
    validMFMA["B"],
    validMFMA["D"],
    validMFMA["X"],
    validMFMA["F8N"],
    validWMMA,
]:
    for MI in MFMA:
        for bm in range(int(math.log(MI[3], 2)) + 1):
            for tt0 in range(1, validTT + 1):
                for tt1 in range(1, validTT + 1):
                    for wave_m in range(3):
                        for wave_n in range(3):
                            validMFMA["_format9"].append(
                                [MI[0], MI[1], MI[2], MI[3], 2**bm, tt0, tt1, 2**wave_m, 2**wave_n]
                            )
validMatrixInstructions = (
    [[], [-1]]
    + validMFMA["H"]
    + validMFMA["S"]
    + validMFMA["B"]
    + validMFMA["D"]
    + validMFMA["B1k"]
    + validMFMA["X"]
)
validMatrixInstructions = validMatrixInstructions + validMFMA["_format9"]

validSMFMA = {}
validSMFMA["H"] = [[32, 32, 16, 1], [16, 16, 32, 1]]
validSMFMA["B"] = [[32, 32, 16, 1], [16, 16, 32, 1]]
validSMFMA["4xi8"] = [[32, 32, 32, 1], [16, 16, 64, 1]]
validSMFMA["I8"] = validSMFMA["4xi8"]
validSMFMA["F8"] = [[32, 32, 32, 1], [16, 16, 64, 1]]
validSMFMA["B8"] = validSMFMA["F8"]
validSMFMA["F8B8"] = validSMFMA["F8"]
validSMFMA["B8F8"] = validSMFMA["F8"]
validSMFMA["F8N"] = [[32, 32, 32, 1], [16, 16, 64, 1]]
validSMFMA["B8N"] = validSMFMA["F8N"]
validSMFMA["F8B8N"] = validSMFMA["F8N"]
validSMFMA["B8F8N"] = validSMFMA["F8N"]
validSMFMA["_format9"] = []
for SMFMA in [validSMFMA["H"], validSMFMA["B"], validSMFMA["4xi8"], validSMFMA["F8N"]]:
    for MI in SMFMA:
        for bm in range(int(math.log(MI[3], 2)) + 1):
            for tt0 in range(1, validTT + 1):
                for tt1 in range(1, validTT + 1):
                    for wave_m in range(3):
                        for wave_n in range(3):
                            validSMFMA["_format9"].append(
                                [MI[0], MI[1], MI[2], MI[3], 2**bm, tt0, tt1, 2**wave_m, 2**wave_n]
                            )
validSparseMatrixInstructions = validSMFMA["H"] + validSMFMA["B"] + validSMFMA["4xi8"]
validMatrixInstructions = (
    validMatrixInstructions + validSparseMatrixInstructions + validSMFMA["_format9"]
)


def validateMatrixInstruction(solution: dict, globalParams: dict, filepath: str):
    """
    Validates the matrix instruction configured in the given solution.

    The function performs the following checks:
    - Ensures that the solution contains the required keys for matrix instruction support.
    - Ensures that the matrix instruction is not empty when it is enabled.
    - Validates that the matrix instruction is in the list of valid matrix instructions.
    - If the matrix instruction has 9 elements, it performs detailed validation checks:
        - Validates the work group dimensions.
        - Checks if the matrix instruction is supported by the assembler capabilities (MFMA or WMMA).
        - Validates the input per thread for sparse and non-sparse configurations.
        - Validates the matrix instruction block, wave group, and wave tile dimensions.
    - If the matrix instruction has 4 elements, it ensures that matrix instructions are enabled.
    - If the matrix instruction is empty, it ensures that matrix instructions are disabled.

    Args:
        solution: The solution to validate.
        filepath: The path to the file containing the solution.
        params: The global parameters for the solution.

    Raises:
        AssertionError: If any of the validation checks fail.
    """
    try:
        validateMIParameters(solution, globalParams)
        assert solution["Valid"], f"Solution was rejected: {elineno()}"
        return True
    except AssertionError as e:
        print(
            f"Error: Validation failed: {e} (file: {filepath}, index: {solution['SolutionIndex']})"
        )
        return False


def validateMIParameters(solution: dict, globalParams: dict):
    assert MI_KEY in solution, elineno() + ": missing MatrixInstruction"
    assert MI_ENABLED_KEY in solution, elineno() + ": missing EnableMatrixInstruction"
    assert not (solution[MI_KEY] == [] and solution[MI_ENABLED_KEY] == True), elineno() + ": MI empty but enabled"

    isa = tuple(solution["ISA"])

    # TODO: Temporary until all 940/941 ISAs are removed
    if (9, 4, 0) <= isa <= (9, 4, 1):
        isa = (9, 4, 2)

    mi4 = solution[MI_KEY]
    mi9 = [mi4[0], mi4[1], mi4[2], mi4[3]]
    assert "MatrixInstBM" in solution, elineno() + ": missing MatrixInstBM"
    mi9.append(solution["MatrixInstBM"])
    assert "MIWaveTile" in solution, elineno() + ": missing MIWaveTile"
    mi9.extend(solution["MIWaveTile"])
    assert "MIWaveGroup" in solution, elineno() + ": missing MIWaveGroup"
    mi9.extend(solution["MIWaveGroup"])

    miEnabled = solution[MI_ENABLED_KEY]

    if len(mi4) == 0:
        assert miEnabled == False, elineno()
    else:
        assert len(mi4) == 4 and len(mi9) == 9, (
            elineno() + " MI4: " + str(mi4) + " MI9: " + str(mi9)
        )

    if not miEnabled:
        return

    assert mi4 in validMatrixInstructions, elineno()

    wfsize = solution["WavefrontSize"]
    waves = solution["MIWaveGroup"][0] * solution["MIWaveGroup"][1]
    miwg0 = mi9[4] * mi9[0] * mi9[7]  # Matrix instruction work group 0
    miwg1 = waves * wfsize // miwg0

    hasMFMA = globalParams["AsmCaps"][isa]["HasMFMA"]
    hasWMMA = globalParams["AsmCaps"][isa]["HasWMMA"]

    ptype = solution["ProblemType"]
    isSparse = ptype.get("Sparse", 0)
    miDataType = DataType(
        ptype["DataType"]
        if not solution.get("EnableF32XdlMathOp", False)
        else ptype["F32XdlMathOp"]
    )

    miBlock = solution["MIBlock"]
    miWaveGroup = solution["MIWaveGroup"]
    miWaveTile = solution["MIWaveTile"]

    # Check datatype
    if not isSparse:
        if hasMFMA:
            if not (miDataType.toChar() in validMFMA and mi4 in validMFMA[miDataType.toChar()]):
                if miDataType.isBFloat16() and mi4 in validMFMA["B1k"]:
                    assert solution["MFMA_BF16_1K"], elineno()
                else:
                    reject(solution, f"Invalid MFMA BFloat16 configuration: {solution}")
        elif hasWMMA and (not mi4 in validWMMA):
            reject(solution, f"Invalid WMMA configuration: {solution}")
    else:
        if not (miDataType.toChar() in validSMFMA and mi4 in validSMFMA[miDataType.toChar()]):
            reject(solution, f"Invalid SMFMA configuration: {solution}")

    if (not hasWMMA) and globalParams["AsmCaps"][isa]["HasWMMA"]:
        if isa[0] == 10 or isa[0] == 11:
            assert miInputPerThread == mi4[2], elineno()

    # Check MIBlock
    assert miBlock[0] == mi4[0], elineno()
    assert miBlock[1] == mi4[1], elineno()
    assert miBlock[2] == mi4[2], elineno()
    assert miBlock[3] == mi4[3], elineno()
    assert miBlock[4] == min(miwg0 // mi4[0], mi4[3]), elineno()
    assert miBlock[5] == mi4[3] // miBlock[4], elineno()

    # Check MIWaveGroup
    assert miWaveGroup[0] == min((miwg0 // mi4[0]) // miBlock[4], waves), elineno()
    assert miWaveGroup[1] == waves // miWaveGroup[0], elineno()

    # Check MIWaveTile
    assert miWaveTile[0] == mi9[5], elineno()
    assert miWaveTile[1] == mi9[6], elineno()

    # Check MIInputPerThread
    miInputPerThread = solution["MIInputPerThread"]

    # If Navi architecture, the input per thread is different
    if (10, 0, 0) <= isa <= (11, 0, 2):
        assert miInputPerThread == mi4[2], elineno()
    else:
        assert miInputPerThread == mi4[0] * mi4[2] * mi4[3] // wfsize, elineno()

    # miInputPerThreadA = solution["MIInputPerThreadA"]
    # miInputPerThreadB = solution["MIInputPerThreadB"]
    # miInutPerThreadMeta = solution["MIInputPerThreadMetadata"]
    # sparseA = not isSparse if isSparse != 2 else False
    # sparseB = isSparse == 2 if isSparse else False
    # assert miInputPerThreadA == miInputPerThread if not sparseA else miInputPerThread // 2, elineno()
    # assert miInputPerThreadB == miInputPerThread if not sparseB else miInputPerThread // 2, elineno()
    # assert miInutPerThreadMeta == miInputPerThread if not isSparse else miInputPerThread // 8, elineno()
