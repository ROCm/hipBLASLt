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

import sys
import pprint
from typing import Dict, Optional

from Tensile.Common import IsaVersion, IsaInfo, print1, print2
from Tensile.Common.ValidParameters import validMFMA
from Tensile.TensileInstructions import DataType

def reject(state: dict, printSolutionRejectionReason: bool = True, *args) -> bool:
  """
  Reject a solution based on its internal state.

  Args:
      state: The state of the solution.
      printSolutionRejectionReason: If True, print the rejection reason.
      *args: Additional arguments to print if rejection occurs.

  Returns:
      True if the solution is rejected, False otherwise.
  """
  if state and "NoReject" in state and state["NoReject"]:
    return False
  if printSolutionRejectionReason:
    sys.stdout.write("\nreject: ")
    for a in args:
      print(a)
    #traceback.print_stack(None, 2)
    solutionIndex = state["SolutionIndex"] if (state != None and "SolutionIndex" in state) else -1
    if solutionIndex != -1:
      # If we have valid solutionIndex, this means we are during TensileCreateLibrary stage
      # In this stage, all solutions in the logic should be valid
      # So if any rejection happens, print the warning for further check
      # This will be done only when --global-parameters=PrintSolutionRejectionReason=True
      solutionNameMin = state["SolutionNameMin"] if ("SolutionNameMin" in state) else None
      # if we don't have SolutionNameMin, we simply use the problemTypeName
      solutionNameMin = str(state["ProblemType"]) if (solutionNameMin == None) else solutionNameMin
      raise Exception("!! Warning: Any rejection of a LibraryLogic is not expected, please check. \
        SolutionIndex: %d (or SolutionName/ProblemType: %s)"%(solutionIndex, solutionNameMin))
  if state != None:
    state["Valid"] = False
    return True

def matrixInstructionToMIParameters(
      mi: list,
      isa: IsaVersion,
      wavefrontSize: int,
      problemType: dict,
      workGroup: list,
      isaInfoMap: Dict[IsaVersion, IsaInfo]
    ):
    """
    Converts a 9-item matrix instruction into the associated 4-item representation and
    populates supporting MI parameters.

    Args:
        mi: The matrix instruction to convert. Must have length 9.
        isa: The ISA tuple.
        wavefrontSize: The wavefront size. Typically "WavefrontSize" in a solution.
        problemType: The problem type dictionary. Typically "ProblemType" in a solution.
    """
    print1(f">> --DBG-- Converting MatrixInstruction {mi} to MI parameters")

    if len(mi) != 9:
      raise ValueError(f"MatrixInstruction must be 9 items long to convert into MI"
                       f" Parameters, found {mi} with length {len(mi)}")

    result = {}
    result["ISA"] = isa

    # Enable F32 XDL math operation only when the input type is f32.
    enableF32xdl = (
      "F32XdlMathOp" in problemType
      and not problemType["F32XdlMathOp"].isSingle()
      and problemType["DataType"].isSingle()
    )
    result["EnableF32XdlMathOp"] = enableF32xdl

    mi4  = [mi[0], mi[1], mi[2], mi[3]]
    result["MatrixInstruction"] = mi4
    result["EnableMatrixInstruction"] = True
    result["MatrixInstM"] = mi[0]
    result["MatrixInstN"] = mi[1]
    result["MatrixInstK"] = mi[2]
    result["MatrixInstB"] = mi[3]

    waves = mi[7]* mi[8]
    wg0 = mi[4] * mi[0] * mi[7]

    result["WavefrontSize"] = wavefrontSize
    result["WorkGroup"] = [wg0, waves*wavefrontSize // wg0, workGroup[2]]
    result["ThreadTile"] = [1, 1]  # dummy

    isSparse = problemType.get("Sparse", 0)
    miDataType = DataType(
        problemType["DataType"]
        if not enableF32xdl
        else problemType["F32XdlMathOp"]
    )

    result["MFMA_BF16_1K"] = (
        not isSparse
        and isaInfoMap[isa].asmCaps["HasMFMA"]
        and not (miDataType.toChar() in validMFMA and mi4 in validMFMA[miDataType.toChar()])
        and miDataType.isBFloat16()
        and mi4 in validMFMA["B1k"]
    )

    # set MIBlock
    MIBlockBM = wg0 // mi[0]
    MIBlockBM = min(MIBlockBM, mi[3])
    MIBlockBN = mi[3] // MIBlockBM
    result["MatrixInstBM"] = MIBlockBM
    result["MatrixInstBN"] = MIBlockBN
    result["MIBlock"]    = [mi[0], mi[1], mi[2], mi[3], MIBlockBM, MIBlockBN]

    # set MIWaveGroup
    miwg0 = min((wg0 // mi[0]) // MIBlockBM, waves)
    result['MIWaveGroup'] = [miwg0, waves // miwg0]

    # set MIWaveTile
    result['MIWaveTile'] = [mi[5], mi[6]]

    # set MIInputPerThread
    hasMFMA = isaInfoMap[isa].asmCaps["HasMFMA"]
    hasWMMA = isaInfoMap[isa].asmCaps["HasWMMA"]

    result['MIInputPerThread'] = mi[0] * mi[2] * mi[3] // wavefrontSize
    if (not hasMFMA) and hasWMMA and (isa[0] == 10 or isa[0] == 11):
      result['MIInputPerThread'] = mi[2]

    sparseA = False if not isSparse else False if isSparse == 2 else True
    sparseB = False if not isSparse else True if isSparse == 2 else False
    result['MIInputPerThreadA'] = result['MIInputPerThread'] if not sparseA else result['MIInputPerThread'] // 2
    result['MIInputPerThreadB'] = result['MIInputPerThread'] if not sparseB else result['MIInputPerThread'] // 2
    result['MIInputPerThreadMetadata'] = result['MIInputPerThread'] if not isSparse else result['MIInputPerThread'] // 8
    result['Sparse'] = isSparse

    print2(f">> MI Parameters: {pprint.pformat(result)}")
    return result
