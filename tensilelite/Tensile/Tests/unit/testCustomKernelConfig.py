import pytest
import yaml
from pprint import pformat

from Tensile.CustomKernels import getCustomKernelConfig
from Tensile.SolutionStructs import matrixInstructionToMIParameters
from Tensile.Common.Architectures import SUPPORTED_ISA
from Tensile.Common.Capabilities import makeIsaInfoMap
from Tensile.Common.Types import IsaVersion
from Tensile.Common.GlobalParameters import defaultSolution
from Tensile.Toolchain.Validators import validateToolchain
from Tensile.TensileLogic.ValidMatrixInstruction import validateMIParameters

cxxCompiler = validateToolchain("amdclang++")

ISA_INFO_MAP = makeIsaInfoMap(SUPPORTED_ISA, cxxCompiler)


# @pytest.mark.parametrize("objs", [("TestKernel", testKernelDir, configResult)])
def testConvert9ItemCustomKernelConfig():

    inputConf = yaml.load(
        """
custom.config:
   ProblemType:
      OperationType: GEMM
      DataTypeA: f8n
      DataTypeB: h
      UseScaleAB: "Scalar"
      DataType: h
      DestDataType: s
      ComputeDataType: s
      HighPrecisionAccumulate: True
      TransposeA: False
      TransposeB: False
      UseBias: 1
      Activation: True
      UseScaleAlphaVec: 1
      UseBeta: True
      Batched: True
      GroupedGemm:   True
      SupportUserArgs: True
   MatrixInstruction: [32, 32, 8, 1, 5, 6, 7, 8, 9]
   1LDSBuffer: 1
   ScheduleIterAlg: 3
   DepthU: 32
   StaggerU: 0
   WorkGroupMapping: 8
   WaveSeparateGlobalReadA: 1
   WaveSeparateGlobalReadB: 1
   GlobalReadVectorWidthA: 4
   GlobalReadVectorWidthB: 2
   AssertFree0ElementMultiple: 4
   AssertSummationElementMultiple: 1
   NoReject: 1
   InternalSupportParams:
      KernArgsVersion: 0
      SupportUserGSU: False
      SupportCustomWGM: False
      SupportCustomStaggerU: False
      UseUniversalArgs: False
""",
        yaml.SafeLoader,
    )
    inputConf = inputConf["custom.config"]

    isa = IsaVersion(9, 4, 2)
    wavefrontSize = 48
    workGroup = [4, 5, 6]

    outputConf = matrixInstructionToMIParameters(
        inputConf["MatrixInstruction"],
        isa,
        wavefrontSize,
        inputConf["ProblemType"],
        workGroup,
        ISA_INFO_MAP,
    )

    input = {
        "MatrixInstruction": inputConf["MatrixInstruction"],
    }

    print("inputConf: ", pformat(input))
    print("outputConf: ", pformat(outputConf))

    assert outputConf["MatrixInstruction"] == [32, 32, 8, 1]
    assert outputConf["EnableMatrixInstruction"] == True
    assert outputConf["MIBlock"] == [32, 32, 8, 1, 1, 1]
    assert outputConf["MIWaveGroup"] == [40, 1]
    assert outputConf["MIWaveTile"] == [6, 7]
    assert outputConf["MatrixInstBM"] == 1
    assert outputConf["MatrixInstBN"] == 1
    assert outputConf["MIInputPerThread"] == 5
    assert outputConf["MIInputPerThreadA"] == 5
    assert outputConf["MIInputPerThreadB"] == 5
    assert outputConf["MIInputPerThreadMetadata"] == 5
    assert outputConf["ThreadTile"] == [1, 1]
    assert outputConf["Sparse"] == 0
    assert outputConf["WorkGroup"] == [1280, 2, 6]  # Why do we change the workgroup here?
    assert outputConf["WavefrontSize"] == 48
    assert outputConf["ISA"] == isa
    assert outputConf["EnableF32XdlMathOp"] == False
    assert outputConf["MFMA_BF16_1K"] == False

    solution = defaultSolution
    solution.update(inputConf)
    solution.update(outputConf)

    assert validateMIParameters(solution, ISA_INFO_MAP, True) == True

"""
def testConvert4ItemCustomKernelConfig():

    inputConf = yaml.load(
        """
custom.config:
   ProblemType:
      OperationType: GEMM
      DataTypeA: f8n
      DataTypeB: h
      UseScaleAB: "Scalar"
      DataType: h
      DestDataType: s
      ComputeDataType: s
      HighPrecisionAccumulate: True
      TransposeA: False
      TransposeB: False
      UseBias: 1
      Activation: True
      UseScaleAlphaVec: 1
      UseBeta: True
      Batched: True
      GroupedGemm:   True
      SupportUserArgs: True
   MatrixInstruction: [32, 32, 8, 1, 5, 6, 7, 8, 9]
   1LDSBuffer: 1
   ScheduleIterAlg: 3
   DepthU: 32
   StaggerU: 0
   WorkGroupMapping: 8
   WaveSeparateGlobalReadA: 1
   WaveSeparateGlobalReadB: 1
   GlobalReadVectorWidthA: 4
   GlobalReadVectorWidthB: 2
   AssertFree0ElementMultiple: 4
   AssertSummationElementMultiple: 1
   NoReject: 1
   InternalSupportParams:
      KernArgsVersion: 0
      SupportUserGSU: False
      SupportCustomWGM: False
      SupportCustomStaggerU: False
      UseUniversalArgs: False
""",
        yaml.SafeLoader,
    )
    inputConf = inputConf["custom.config"]

    isa = IsaVersion(9, 4, 2)
    wavefrontSize = 48
    workGroup = [4, 5, 6]

    outputConf = matrixInstructionToMIParameters(
        inputConf["MatrixInstruction"],
        isa,
        wavefrontSize,
        inputConf["ProblemType"],
        workGroup,
        ISA_INFO_MAP,
    )

    input = {
        "MatrixInstruction": inputConf["MatrixInstruction"],
    }

    print("inputConf: ", pformat(input))
    print("outputConf: ", pformat(outputConf))

    assert outputConf["MatrixInstruction"] == [32, 32, 8, 1]
    assert outputConf["EnableMatrixInstruction"] == True
    assert outputConf["MIBlock"] == [32, 32, 8, 1, 1, 1]
    assert outputConf["MIWaveGroup"] == [40, 1]
    assert outputConf["MIWaveTile"] == [6, 7]
    assert outputConf["MatrixInstBM"] == 1
    assert outputConf["MatrixInstBN"] == 1
    assert outputConf["MIInputPerThread"] == 5
    assert outputConf["MIInputPerThreadA"] == 5
    assert outputConf["MIInputPerThreadB"] == 5
    assert outputConf["MIInputPerThreadMetadata"] == 5
    assert outputConf["ThreadTile"] == [1, 1]
    assert outputConf["Sparse"] == 0
    assert outputConf["WorkGroup"] == [1280, 2, 6]  # Why do we change the workgroup here?
    assert outputConf["WavefrontSize"] == 48
    assert outputConf["ISA"] == isa
    assert outputConf["EnableF32XdlMathOp"] == False
    assert outputConf["MFMA_BF16_1K"] == False

    solution = defaultSolution
    solution.update(inputConf)
    solution.update(outputConf)

    assert validateMIParameters(solution, ISA_INFO_MAP, True) == True

"""
