################################################################################
#
# Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

from ..Component import Signature
from ..Common import globalParameters
from ..Utils import DataDirection
from ..TensileInstructions import SignatureBase
from ..TensileInstructions import SignatureValueKind as SVK
from ..Activation import ActivationType

from math import ceil
from dataclasses import dataclass, field

@dataclass
class UserArgumentsInfo:
    # variable related fixed parameters
    alphaMaxSize: int = 16
    alphaMaxRegisterSize: int = field(init=False)
    betaMaxSize: int = 16
    betaMaxRegisterSize: int = field(init=False)
    scaleASize: int = 0
    scaleBSize: int = 0
    scaleCSize: int = 0
    scaleDSize: int = 0
    actMaxSize: int = 4
    actMaxRegisterSize: int = field(init=False)
    # gemm related
    gemmArgumentSize: int = 0
    # Epilogue related
    scaleAlphaVecSize: int = 0
    biasSize: int = 0
    eSize: int = 0
    activationSize: int = 0
    # Total argument size
    totalSize: int = 0

    def __post_init__(self):
        self.alphaMaxRegisterSize = self.alphaMaxSize // 4
        self.betaMaxRegisterSize  = self.betaMaxSize // 4
        self.actMaxRegisterSize   = self.actMaxSize // 4

def getSrcValueType(kernel, cov, isTypeA):
    # special cases for F8 datatypes
    if kernel["ProblemType"]["DataType"].isFloat8():
        srcValueType = "FP8"
    elif kernel["ProblemType"]["DataType"].isBFloat8():
        srcValueType = "BF8"
    elif kernel["ProblemType"]["DataType"].isFloat8BFloat8():
        srcValueType = "FP8" if isTypeA else "BF8"
    elif kernel["ProblemType"]["DataType"].isBFloat8Float8():
        srcValueType = "BF8" if isTypeA else "FP8"
    else:
        srcValueType = kernel["ProblemType"]["DataType"].toNameAbbrev().upper()

    if cov == "V3":
        srcValueType = srcValueType.lower()
    return srcValueType

def getDstValueType(kernel, cov):
    # special cases for F8 datatypes
    if kernel["ProblemType"]["DataType"].isFloat8():
        dstValueType = "FP8"
    elif kernel["ProblemType"]["DataType"].isBFloat8():
        dstValueType = "BF8"
    else:
        dstValueType = kernel["ProblemType"]["DataType"].toNameAbbrev().upper()

    if cov == "V3":
        dstValueType = dstValueType.lower()
    return dstValueType

class SignatureCOV3(Signature):
    kernel = {"CodeObjectVersion": "V3"}

    def __call__(self, writer) -> SignatureBase:
        kernel = writer.states.kernel

        userArgumentsInfo = UserArgumentsInfo()

        # kern arg size
        kernArgReg = 0
        kernArgReg += 3*writer.states.rpga
        kernArgReg += max(1,int(writer.states.bpeAB/4)) # alpha
        if kernel["ProblemType"]["UseBeta"]:
            kernArgReg += max(1,int(writer.states.bpeCexternal/4)) # beta
        kernArgReg += kernel["ProblemType"]["NumIndicesC"] # strides
        kernArgReg += kernel["ProblemType"]["NumIndicesC"] # strides
        kernArgReg += len(kernel["ProblemType"]["IndexAssignmentsA"]) # strides
        kernArgReg += len(kernel["ProblemType"]["IndexAssignmentsB"]) # strides
        if not kernel["ProblemType"]["UseInitialStridesAB"]:
            kernArgReg -= 2 # strides
        if not kernel["ProblemType"]["UseInitialStridesCD"]:
            kernArgReg -= 2 # strides
        kernArgReg += kernel["ProblemType"]["NumIndicesSummation"]
        kernArgReg += kernel["ProblemType"]["NumIndicesC"]
        if globalParameters["DebugKernel"]:
            kernArgReg += writer.states.rpga # debug buffer
        # kernArgBytes = kernArgReg * 4 # bytes/reg

        group_segment_size = kernel["LdsNumElements"] * writer.states.bpeAB

        sgprWgZ = 1 if kernel["ProblemType"]["NumIndicesC"] > 2 else 0
        signature = SignatureBase(kernelName=writer.states.kernelName,
                                    codeObjectVersion="v3",
                                    groupSegmentSize=group_segment_size,
                                    sgprWorkGroup=[1, 1, sgprWgZ],
                                    vgprWorkItem=0,
                                    flatWorkGroupSize=(kernel["NumThreads"]),
                                    preloadKernArgs=kernel["PreloadKernArgs"])

        srcValueTypeA = getSrcValueType(kernel, "V3", True)
        srcValueTypeB = getSrcValueType(kernel, "V3", False)
        dstValueType  = kernel["ProblemType"]["DestDataType"].toNameAbbrev()
        cptValueType  = kernel["ProblemType"]["ComputeDataType"].toNameAbbrev()
        biasValueType = "void"
        actValueType  = kernel["ProblemType"]["ActivationComputeDataType"].toNameAbbrev()

        for i in range(0, writer.states.numSgprSizesFree):
            signature.addArg(            "SizesFree%u"%i, SVK.SIG_VALUE,               "u32")
            userArgumentsInfo.gemmArgumentSize += 4

        for i in range(0, writer.states.numSgprSizesSum):
            signature.addArg(             "SizesSum%u"%i, SVK.SIG_VALUE,               "u32")
            userArgumentsInfo.gemmArgumentSize += 4

        if globalParameters["DebugKernel"]:
            signature.addArg("AddressDbg", SVK.SIG_GLOBALBUFFER, "struct", "generic")
        signature.addArg(    "D", SVK.SIG_GLOBALBUFFER, dstValueType, "generic")
        signature.addArg(    "C", SVK.SIG_GLOBALBUFFER, dstValueType, "generic")
        signature.addArg(    "A", SVK.SIG_GLOBALBUFFER, srcValueTypeA, "generic")
        signature.addArg(    "B", SVK.SIG_GLOBALBUFFER, srcValueTypeB, "generic")
        userArgumentsInfo.gemmArgumentSize += (8 + 8 + 8 + 8)  # A, B, C, D buffer

        if kernel["ProblemType"]["SparseA"]:
            signature.addArg("MetaData", SVK.SIG_GLOBALBUFFER, "void" , "generic")

        for i in range(0, writer.states.d.numSgprStrides):
            signature.addArg(              "strideD%u"%i, SVK.SIG_VALUE,               "u32")
            userArgumentsInfo.gemmArgumentSize += 4

        for i in range(0, writer.states.c.numSgprStrides):
            signature.addArg(              "strideC%u"%i, SVK.SIG_VALUE,               "u32")
            userArgumentsInfo.gemmArgumentSize += 4

        for i in range(0, writer.states.a.numSgprStrides):
            signature.addArg(              "strideA%u"%i, SVK.SIG_VALUE,               "u32")
            userArgumentsInfo.gemmArgumentSize += 4

        for i in range(0, writer.states.b.numSgprStrides):
            signature.addArg(              "strideB%u"%i, SVK.SIG_VALUE,               "u32")
            userArgumentsInfo.gemmArgumentSize += 4

        if kernel["ProblemType"]["SparseA"]:
            for i in range(0, writer.states.m.numSgprStrides):
                signature.addArg(   "strideMetadata%u"%i, SVK.SIG_VALUE,               "u32")

        for idxChar in kernel["PackedC0IdxChars"][:-1]:
            signature.addArg("MagicNumberSize%s"%idxChar, SVK.SIG_VALUE,               "u32")
            signature.addArg( "MagicShiftSize%s"%idxChar, SVK.SIG_VALUE,               "u32")

        # Note: We use packed f16 if alpha and beta are f16
        pack_cptValueType = 'pkf16' if kernel["ProblemType"]["ComputeDataType"].isHalf() else cptValueType
        signature.addArg(   "alpha",        SVK.SIG_VALUE, pack_cptValueType)
        if kernel["ProblemType"]["UseBeta"]:
            signature.addArg("beta",        SVK.SIG_VALUE, pack_cptValueType)
        # These are fixed sizes
        userArgumentsInfo.gemmArgumentSize += userArgumentsInfo.alphaMaxSize
        userArgumentsInfo.gemmArgumentSize += userArgumentsInfo.betaMaxSize

        # Kernel related arguments
        if not kernel["ProblemType"]["GroupedGemm"]:
            signature.addArg(       "gsu",                SVK.SIG_VALUE,               "u32")

        if kernel["ProblemType"]["UseScaleAB"]:
            if (kernel["GlobalSplitU"] == 1) or (kernel["ProblemType"]["DataTypeA"].numRegisters() > kernel["ProblemType"]["DataType"].numRegisters()):
                signature.addArg("AddressScaleA", SVK.SIG_GLOBALBUFFER, cptValueType, "generic")
            if (kernel["GlobalSplitU"] == 1) or (kernel["ProblemType"]["DataTypeB"].numRegisters() > kernel["ProblemType"]["DataType"].numRegisters()):
                signature.addArg("AddressScaleB", SVK.SIG_GLOBALBUFFER, cptValueType, "generic")
        userArgumentsInfo.scaleASize += 8
        userArgumentsInfo.scaleBSize += 8
        if kernel["ProblemType"]["UseScaleCD"] and (kernel["GlobalSplitU"] == 1):
            signature.addArg("AddressScaleC", SVK.SIG_GLOBALBUFFER, cptValueType, "generic")
            signature.addArg("AddressScaleD", SVK.SIG_GLOBALBUFFER, cptValueType, "generic")
        userArgumentsInfo.scaleCSize += 8
        userArgumentsInfo.scaleDSize += 8

        if kernel["ProblemType"]["UseScaleAlphaVec"] and (kernel["GlobalSplitU"] == 1):
            signature.addArg("AddressScaleAlphaVec", SVK.SIG_GLOBALBUFFER, cptValueType, "generic")
        userArgumentsInfo.scaleAlphaVecSize += 8

        if writer.states.useBias != DataDirection.NONE and (kernel["GlobalSplitU"] == 1):
            signature.addArg("bias", SVK.SIG_GLOBALBUFFER, biasValueType, "generic")  # Note: We append the data in ws_d
            if writer.states.needBiasType:
                signature.addArg("biasType",        SVK.SIG_VALUE,        "u32")
                signature.addArg("StrideBias",      SVK.SIG_VALUE,        "u32")
        userArgumentsInfo.biasSize += (8 + 4 + 4)

        if kernel["ProblemType"]["UseE"] and (kernel["GlobalSplitU"] == 1):
            signature.addArg(      "E", SVK.SIG_GLOBALBUFFER, cptValueType, "generic")
            for i in range(0, writer.states.e.numSgprStrides):
                signature.addArg("StrideE%u"%i,        SVK.SIG_VALUE,        "u32")
        userArgumentsInfo.eSize += 8
        for i in range(0, writer.states.e.numSgprStrides):
            userArgumentsInfo.eSize += 4

        if ((kernel["ProblemType"]["ActivationType"] != 'none') and (kernel["GlobalSplitU"] == 1) \
            and kernel["ActivationFused"]):
            if kernel["ProblemType"]["ActivationComputeDataType"].isHalf():
                actValueType = 'pkf16'
            for name in kernel["ProblemType"]["ActivationType"].getAdditionalArgStringList():
                signature.addArg(                   name, SVK.SIG_VALUE,        actValueType)
            if kernel["ProblemType"]["ActivationType"] == 'all':
                signature.addArg(       "activationType", SVK.SIG_VALUE,               "u32")
        activationType = ActivationType("all")
        for name in activationType.getAdditionalArgStringList():
            userArgumentsInfo.activationSize += userArgumentsInfo.actMaxSize
        userArgumentsInfo.activationSize += 4  # Type size

        # Calculate total size
        userArgumentsInfo.totalSize = userArgumentsInfo.gemmArgumentSize + \
                                      userArgumentsInfo.scaleASize + \
                                      userArgumentsInfo.scaleBSize + \
                                      userArgumentsInfo.scaleCSize + \
                                      userArgumentsInfo.scaleDSize + \
                                      userArgumentsInfo.scaleAlphaVecSize + \
                                      userArgumentsInfo.biasSize + \
                                      userArgumentsInfo.eSize + \
                                      userArgumentsInfo.activationSize

        writer.states.userArgsInfo = userArgumentsInfo

        self.addOptConfigComment(signature,
                                tt=[kernel["ThreadTile0"], kernel["ThreadTile1"]],
                                sg=[kernel["SubGroup0"], kernel["SubGroup1"]],
                                vwA=kernel["VectorWidthA"],
                                vwB=kernel["VectorWidthB"],
                                glvwA=kernel["GlobalReadVectorWidthA"],
                                glvwB=kernel["GlobalReadVectorWidthB"],
                                d2lA=kernel["DirectToLdsA"],
                                d2lB=kernel["DirectToLdsB"],
                                useSgprForGRO=kernel["_UseSgprForGRO"])

        return signature

    def addOptConfigComment(self, signature: SignatureBase, tt, sg, vwA, vwB, glvwA, glvwB, d2lA, d2lB, useSgprForGRO):
        signature.addDescriptionTopic("Optimizations and Config:")
        signature.addDescriptionBlock("ThreadTile= %u x %u" % (tt[0], tt[1]) )
        signature.addDescriptionBlock("SubGroup= %u x %u" % (sg[0], sg[1]) )
        signature.addDescriptionBlock("VectorWidthA=%u" % vwA )
        signature.addDescriptionBlock("VectorWidthB=%u" % vwB )
        signature.addDescriptionBlock("GlobalReadVectorWidthA=%u, GlobalReadVectorWidthB=%u" % (glvwA, glvwB) )
        signature.addDescriptionBlock("DirectToLdsA=%s" % d2lA )
        signature.addDescriptionBlock("DirectToLdsB=%s" % d2lB )
        signature.addDescriptionBlock("UseSgprForGRO=%s" % useSgprForGRO )
