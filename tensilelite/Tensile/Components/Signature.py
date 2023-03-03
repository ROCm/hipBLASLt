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
from ..TensileInstructions import SignatureBase
from ..TensileInstructions import SignatureValueKind as SVK

from math import ceil

class SignatureCOV3(Signature):
    kernel = {"CodeObjectVersion": "V3"}

    def __call__(self, writer) -> SignatureBase:
        kernel = writer.states.kernel

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
                                    flatWorkGroupSize=(kernel["SubGroup0"] * kernel["SubGroup1"] * kernel["LocalSplitU"]))

        srcValueType  = kernel["ProblemType"]["DataType"].toNameAbbrev()
        dstValueType  = kernel["ProblemType"]["DestDataType"].toNameAbbrev()
        cptValueType  = kernel["ProblemType"]["ComputeDataType"].toNameAbbrev()
        biasValueType = "void"
        actValueType  = kernel["ProblemType"]["ActivationComputeDataType"].toNameAbbrev()

        if globalParameters["DebugKernel"]:
            signature.addArg("AddressDbg", SVK.SIG_GLOBALBUFFER, "struct", "generic")
        signature.addArg(    "D", SVK.SIG_GLOBALBUFFER, dstValueType, "generic")
        signature.addArg(    "C", SVK.SIG_GLOBALBUFFER, dstValueType, "generic")
        signature.addArg(    "A", SVK.SIG_GLOBALBUFFER, srcValueType, "generic")
        signature.addArg(    "B", SVK.SIG_GLOBALBUFFER, srcValueType, "generic")

        # Note: We use packed f16 if alpha and beta are f16
        if kernel["ProblemType"]["ComputeDataType"].isHalf():
            cptValueType = 'pkf16'
        signature.addArg(   "alpha",        SVK.SIG_VALUE, cptValueType)
        if kernel["ProblemType"]["UseBeta"]:
            signature.addArg("beta",        SVK.SIG_VALUE, cptValueType)
        if kernel["ProblemType"]["UseScaleD"] and (kernel["GlobalSplitU"] == 1):
            signature.addArg("AddressScaleD", SVK.SIG_GLOBALBUFFER, cptValueType, "generic")
        for i in range(0, writer.states.d.numSgprStrides):
            signature.addArg(              "strideD%u"%i, SVK.SIG_VALUE,               "u32")

        for i in range(0, writer.states.c.numSgprStrides):
            signature.addArg(              "strideC%u"%i, SVK.SIG_VALUE,               "u32")

        for i in range(0, writer.states.a.numSgprStrides):
            signature.addArg(              "strideA%u"%i, SVK.SIG_VALUE,               "u32")

        for i in range(0, writer.states.b.numSgprStrides):
            signature.addArg(              "strideB%u"%i, SVK.SIG_VALUE,               "u32")

        for i in range(0, writer.states.numSgprSizesFree):
            signature.addArg(            "SizesFree%u"%i, SVK.SIG_VALUE,               "u32")

        for i in range(0, writer.states.numSgprSizesSum):
            signature.addArg(             "SizesSum%u"%i, SVK.SIG_VALUE,               "u32")

        for idxChar in kernel["PackedC0IdxChars"][:-1]:
            signature.addArg("MagicNumberSize%s"%idxChar, SVK.SIG_VALUE,               "u32")
            signature.addArg( "MagicShiftSize%s"%idxChar, SVK.SIG_VALUE,               "u32")

        signature.addArg(             "OrigStaggerUIter", SVK.SIG_VALUE,              "i32")

        signature.addArg(               "NumWorkGroups0", SVK.SIG_VALUE,              "u32")
        signature.addArg(               "NumWorkGroups1", SVK.SIG_VALUE,              "u32")

        if kernel["WorkGroupMapping"] > 1:
            signature.addArg(                "NumFullBlocks", SVK.SIG_VALUE,              "u32")
            signature.addArg(                "WgmRemainder1", SVK.SIG_VALUE,              "u32")
            signature.addArg(     "MagicNumberWgmRemainder1", SVK.SIG_VALUE,              "u32")
        else:
            signature.addArg(                      "padding", SVK.SIG_VALUE,              "u32")

        if kernel["ProblemType"]["UseBias"] and (kernel["GlobalSplitU"] == 1):
            signature.addArg("bias", SVK.SIG_GLOBALBUFFER, biasValueType, "generic")

        if kernel["ProblemType"]["UseE"] and (kernel["GlobalSplitU"] == 1):
            signature.addArg(      "E", SVK.SIG_GLOBALBUFFER, cptValueType, "generic")

        if kernel["ProblemType"]["UseBias"] and (kernel["GlobalSplitU"] == 1):
            signature.addArg("biasType",    SVK.SIG_VALUE,        "u32")

        if kernel["ProblemType"]["UseE"] and (kernel["GlobalSplitU"] == 1):
            for i in range(0, writer.states.e.numSgprStrides):
                signature.addArg("StrideE%u"%i,        SVK.SIG_VALUE,        "u32")


        if ((kernel["ProblemType"]["ActivationType"] != 'none') and (kernel["GlobalSplitU"] == 1) \
            and kernel["ActivationFused"]):
            if kernel["ProblemType"]["ActivationComputeDataType"].isHalf():
                actValueType = 'pkf16'
            for name in kernel["ProblemType"]["ActivationType"].getAdditionalArgStringList():
                signature.addArg(                   name, SVK.SIG_VALUE,        actValueType)
            if kernel["ProblemType"]["ActivationType"] == 'all':
                signature.addArg(       "activationType", SVK.SIG_VALUE,               "u32")

        self.addOptConfigComment(signature,
                                tt=[kernel["ThreadTile0"], kernel["ThreadTile1"]],
                                sg=[kernel["SubGroup0"], kernel["SubGroup1"]],
                                vw=kernel["VectorWidth"],
                                glvwA=kernel["GlobalLoadVectorWidthA"],
                                glvwB=kernel["GlobalLoadVectorWidthB"],
                                d2lA=kernel["DirectToLdsA"],
                                d2lB=kernel["DirectToLdsB"],
                                useSgprForGRO=kernel["_UseSgprForGRO"])

        return signature

    def addOptConfigComment(self, signature: SignatureBase, tt, sg, vw, glvwA, glvwB, d2lA, d2lB, useSgprForGRO):
        signature.addDescriptionTopic("Optimizations and Config:")
        signature.addDescriptionBlock("ThreadTile= %u x %u" % (tt[0], tt[1]) )
        signature.addDescriptionBlock("SubGroup= %u x %u" % (sg[0], sg[1]) )
        signature.addDescriptionBlock("VectorWidth=%u" % vw )
        signature.addDescriptionBlock("GlobalLoadVectorWidthA=%u, GlobalLoadVectorWidthB=%u" % (glvwA, glvwB) )
        signature.addDescriptionBlock("DirectToLdsA=%s" % d2lA )
        signature.addDescriptionBlock("DirectToLdsB=%s" % d2lB )
        signature.addDescriptionBlock("UseSgprForGRO=%s" % useSgprForGRO )
