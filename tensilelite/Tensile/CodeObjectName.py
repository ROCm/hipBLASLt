

from Tensile.Properties import Predicate
from Tensile.Contractions import ProblemType


def codeObjectFileBaseName(d: dict, useLazyLibraryLoading: bool=True):
    """Compute the code object library name from the solution metadata.

    Args:
        d: Solution metadata used to create the code object name.
        lazyLibary: Use naming conventions necessary for lazy library loading.

    Returns:
        A string with the.
    """
    placeholderName = "TensileLibrary"
    problemType = ProblemType.FromOriginalState(d["ProblemType"])
    if useLazyLibraryLoading:
        placeholderName += '_' + str(problemType.aType) + str(problemType.bType)
        placeholderName += '_' + str(problemType.cType) + str(problemType.computeInputType)
        if problemType.activationType != 'none':
            if str(problemType.activationType).upper() == 'ALL':
                placeholderName += "_A"
            elif str(problemType.activationType).upper() == 'HIPBLASLT_ALL':
                placeholderName += "_HA"
            else:
                placeholderName += "_%s"%str(problemType.activationType).upper()
        if problemType.swizzleTensorA:
            placeholderName += '_STA'
        if problemType.swizzleTensorB:
            placeholderName += '_STB'
        if problemType.useBias:
            placeholderName += '_Bias'
        if problemType.useE:
            placeholderName += '_Grad' if problemType.useGradient else '_Aux'
        if problemType.groupedGemm:
            placeholderName += "_GG"
        else:
            placeholderName += "" if problemType.stridedBatched else "_GB" # legacy
        if problemType.useScaleAB == "Scalar":
            placeholderName += '_SAB'
        elif problemType.useScaleAB == "Vector":
            placeholderName += '_SABV'
        if problemType.useScaleCD:
            placeholderName += '_SCD'
        if problemType.useScaleAlphaVec:
            placeholderName += '_SAV'
        if problemType.sparse:
            placeholderName += '_SPB' if problemType.sparse == 2 else '_SPA'
        if not problemType.f32XdlMathOp.isSingle() and problemType.computeInputType.isSingle():
            placeholderName += '_M' + str(problemType.f32XdlMathOp)
        if problemType.supportDeviceUserArguments:
            placeholderName += '_UA'

        placeholderName += problemType.placeholderStr(includeBatch=True, includeType=True)

        if d.get("PerfMetric", "DeviceEfficiency") != "DeviceEfficiency":
            predicate = Predicate(tag=d["PerfMetric"])
        else:
            predicate = Predicate(tag="TruePred")
        if predicate.tag != "TruePred":
            placeholderName += "_" + predicate.tag

        operationID = problemType.operationIdentifier
        placeholderName += "_" + operationID

        devicePart = d["ArchitectureName"]
        cuCount = d["CUCount"]
        if cuCount: placeholderName += "_CU" + str(cuCount)
        placeholderName += "_" + str(devicePart)

    return placeholderName