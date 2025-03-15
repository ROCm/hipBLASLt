import yaml
import functools
from pathlib import Path

from Tensile.Common import globalParameters, assignGlobalParameters, ParallelMap2
from Tensile.LibraryIO import readYAML
from Tensile.Toolchain.Validators import validateToolchain

from .ParseArguments import parseArguments
from .ValidMatrixInstruction import validateMatrixInstruction


def getParams(cxxCompiler):
    gp = globalParameters

    gpcache = Path.cwd() / "gpcache.yaml"
    if gpcache.exists():
        with open(gpcache, "r") as f:
            gp = yaml.load(f, yaml.CSafeLoader)
    else:
        assignGlobalParameters({}, cxxCompiler)
        with open(gpcache, "w") as f:
            yaml.dump(gp, f, yaml.CSafeDumper)

    return gp


def runChecks(logicPath, gp, file):
    if "Experimental" in file.parts:
        return 0, 0

    keep, total = 0, 0
    solutions = readYAML(file)[5]  # Solutions are the 5th index
    for s in solutions:
        total += 1
        keep += validateMatrixInstruction(s, file.relative_to(logicPath), gp)
    print(f">> {file.relative_to(logicPath)}")
    return keep, total


def main():
    args = parseArguments()
    cxxCompiler = validateToolchain(args.CxxCompiler)
    gp = getParams(cxxCompiler)

    logicPath = Path(args.LogicPath)
    pattern = "**/*.yaml"
    files = logicPath.glob(pattern)
    print(f"Checking logic files with glob {args.LogicPath}{pattern}")

    if not any([args.CheckMatrixInstruction]):
        print("No checks specified. Exiting.")
        exit(0)

    fn = functools.partial(runChecks, logicPath, gp)
    results = ParallelMap2(fn, files, multiArg=False, procs=args.Jobs)

    keep = sum([x[0] for x in results])
    total = sum([x[1] for x in results])

    rejects = total - keep
    print(f"Total  {total} solutions")
    print(f"Keep   {keep} solutions")
    print(f"Reject {rejects} solutions")

    if rejects > 0:
        exit(1)
