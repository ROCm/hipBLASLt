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

from argparse import ArgumentParser
from typing import Any, Dict

from Tensile.Toolchain.Validators import ToolchainDefaults


def parseArguments() -> Dict[str, Any]:
    """
    Returns:
        A dictionary containing the keys representing options and their values.
    """

    argParser = ArgumentParser(
        description="TensileValidateLogic runs critical checks to ensure the "
        "integrity of the supplied logic files.",
    )

    argParser.add_argument("LogicPath", help="Path to LibraryLogic.yaml files.")
    argParser.add_argument(
        "--check-matrix-instruction",
        dest="CheckMatrixInstruction",
        action="store_true",
        help="Checks that matrix instructions are valid for all target ISAs.",
    )
    argParser.add_argument(
        "--jobs",
        "-j",
        dest="Jobs",
        action="store",
        default=48,
        help="Number of worker processes to use during validation checks.",
    )
    argParser.add_argument(
        "--cxx-compiler",
        dest="CxxCompiler",
        action="store",
        default=ToolchainDefaults.CXX_COMPILER,
        help=f"Default: {ToolchainDefaults.CXX_COMPILER}",
    )
    args = argParser.parse_args()

    return args
