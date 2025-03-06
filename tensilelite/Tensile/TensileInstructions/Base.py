################################################################################
#
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

from rocisa import rocIsa
from rocisa.base import KernelInfo

import pickle
import threading

from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple

from ..Common import initAsmCaps, initArchCaps, initRegisterCaps, initAsmBugs
from .Formatting import __TI_DEBUG_LEVEL__, printExit


def fastdeepcopy(x):
    # Note: Some object can't be pickled
    return pickle.loads(pickle.dumps(x))

def printItemList(listOfItems, tag="__unnamed__") -> None:
    header = "="*40
    print("%s\nbegin list %s\n%s"%(header, tag, header))
    for i, item in enumerate(listOfItems):
        item = list(item) if isinstance(item, tuple) else [item]
        print("list[%s] %s"%(i, "-"*30))
        for j, t in enumerate(item):
            ostream = t.prettyPrint()
            ostream = ostream[:-1] if len(ostream)>0 and ostream[-1:] == '\n' else ostream
            print(ostream)
    print("%s\nend list %s\n%s"%(header, tag, header))

# Global
_global_ti = rocIsa.getInstance()

class Item:
    """
    Base class for Modules, Instructions, etc
    Item is a atomic collection of or more instructions and commentsA
    """

    def __init__(self, name: str="") -> None:
        self.parent = ""
        self.name = name

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    @property
    def asmCaps(self) -> dict:
        return _global_ti.getAsmCaps()

    @property
    def archCaps(self) -> dict:
        return _global_ti.getArchCaps()
    
    @property
    def regCaps(self) -> dict:
        return _global_ti.getRegCaps()

    @property
    def asmBugs(self) -> dict:
        return _global_ti.getAsmBugs()

    @property
    def kernel(self) -> KernelInfo:
        return _global_ti.getKernel()

    def countType(self, ttype) -> int:
        return int(isinstance(self, ttype))

    def prettyPrint(self, indent="") -> str:
        ostream = ""
        ostream += "%s%s "%(indent, type(self).__name__)
        ostream += str(self)
        return ostream

def getGlcBitName(hasGLCModifier):
  if hasGLCModifier:
    return "glc"
  return "sc0"

def getSlcBitName(hasGLCModifier):
  if hasGLCModifier:
    return "slc"
  return "sc1"

def _removeIdent(isaDict) -> list:
    ids = [th.ident for th in threading.enumerate()]
    isaDict = [id for id in isaDict if id in ids]
    return isaDict
