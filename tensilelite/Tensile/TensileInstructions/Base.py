################################################################################
#
# Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

from .Formatting import __TI_DEBUG_LEVEL__, printExit

from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple

import pickle
import subprocess
import threading

def fastdeepcopy(x):
    # Note: Some object can't be pickled
    return pickle.loads(pickle.dumps(x))

class TensileInstructions:

    _instance  = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._isaInfo = {}  # type: ignore
                cls._instance._kernelInfo = {}
        return cls._instance

    @dataclass
    class IsaInfo:
        assemblerPath: str
        asmCaps: dict
        archCaps: dict
        asmBugs: dict

    @dataclass
    class kernelInfo:
        isa: Tuple[int, int, int]
        wavefrontSize: int = 64

    def init(self, isaVersion: Tuple[int, int, int], assemblerPath: str, debug: bool=False) -> None:
        with self._lock:
            if len(self._kernelInfo) > 1000:
                self._kernelInfo = _removeIdent(self._kernelInfo)
            self._kernelInfo[threading.get_ident()] = TensileInstructions.kernelInfo(isa=isaVersion)
            if isaVersion not in self._isaInfo: # type: ignore
                asmCaps  = _initAsmCaps(isaVersion, assemblerPath, debug)
                archCaps = _initArchCaps(isaVersion)
                asmBugs  = _initAsmBugs(asmCaps)
                self._isaInfo[isaVersion] = TensileInstructions.IsaInfo(assemblerPath, # type: ignore
                    asmCaps, archCaps, asmBugs)

    def setDebugLevel(self, level: int) -> None:
        __TI_DEBUG_LEVEL__ = level

    def setKernelInfo(self, isaVersion: Tuple[int, int, int], wavefrontSize: int) -> None:
        if isaVersion not in self._isaInfo: # type: ignore
            printExit("Current isa %s not initialized. Initialized isas are %s"%(str(isaVersion), str(self._isaInfo.keys())))
        with self._lock:
            if len(self._kernelInfo) > 1000:
                self._kernelInfo = _removeIdent(self._kernelInfo)
            tid = threading.get_ident()
            if tid not in self._kernelInfo:
                self._kernelInfo[threading.get_ident()] = \
                    TensileInstructions.kernelInfo(isa=isaVersion, wavefrontSize=wavefrontSize)
            else:
                self._kernelInfo[threading.get_ident()].isa           = isaVersion
                self._kernelInfo[threading.get_ident()].wavefrontSize = wavefrontSize

    def getCurrentIsa(self) -> Tuple[int]:
        return self._kernelInfo[threading.get_ident()].isa

    def getAsmCaps(self) -> dict:
        return self._isaInfo[self._kernelInfo[threading.get_ident()].isa].asmCaps # type: ignore

    def getArchCaps(self) -> dict:
        return self._isaInfo[self._kernelInfo[threading.get_ident()].isa].archCaps # type: ignore

    def getAsmBugs(self) -> dict:
        return self._isaInfo[self._kernelInfo[threading.get_ident()].isa].asmBugs # type: ignore

    def getKernel(self) -> kernelInfo:
        return self._kernelInfo[threading.get_ident()]

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
_global_ti = TensileInstructions()

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
    def asmBugs(self) -> dict:
        return _global_ti.getAsmBugs()

    @property
    def kernel(self) -> TensileInstructions.kernelInfo:
        return _global_ti.getKernel()

    def countType(self, ttype) -> int:
        return int(isinstance(self, ttype))

    def prettyPrint(self, indent="") -> str:
        ostream = ""
        ostream += "%s%s "%(indent, type(self).__name__)
        ostream += str(self)
        return ostream

def getGfxName(arch: Tuple[int, int, int]) -> str:
    # convert last digit to hex because reasons
    name = str(arch[0]) + str(arch[1]) + ('%x' % arch[2])
    return 'gfx' + ''.join(map(str,name))

def _removeIdent(isaDict) -> list:
    ids = [th.ident for th in threading.enumerate()]
    isaDict = [id for id in isaDict if id in ids]
    return isaDict

def _tryAssembler(isaVersion: Tuple[int, int, int], assemblerPath: str, asmString: str, \
                debug: bool=False, *options) -> bool:
    """
    Try to assemble the asmString for the specified target processor
    Success is defined as assembler returning no error code or stderr/stdout
    """
    options = list(options)

    if isaVersion[0] >= 10:
        options += ['-mwavefrontsize64']

    args = [assemblerPath, '-x', 'assembler',
            '-target', 'amdgcn-amdhsa',
            '-mcpu='+ getGfxName(isaVersion),
            '-mcode-object-version=3',
            *options,
            '-']

    result = subprocess.run(args, input=asmString.encode(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = result.stdout.decode()

    if debug:
        print("isaVersion: ", isaVersion)
        print("asm_cmd:", ' '.join(args))
        print("asmString: ", asmString)
        print("output: ", output)
        print("return code: ", result.returncode)

    if output != "" or result.returncode != 0:
        return False
    return True

########################################
# Get Caps
########################################

def _initAsmCaps(isaVersion, assemblerPath, isDebug) -> dict:
    """ Determine assembler capabilities by testing short instructions sequences """
    rv = {}
    rv["SupportedISA"]      = _tryAssembler(isaVersion, assemblerPath, "", isDebug)
    rv["HasExplicitCO"]     = _tryAssembler(isaVersion, assemblerPath, "v_add_co_u32 v0,vcc,v0,1", isDebug)
    rv["HasExplicitNC"]     = _tryAssembler(isaVersion, assemblerPath, "v_add_nc_u32 v0,v0,1", isDebug)

    rv["HasDirectToLds"]    = _tryAssembler(isaVersion, assemblerPath, "buffer_load_dword v40, v36, s[24:27], s28 offen offset:0 lds", isDebug) \
                                or _tryAssembler(isaVersion, assemblerPath, "buffer_load_b32 v40, v36, s[24:27], s28 offen offset:0 lds", isDebug)
    rv["HasAddLshl"]        = _tryAssembler(isaVersion, assemblerPath, "v_add_lshl_u32 v47, v36, v34, 0x2", isDebug)
    rv["HasLshlOr"]         = _tryAssembler(isaVersion, assemblerPath, "v_lshl_or_b32 v47, v36, 0x2, v34", isDebug)
    rv["HasSMulHi"]         = _tryAssembler(isaVersion, assemblerPath, "s_mul_hi_u32 s47, s36, s34", isDebug)
    rv["HasCodeObjectV3"]   = _tryAssembler(isaVersion, assemblerPath, "", isDebug, "-mcode-object-version=2")

    rv["HasMFMA"]           = _tryAssembler(isaVersion, assemblerPath, "v_mfma_f32_32x32x2bf16 a[0:31], v32, v33, a[0:31]", isDebug)
    rv["HasMFMA_f64"]       = _tryAssembler(isaVersion, assemblerPath, "v_mfma_f64_16x16x4f64 v[0:7], v[32:33], v[36:37], v[0:7]", isDebug)
    rv["HasMFMA_bf16_1k"]   = _tryAssembler(isaVersion, assemblerPath, "v_mfma_f32_32x32x4bf16_1k a[0:31], v[32:33], v[36:37], a[0:31]", isDebug)

    rv["v_mac_f16"]         = _tryAssembler(isaVersion, assemblerPath, "v_mac_f16 v47, v36, v34", isDebug)

    rv["v_fma_f16"]         = _tryAssembler(isaVersion, assemblerPath, "v_fma_f16 v47, v36, v34, v47, op_sel:[0,0,0,0]", isDebug)
    rv["v_fmac_f16"]        = _tryAssembler(isaVersion, assemblerPath, "v_fma_f16 v47, v36, v34", isDebug)

    rv["v_pk_fma_f16"]      = _tryAssembler(isaVersion, assemblerPath, "v_pk_fma_f16 v47, v36, v34, v47, op_sel:[0,0,0]", isDebug)
    rv["v_pk_fmac_f16"]     = _tryAssembler(isaVersion, assemblerPath, "v_pk_fma_f16 v47, v36, v34", isDebug)

    rv["v_pk_add_f32"]      = _tryAssembler(isaVersion, assemblerPath, "v_pk_add_f32 v[48:49], v[36:37], v[0:1]", isDebug)

    rv["v_mad_mix_f32"]     = _tryAssembler(isaVersion, assemblerPath, "v_mad_mix_f32 v47, v36, v34, v47, op_sel:[0,0,0] op_sel_hi:[1,1,0]", isDebug)
    rv["v_fma_mix_f32"]     = _tryAssembler(isaVersion, assemblerPath, "v_fma_mix_f32 v47, v36, v34, v47, op_sel:[0,0,0] op_sel_hi:[1,1,0]", isDebug)

    rv["v_dot2_f32_f16"]    = _tryAssembler(isaVersion, assemblerPath, "v_dot2_f32_f16 v20, v36, v34, v20", isDebug)
    rv["v_dot2c_f32_f16"]   = _tryAssembler(isaVersion, assemblerPath, "v_dot2c_f32_f16 v47, v36, v34", isDebug) \
                                or _tryAssembler(isaVersion, assemblerPath, "v_dot2acc_f32_f16 v47, v36, v34", isDebug)

    rv["v_dot4_i32_i8"]     = _tryAssembler(isaVersion, assemblerPath, "v_dot4_i32_i8 v47, v36, v34", isDebug)
    rv["v_dot4c_i32_i8"]    = _tryAssembler(isaVersion, assemblerPath, "v_dot4c_i32_i8 v47, v36, v34", isDebug)
    rv["VOP3v_dot4_i32_i8"] = _tryAssembler(isaVersion, assemblerPath, "v_dot4_i32_i8 v47, v36, v34, v47", isDebug)

    rv["v_mac_f32"]         = _tryAssembler(isaVersion, assemblerPath, "v_mac_f32 v20, v21, v22", isDebug)
    rv["v_fma_f32"]         = _tryAssembler(isaVersion, assemblerPath, "v_fma_f32 v20, v21, v22, v23", isDebug)
    rv["v_fmac_f32"]        = _tryAssembler(isaVersion, assemblerPath, "v_fmac_f32 v20, v21, v22", isDebug)

    rv["v_fma_f64"]         = _tryAssembler(isaVersion, assemblerPath, "v_fma_f64 v[20:21], v[22:23], v[24:25], v[20:21]", isDebug)

    rv["HasAtomicAdd"]      = _tryAssembler(isaVersion, assemblerPath, "buffer_atomic_add_f32 v0, v1, s[0:3], 0 offen offset:0", isDebug)


    if _tryAssembler(isaVersion, assemblerPath, "s_waitcnt vmcnt(63)", isDebug):
        rv["MaxVmcnt"] = 63
    elif _tryAssembler(isaVersion, assemblerPath, "s_waitcnt vmcnt(15)", isDebug):
        rv["MaxVmcnt"] = 15
    else:
        rv["MaxVmcnt"] = 0

    # TODO- Need to query the max cap, just like vmcnt as well?
    rv["MaxLgkmcnt"] = 15

    rv["SupportedSource"] = True

    return rv

def _initArchCaps(isaVersion) -> dict:
    rv = {}
    rv["HasEccHalf"]         = (isaVersion==(9,0,6) or isaVersion==(9,0,8) or isaVersion==(9,0,10))
    rv["Waitcnt0Disabled"]   = (isaVersion == (9,0,8) or isaVersion==(9,0,10))
    rv["SeparateVscnt"]      = isaVersion[0] in (10, 11)
    rv["CMPXWritesSGPR"]     = isaVersion[0] not in (10, 11)
    rv["HasWave32"]          = isaVersion[0] in (10, 11)
    rv["HasAccCD"]           = (isaVersion==(9,0,10))
    rv["ArchAccUnifiedRegs"] = (isaVersion==(9,0,10))

    return rv

def _initAsmBugs(asmCaps) -> dict:
    rv = {}
    rv["ExplicitCO"] = asmCaps["HasExplicitCO"]
    rv["ExplicitNC"] = asmCaps["HasExplicitNC"]

    return rv
