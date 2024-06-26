################################################################################
#
# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

from ..TensileInstructions import Module, Label, VMovB32, vgpr
from ..Component import Component
import abc

class PersistentLoop(Component):
    """
    Persistent loop code.
    """
    def __call__(self):
        assert(0)

    @abc.abstractmethod
    def openPersistentLoop(self, writer, kernel):
        pass

    @abc.abstractmethod
    def recalcLocalWriteAddresses(self, writer, kernel, tc):
        pass

    @abc.abstractmethod
    def recalcLocalReadAddressesAB(self, writer, kernel):
        pass


class PersistentLoopOff(PersistentLoop):
    kernel = {"StreamK": 0}

    def openPersistentLoop(self, writer, kernel):
        module = Module("PersistentLoop Off openPersistentLoop")
        return module
    
    def recalcLocalWriteAddresses(self, writer, kernel, tc):
        module = Module("PersistentLoop Off recalcLocalWriteAddresses")
        return module

    def recalcLocalReadAddressesAB(self, writer, kernel):
        module = Module("PersistentLoop Off recalcLocalReadAddressesAB")
        return module
    

class PersistentLoopOn(PersistentLoop):

    @classmethod
    def matches(cls, writer, debug=False):
        return writer.states.kernel["StreamK"] > 0
    
    def openPersistentLoop(self, writer, kernel):
        module = Module("PersistentLoop On openPersistentLoop")

        # Label start of persistent loop
        module.addComment2("Persistent Loop Start")
        persistentLabel = Label(label="PersistentLoopStart", comment="")
        module.add(persistentLabel)

        # TODO remove?
        # kStr += inst("s_add_u32", sgpr("PersistentLoopIter"), sgpr("PersistentLoopIter"), hex(1), "Inc PersistentLoop Iter")     # Back-up: not needed now
        #kStr += str(Code.WaitCnt(self.version, 0,0,"wait for outstanding stores"))
        return module
    
    def recalcLocalWriteAddresses(self, writer, kernel, tc):
        module = Module("PersistentLoop On recalcLocalWriteAddresses")

        if getattr(writer, "oriLwa%s" % tc) is None:
            setattr(writer, "oriLwa%s" % tc, writer.vgprPool.checkOut(1, "OriLocalWriteddr%s" % tc))
            module.add(VMovB32(dst=vgpr(getattr(writer, "oriLwa%s" % tc)), src=vgpr("LocalWriteAddr%s" % tc), comment="back up LWA for persistent kernel + wider local read"))

        return module
    
    def recalcLocalReadAddressesAB(self, writer, kernel):
        module = Module("PersistentLoop On recalcLocalReadAddressesAB")

        needRecalc = writer.states.numReadsIterCoalescedA > 1 or writer.states.numReadsIterCoalescedB > 1
        # backup LocalReadAddr
        # LdsPad + LBSPP case, need to backup LocalReadAddr even if recalc is not done
        needBackupLRAddr = needRecalc or (kernel["LdsPadA"] and kernel["LdsBlockSizePerPadA"] or kernel["LdsPadB"] and kernel["LdsBlockSizePerPadB"])

        if needBackupLRAddr:
            # need to back-up the LRA before reCalculation for wider local read (when no wlr, no need to do this)
            if writer.oriLraA is None: # and not kernel["DirectToVgprA"]: # no local read code if DirectToVgpr is enabled
                writer.oriLraA = writer.vgprPool.checkOut(1, "OriLocalReadAddrA")
                module.add(VMovB32(dst=vgpr(writer.oriLraA), src=vgpr("LocalReadAddrA"), comment="back up LRA for persistent kernel + wider local read"))
            if writer.oriLraB is None: # and not kernel["DirectToVgprB"]: # no local read code if DirectToVgpr is enabled
                writer.oriLraB = writer.vgprPool.checkOut(1, "OriLocalReadAddrB")
                module.add(VMovB32(dst=vgpr(writer.oriLraB), src=vgpr("LocalReadAddrB"), comment="back up LRA for persistent kernel + wider local read"))

        return module
