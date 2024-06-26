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

from ..TensileInstructions import Module, Label
from ..Component import Component
import abc

class PersistentLoop(Component):
    """
    Persistent loop code.
    """
    @abc.abstractmethod
    def openPersistentLoop(self, writer, kernel):
        pass

class PersistentLoopOff(PersistentLoop):
    kernel = {"StreamK": 0}

    def openPersistentLoop(self, writer, kernel):
        module = Module("PersistentLoop Off openPersistentLoop")
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
