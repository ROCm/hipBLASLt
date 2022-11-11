################################################################################
# Copyright 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
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

from . import Common
from .TensileInstructions import Item, TensileInstructions, slash50, replaceHolder, \
                          KernelBody, Module, StructuredModule, TextBlock, \
                          TensileInstructionsPass
from .TensileInstructions.Instructions import *
from .TensilePass import TensilePass, TensilePassOptions
from .Common import globalParameters, CHeader, roundUp, Backup, print2, printExit
from .Component import Component
from .CustomKernels import isCustomKernelConfig
from .SolutionStructs import Solution

import abc
import os
import subprocess
import copy
from typing import NamedTuple

################################################################################
# Kernel Writer
################################################################################
class KernelWriter(metaclass=abc.ABCMeta):
  #__metaclass__=abc.ABCMeta

  ##############################################################################
  # Init
  ##############################################################################
  def __init__( self, kernelMinNaming, kernelSerialNaming ):
    self.kernelMinNaming = kernelMinNaming
    self.kernelSerialNaming = kernelSerialNaming
    self.overflowedResources = 0

  @property
  def asmCaps(self):
    """
    Assembler capabilities for the current ISA version.
    """
    return globalParameters["AsmCaps"][self.version]

  @property
  def archCaps(self):
    """
    Architectural capabilities for the current ISA version.
    """
    return globalParameters["ArchCaps"][self.version]

  @property
  def asmBugs(self):
    """
    Assembler bugs for the current ISA version.
    """
    return globalParameters["AsmBugs"][self.version]

  @property
  def globalParams(self):
    """
    Global parameters for current configuration.
    """
    return globalParameters

  ##############################################################################
  # makeSchedule:  Schedule work into interations.

  # Tensile uses a two-level scheduler.  This the first-level, which
  # schedules global reads, global incs, and local writes into iteration.
  # Then makeSubIterSchedule schedules the instructions within the iteration.
  #
  # Inputs:
  #   localWriteEndIter: loop iteration where last writes should be inserted
  #      If scheduleLocalWrite=0, all writes will be be placed in this iteration.
  #      If scheduleLocalWrite=1, the scheduler will work backwards from this
  #      iteration.
  #
  # Outputs:
  #   self.unrollLoopHeaderCode:
  #      - Code module that should be added into the unroll loop header
  #        In unscheduled code this contains global loads and global address increment
  #   self.perIterGlobalReadCode[], self.perIterLocalWriteCode[]
  #      - List indexed by unroll iteration.
  #        Each entry in the list is a code module that should be added into that iteration.
  #        May be None, indicating no extra code for that iteration
  #   self.grEndMfmaIndex
  #   self.lwStartMfmaIndex
  #   self.lwEndMfmaIndex
  #   self.barrierMfmaIndex
  #   self.numMfmaForNextLoopLR
  # This routine is responsible for setting the schedule including determining
  # that all necessary dependency are met.  The driver code in kernelBody
  # blindly follows the plan set in unrollLoopHeaderCode and perIterCode
  ##############################################################################
  def makeSchedule(self, kernel, tensorParametersA, tensorParametersB, localWriteEndIter, uDu=0, skipGlobalReadInc=False, firstIter=False, lastLoop=False, lastLc=False):

    currentIsa = globalParameters["CurrentISA"]
    maxVmcnt = globalParameters["AsmCaps"][currentIsa]["MaxVmcnt"]

    self.unrollLoopHeaderCode = Module()
    # schedule of work for each local_read iteration:
    self.perIterGlobalReadCode = [ Module() for i in range (kernel["LoopIters"]) ]
    self.perIterLocalWriteCode = [ Module() for i in range (kernel["LoopIters"]) ]
    if lastLc:
      self.perIterLocalWriteCodeNGLL = [ Module() for i in range (kernel["LoopIters"]) ]
    self.perIterLocalWriteCanSkip = [ 0 for i in range (kernel["LoopIters"]) ]
    self.perIterGlobalReadCodeDTV = [ Module() for i in range (kernel["LoopIters"]) ] # global read for DirectToVgpr
    assert([item.name for item in self.globalReadIncrements.itemList] == ['globalReadIncrementA', 'globalReadIncrementB'])

    globalReadIncACode  = self.globalReadIncrements.findNamedItem("globalReadIncrementA")
    globalReadIncBCode  = self.globalReadIncrements.findNamedItem("globalReadIncrementB")

    if uDu < kernel["DepthULdsDivisor"] - 1 and kernel.enabledSplitLDS and kernel["PrefetchGlobalRead"] \
       or skipGlobalReadInc:
      globalReadIncACode  = Module()
      globalReadIncBCode  = Module()

    grBackup = None
    if uDu != kernel["DepthULdsDivisor"] - 2 and kernel.enabledSplitLDS:
      # hack RAII object for auto restore
      # withhold issuing global read codes until in the 2nd last subloop, meaning we empty the code
      # modules in other subloops.
      grBackup = Backup(self, globalReadACode = self.globalReadACode, globalReadBCode = self.globalReadBCode)
      self.globalReadACode = StructuredModule() # empty
      self.globalReadBCode = StructuredModule() # empty

    siaComponent = Component.SIA.find(self)
    siaComponent.schedIntoIteration(self, kernel, tensorParametersA, tensorParametersB, \
      localWriteEndIter, uDu, firstIter, lastLoop, lastLc, maxVmcnt, globalReadIncACode, \
      globalReadIncBCode)

    if grBackup is not None:
      del grBackup

  ##############################################################################
  # Schedule work into the each unroll loop iteration
  # localReadCode is the local reads for this loop iteration
  #  (returned by localReadDo). The instructions in localReadCode
  #  will retain their relative order, but may be interleaved
  #  with instructions from otherCode.

  # globalReadCode is the 'other' buffer loads and addr increments
  # localWriteCode is the 'other' local writes
  #  to schedule in with the ds reads.  The instructions
  #  will retain their relative order, but may be interleaved
  #  with instructions from localReadCode.

  # pointerCode contains local pointer changes (if needed)
  # waitCode contains s_waitcnt before macs.
  #   - Cannot be "" or None
  #   - may be empty Module if not waiting is desired (perhaps for debug)
  #   - may be multiple instructions (ConservativeWaitCnt)
  #   - typically is a single SWaitCnt.  This routine will
  #     modify the lgkmcnt to account for any scheduling decisions.
  #     If this is not desired, add the waitCnt to pointerCode and
  #     set waitCode to an empty module
  # macIterCode contains the mac iters.  May be a macro call.
  #
  # returns: a Module with the combined, optimally scheduled
  #  localReadCode + otherCode
  ##############################################################################
  def makeSubIterSchedule(self, kernel, tPA, tPB, localReadCode, iteration, pointerLWCode, pointerLRCode, waitCode, macIterCode, \
      waitLWCode = Module(), syncCode = Module(), packCode = Module(), isDTVodd = False, NLLlast = False):

    iterCode = Module()
    globalReadCode = copy.deepcopy(self.perIterGlobalReadCode[iteration])
    globalReadCodeDTV = self.perIterGlobalReadCodeDTV[iteration]
    origLenGlobalReadCodeDTV = len(list(self.perIterGlobalReadCodeDTV[iteration].items()))
    localWriteCode = self.perIterLocalWriteCode[iteration]
    isBarrier = kernel["LoopIters"] - self.numItersPLR
    hasLocalRead = localReadCode.countType(LocalReadInstruction)
    # Default schedule is other, local reads, then local writes:
    if self.scheduleIterAlg==0:
      # simple schedule, just add the modules in-order
      iterCode.add(globalReadCode)
      iterCode.add(waitLWCode)
      iterCode.add(syncCode)
      iterCode.add(localReadCode)
      iterCode.add(localWriteCode)
      iterCode.add(pointerLWCode)
      iterCode.add(pointerLRCode)
      iterCode.add(waitCode)
      iterCode.add(packCode)
      iterCode.add(macIterCode)
    elif self.scheduleIterAlg == 1:
      iterCode.add(waitLWCode)
      iterCode.add(syncCode)
      #import pdb
      #pdb.set_trace()
      # simple algorithm - do half the reads first:
      readsToSchedule = localReadCode.countType(LocalReadInstruction) / 2
      #localReadCode.prettyPrint()
      readItems = localReadCode.flatitems()
      while readItems:
        item = readItems.pop(0)
        #print "readsToSchedule=", readsToSchedule, "item=", item
        iterCode.add(item)
        readsThisItem = item.countType(LocalReadInstruction)
        if readsThisItem:
          assert readsThisItem==1, "Scheduler assumes 1 read per item"
          readsToSchedule = readsToSchedule - 1
          if readsToSchedule == 0:
            break

      iterCode.add(globalReadCode)

      # add rest of the reads here
      for item in readItems:
        iterCode.add(item)

      #move down write to be the last
      iterCode.add(localWriteCode)
      # tack on the pointer and mac code:
      iterCode.add(pointerLWCode)
      iterCode.add(pointerLRCode)
      iterCode.add(waitCode)
      iterCode.add(packCode)
      iterCode.add(macIterCode)
    elif self.scheduleIterAlg == 2:
    # SIA2 use only 1 iteration and separate compute and fetch by raising compute priority
    # 2 workgroup interleave, while WG0/WG1 doing compute, WG1/WG0 doing fetch
    # EPS need to be 1, or valu instruction will break interleave
      iterCode.add(globalReadCode)
      iterCode.add(waitLWCode)
      iterCode.add(syncCode)
      iterCode.add(localReadCode)
      iterCode.add(waitCode)

      # interleave pack code
      # BF16 or FP16: each packCode is for one 32-bit reg,  1 packing inst: half-to-single x1
      # INT8        : each packCode is for one 32-bit regs, 3 packing inst: byte-to-half x2 + half-to-single x1
      instPerRegPack = 1 / kernel["ProblemType"]["DataType"].numRegisters() - 1
      instPerPack    = int(kernel["MIInputPerThread"] * kernel["ProblemType"]["DataType"].numRegisters() * instPerRegPack)
      packItems = []
      for iui in range(kernel["InnerUnroll"]):
        packINtems = [ [] for j in range(max(self.numReadsIterCoalescedA,self.numReadsIterCoalescedB)) ]
        packA = packCode.findNamedItem("packA_I%s"%(iui))
        packB = packCode.findNamedItem("packB_I%s"%(iui))
        # In case localReadDo not generate pack Module
        # and findNamedItem will return None type
        # TODO: let all type have pack Module
        if not packA:
          packA = Module()
        packAItems = packA.flatitems()
        if not packB:
          packB = Module()
        packBItems = packB.flatitems()
        if packAItems:
          for j in range(self.numReadsIterCoalescedA):
            for n in range(instPerPack):
              packINtems[j].append(packAItems.pop(0))
        if packBItems:
          for j in range(self.numReadsIterCoalescedB):
            for n in range(instPerPack):
              packINtems[j].append(packBItems.pop(0))
        while packAItems:
          for j in range(self.numReadsIterCoalescedA):
            for n in range(instPerPack):
              packINtems[j].append(packAItems.pop(0))
        while packBItems:
          for j in range(self.numReadsIterCoalescedB):
            for n in range(instPerPack):
              packINtems[j].append(packBItems.pop(0))
        for j in range(max(self.numReadsIterCoalescedA,self.numReadsIterCoalescedB)):
          packItems += packINtems.pop(0)

      macIterItem = macIterCode.flatitems()
      # pop the first code which is s_nop 1 for packing
      item = macIterItem.pop(0)

      numMfmaPerIter = self.numMfmaPerIter
      curPackIdx = 0
      packAIdx = 0
      packBIdx = 0

      for i in range(numMfmaPerIter):
        if packItems:
          # how many pack have to be done
          # calculate the data index of this mfma used for A and B
          # if i // kernel["MIWaveTile"][0]==0, mfma will use new A (need to take iu into account)
          # if i % kernel["MIWaveTile"][0]==0, mfma will use new B
          packAIdx += instPerPack if i//(kernel["MIWaveTileA"]+kernel["MIWaveTileA"]*kernel["MIWaveTileB"]*(i//(kernel["MIWaveTileA"]*kernel["MIWaveTileB"]))) == 0 else 0
          packBIdx += instPerPack if i % kernel["MIWaveTileA"] == 0 else 0
          # blockWidth < 1, means 0.5 or 0.25 (BF,H,Int8)
          packAIdx = packAIdx if tPA["localReadInstruction"].blockWidth < 1 else 0
          packBIdx = packBIdx if tPB["localReadInstruction"].blockWidth < 1 else 0
          numPack = (packAIdx + packBIdx)
          iterCode.addComment0("pack scheduling: packAIdx:%u, packBIdx:%u" %(packAIdx,packBIdx))
          # we put 2 pack in each mfma, "2" means A & B
          if packItems:
            for j in range(instPerPack):
              iterCode.add(packItems.pop(0))
              curPackIdx += 1
          if packItems:
            for j in range(instPerPack):
              iterCode.add(packItems.pop(0))
              curPackIdx += 1
          # since packed register need to wait 2 quad cycle to finish packing
          # we insert pack instruction if we can, or s_nop
          while curPackIdx < numPack+2:
            if packItems:
              for j in range(instPerPack):
                iterCode.add(packItems.pop(0))
                curPackIdx += 1
            else:
              iterCode.add(SNop(waitState=0, comment="VALU packing writes to be consumed by matrix instruction"))
              curPackIdx += 1
        if i == 0:
          if not packItems:
            tmpVgpr = self.vgprPool.checkOut(1)
            iterCode.add(VMovB32(dst="v%u"%(tmpVgpr), src="0x0", comment="valu operation to have different priority"))
            self.vgprPool.checkIn(tmpVgpr)
          iterCode.add(SSetPrior(prior=3, comment="Raise priority while processing macs"))
        item = macIterItem.pop(0)
        iterCode.add(item)

      iterCode.add(SSetPrior(prior=1, comment="Raise priority while processing macs"))
      if kernel["1LDSBuffer"]:
        barrier = Module()
        barrier.addComment0("1 LDS buffer: read-sync-write")
        barrier.add(SWaitCnt(lgkmcnt=0, comment=""))
        barrier.add(SBarrier())
        iterCode.add(barrier)
      iterCode.add(localWriteCode)
      iterCode.add(pointerLWCode)
      iterCode.add(pointerLRCode)
      iterCode.add(SSetPrior(prior=2, comment="Raise priority while processing macs"))
      pass
    elif self.scheduleIterAlg == 3:
      iterCode.addComment0(" grEndMfmaIndex:%u, lwStartMfmaIndex:%u, lwEndMfmaIndex:%u " %(self.grEndMfmaIndex,self.lwStartMfmaIndex,self.lwEndMfmaIndex))
      iterCode.addComment0(" numMfmaForLR:%u, barrierMfmaIndex:%u " %(self.numMfmaForNextLoopLR,self.barrierMfmaIndex))
      #####
      # Prepare and Assign parameter
      ####
      if iteration == 0:
        self.localReadsVacancy = []
        self.localReadsWait = [ [] for j in range(kernel["LoopIters"])]
      self.localReadsWait[iteration] = waitCode
      numMfmaPerIter = self.numMfmaPerIter
      isBarrier = kernel["LoopIters"] - self.numItersPLR
      writeItems = list(localWriteCode.items())
      macIterItems = macIterCode.flatitems()
      skipLocalWriteWaitcnt = 0
      localReadsWaitcnt = 0
      curPackIdx = 0
      packAIdx = 0
      packBIdx = 0

      #####
      # Prepare localReadCode
      ####
      localReadCodeAB = Module()
      for iui in range(kernel["InnerUnroll"]):
        localReadCodeA = localReadCode.findNamedItem("LocalReadDoA_I%s"%(iui))
        localReadCodeB = localReadCode.findNamedItem("LocalReadDoB_I%s"%(iui))
        # In case localReadDo not generate localReadCode Module
        # and findNamedItem will return None type
        # TODO: findNamedItem return Module() if not found
        if not localReadCodeA:
          localReadCodeA = Module()
        if not localReadCodeB:
          localReadCodeB = Module()
        if localReadCodeA.items():
          localReadCodeAB.add(localReadCodeA.items().pop(0))
        if localReadCodeB.items():
          localReadCodeAB.add(localReadCodeB.items().pop(0))
        while localReadCodeA.items():
          localReadCodeAB.add(localReadCodeA.items().pop(0))
        while localReadCodeB.items():
          localReadCodeAB.add(localReadCodeB.items().pop(0))
      localReadItems = localReadCodeAB.flatitems()
      localReadItemsThisLoop = localReadItems if iteration < isBarrier else []
      localReadItemsNextLoop = localReadItems if iteration >= isBarrier else []

      #####
      # Prepare pack Code                for B:
      # since the mfma reuse B first =>    for A: mfma[A][B]
      # we need 1 vector A and 1 vector B for first mfma
      # then we prepare remaining A, then remaining B
      # BF16 or FP16: each packCode is for one 32-bit reg,  1 packing inst: half-to-single x1
      # INT8        : each packCode is for one 32-bit regs, 3 packing inst: byte-to-half x2 + half-to-single x1
      ####
      instPerRegPack = 1 / kernel["ProblemType"]["DataType"].numRegisters() - 1
      instPerPack    = int(kernel["MIInputPerThread"] * kernel["ProblemType"]["DataType"].numRegisters() * instPerRegPack)
      packItems = []
      for iui in range(kernel["InnerUnroll"]):
        packINtems = [ [] for j in range(max(self.numReadsIterCoalescedA,self.numReadsIterCoalescedB)) ]
        packA = packCode.findNamedItem("packA_I%s"%(iui))
        packB = packCode.findNamedItem("packB_I%s"%(iui))
        # In case localReadDo not generate pack Module
        # and findNamedItem will return None type
        # TODO: let all type have pack Module
        if not packA:
          packA = Module()
        packAItems = packA.flatitems()
        if not packB:
          packB = Module()
        packBItems = packB.flatitems()
        if packAItems:
          for j in range(self.numReadsIterCoalescedA):
            for n in range(instPerPack):
              packINtems[j].append(packAItems.pop(0))
        if packBItems:
          for j in range(self.numReadsIterCoalescedB):
            for n in range(instPerPack):
              packINtems[j].append(packBItems.pop(0))
        while packAItems:
          for j in range(self.numReadsIterCoalescedA):
            for n in range(instPerPack):
              packINtems[j].append(packAItems.pop(0))
        while packBItems:
          for j in range(self.numReadsIterCoalescedB):
            for n in range(instPerPack):
              packINtems[j].append(packBItems.pop(0))
        for j in range(max(self.numReadsIterCoalescedA,self.numReadsIterCoalescedB)):
          packItems += packINtems.pop(0)

      # remove s_nop for packing
      # we will add s_nop if needed
      if macIterItems:
        macIterItems.pop(0)

      ####
      # scheduled local read to previous iterations
      ####
      if self.numVgprBuffer >= kernel["LoopIters"]:
        for vacancy in self.localReadsVacancy:
          # {"items","latencyLeft","atIter","atMfmaIndex","noReadsAtThisIter"}
          for localRead in list(localReadItemsThisLoop):
            if vacancy["latencyLeft"] > localRead.issueLatency() * 2:
              if not localRead.readToTempVgpr:
                vacancy["latencyLeft"] -= localRead.issueLatency() * 2
                vacancy["items"].add(localRead)
                localReadItemsThisLoop.remove(localRead)
                if vacancy["atMfmaIndex"] > self.lwStartMfmaIndex - 1 and kernel["1LDSBuffer"]:
                  self.overflowedResources = 5
                # update waitCnt
                if self.numItersPLR:
                  for readsIter in range(vacancy["atIter"], iteration + self.numItersPLR):
                    if (vacancy["atMfmaIndex"] % numMfmaPerIter == 0 or readsIter != vacancy["atIter"]) and \
                        (vacancy["noReadsAtThisIter"] or readsIter <= vacancy["atIter"] + self.numItersPLR):
                      if isinstance(self.localReadsWait[readsIter], SWaitCnt):
                        self.localReadsWait[readsIter].lgkmcnt += 1
                        # This line is added for backward compatibility
                        self.localReadsWait[readsIter].vscnt = self.localReadsWait[readsIter].vmcnt \
                          if self.localReadsWait[readsIter].lgkmcnt != -1 and \
                            self.localReadsWait[readsIter].vmcnt != -1 and \
                            self.archCaps["SeparateVscnt"] else -1
            else:
              # make sure the localread sequence remain the same
              vacancy["latencyLeft"] = 0
      numReadsInst = len(localReadItemsThisLoop) if iteration < isBarrier else len(localReadItemsNextLoop)

      for i in range(numMfmaPerIter):
        mfmaIndex = iteration * numMfmaPerIter + i
        iterCode.addComment0(" mfmaIndex:%u " %(mfmaIndex))

        ####
        # scheduled local read
        ####
        readLeft = numReadsInst
        latencyLeft = self.miLatencyLeft
        # with PrefetchLocalRead, localreads can interleave with mfma
        if self.numItersPLR and iteration < isBarrier:
          # take ds_write into account to schedule ds_read, assume A and B localwrite have same width (TLDS=1)
          if (mfmaIndex >= self.lwStartMfmaIndex) and not globalReadCode.countType(GlobalReadInstruction):
            for j in range(min(len(writeItems),self.numLocalWriteModPerMfma)):
              if writeItems[j].countType(LocalWriteInstruction):
                latencyLeft -= (tPA["localWriteInstruction"].issueLatency*2)
          readLeftLROPT = 0
          for j in range(len(localReadItemsThisLoop)):
            latencyLeft -= localReadItemsThisLoop[j].issueLatency()*2
            readLeftLROPT += 1 if latencyLeft >= 0 else 0
          # at least 1 instruction
          readLeftLROPT = max(readLeftLROPT,1)
          # evenly schedule localread with each mfma
          readLeftLREven = numReadsInst // numMfmaPerIter
          if (numReadsInst % (numMfmaPerIter)) > i:
            readLeftLREven += 1
          # we want no localreads at first mfma
          if (iteration == 0) and numMfmaPerIter != 1:
            numMfmaForLR = numMfmaPerIter - 1
            if i < numMfmaPerIter - numMfmaForLR:
              readLeftLREven = 0
              readLeftLROPT = 0
            # rest mfma help to schedule those localReads
            else:
              readLeftLREven = numReadsInst // (numMfmaPerIter-1)
              if (numReadsInst % (numMfmaPerIter-1)) >= i:
                readLeftLREven += 1
          # if there are too many localreads, change strategy to even.
          readLeft = max(readLeftLREven,readLeftLROPT)
        if not self.numItersPLR and iteration < isBarrier:
          for j in range(len(localReadItemsThisLoop)):
            latencyLeft -= localReadItemsThisLoop[j].issueLatency()*2
        # if start to schedule localwrite, but still have localreads not scheduled yet,
        # reject to use 1LDSB, since it will write and read same lds buffer at same time.
        # TODO: force to schedule all remaining localreads before start to schedule localwrite.
        if mfmaIndex >= self.lwStartMfmaIndex and mfmaIndex <= max(self.lwEndMfmaIndex,self.barrierMfmaIndex) and \
          localReadItemsThisLoop and localWriteCode.countType(LocalWriteInstruction) and kernel["1LDSBuffer"]:
          self.overflowedResources = 5
        # DirectToVgpr case, localReadItemsThisLoop and localWriteCode.countType(LocalWriteInstruction) do not satisfy at the same time.
        # However, it is still invaid if localReadItemsThisLoop exists when mfmaIndex > lwStartMfmaIndex
        elif (kernel["DirectToVgprA"] or kernel["DirectToVgprB"]) and \
          mfmaIndex > self.lwStartMfmaIndex and mfmaIndex <= max(self.lwEndMfmaIndex,self.barrierMfmaIndex) and \
          localReadItemsThisLoop and kernel["1LDSBuffer"]:
          self.overflowedResources = 5
        for j in range(readLeft):
          if localReadItemsThisLoop:
            item = localReadItemsThisLoop.pop(0)
            iterCode.add(item)
            if (i == 0):
              localReadsWaitcnt += 1
        if not localReadItemsThisLoop and latencyLeft > 0 and iteration < isBarrier and \
            not(mfmaIndex > self.lwStartMfmaIndex and kernel["1LDSBuffer"]):
          item = Module()
          item.addComment0("localReadsVacancy: latencyLeft %d"%(latencyLeft))
          iterCode.add(item)
          self.localReadsVacancy.append({ "items": item, \
                                          "latencyLeft": latencyLeft, \
                                          "atIter": iteration, \
                                          "atMfmaIndex": mfmaIndex, \
                                          "noReadsAtThisIter": numReadsInst == 0, \
                                        })

        ####
        # scheduled global read
        ####
        for j in range(self.numGlobalReadInsPerMfma):
          if globalReadCode.items():
            loadModule = globalReadCode.items().pop(0)
            if isDTVodd:
              # need to swap Vgpr set for odd code
              loadModule = self.flipVregSetForDirectToVgprInGlobalRead(kernel, loadModule)
            iterCode.add(loadModule)
        # schedule remaining globalReadInst
        if mfmaIndex == self.grEndMfmaIndex:
          while globalReadCode.items() and \
              (globalReadCode.countType(GlobalReadInstruction) or kernel["PrefetchGlobalRead"] == 2):
            loadModule = globalReadCode.items().pop(0)
            if isDTVodd:
              # need to swap Vgpr set for odd code
              loadModule = self.flipVregSetForDirectToVgprInGlobalRead(kernel, loadModule)
            iterCode.add(loadModule)
        # schedule remaining globalReadIncInst
        if i == numMfmaPerIter - 1:
          while globalReadCode.items():
            loadModule = globalReadCode.items().pop(0)
            if isDTVodd:
              # need to swap Vgpr set for odd code
              loadModule = self.flipVregSetForDirectToVgprInGlobalRead(kernel, loadModule)
            iterCode.add(loadModule)

        ####
        # scheduled local write
        ####
        if kernel["1LDSBuffer"] and mfmaIndex == self.lwStartMfmaIndex - 1:
          barrier = Module()
          barrier.addComment0("1 LDS buffer: read-sync-write")
          barrier.add(SWaitCnt(lgkmcnt=0, comment=""))
          barrier.add(SBarrier())
          iterCode.add(barrier)

        if kernel["StorePriorityOpt"]:
          flagInsert = False
          if kernel["PrefetchGlobalRead"] == 2:
            lwStartOffset = 0
            if kernel["DirectToLds"]:
              lwStartOffset = 2
            #  if (mfmaIndex == self.lwStartMfmaIndex or mfmaIndex == self.barrierMfmaIndex+2):
            if (mfmaIndex == self.lwStartMfmaIndex + lwStartOffset or mfmaIndex == self.barrierMfmaIndex+1) :
              flagInsert = True
          elif kernel["PrefetchGlobalRead"] == 1 and numMfmaPerIter >= 4:
            # this setting is good for fixed clock, but not good for auto clock
            #if (mfmaIndex == self.grEndMfmaIndex or mfmaIndex == self.barrierMfmaIndex+1) :
            withGL = (not NLLlast)
            withDTLload = kernel["DirectToLds"] and withGL
            startIndex = 0 if withDTLload else 1
            if (mfmaIndex == startIndex or withGL and mfmaIndex == self.barrierMfmaIndex+1):
              flagInsert = True
          if flagInsert:
            iterCode.add(SSetPrior(prior=3, comment="store optimization"))

        if (mfmaIndex >= self.lwStartMfmaIndex):
          for j in range(self.numLocalWriteModPerMfma):
            # in case there are localWrite and globalread in same iteration
            # we need to make sure globalRead before localWrite
            if writeItems and not globalReadCode.countType(GlobalReadInstruction):
              writeItem = writeItems.pop(0)
              iterCode.add(writeItem)
              # if there is localWrite at first mfma, need to skip it in waitcnt.
              if i == 0:
                skipLocalWriteWaitcnt += writeItem.countType(LocalWriteInstruction)
              if not localReadItemsThisLoop:
                self.perIterLocalWriteCanSkip[iteration] += writeItem.countType(LocalWriteInstruction)
        if mfmaIndex == self.lwEndMfmaIndex:
          while writeItems:
            writeItem = writeItems.pop(0)
            # generate all remaining pre code before the first Store C
            iterCode.add(writeItem)
            if i == 0:
              skipLocalWriteWaitcnt += writeItem.countType(LocalWriteInstruction)
            if not localReadItemsThisLoop:
              self.perIterLocalWriteCanSkip[iteration] += writeItem.countType(LocalWriteInstruction)

        ####
        # scheduled pointer
        ####
        if mfmaIndex == self.lwEndMfmaIndex:
          iterCode.add(pointerLWCode)
        if i == numMfmaPerIter - 1:
          iterCode.add(pointerLRCode)

        ####
        # scheduled sync
        ####
        if mfmaIndex == self.barrierMfmaIndex and self.numItersPLR:
          iterCode.add(waitLWCode)
          iterCode.add(syncCode)

        ####
        # scheduled local read for next loop
        # localReads for next loop should after barrier
        ####
        latencyLeft = self.miLatencyLeft
        if self.numItersPLR and iteration >= isBarrier:
          readLeftLROPT = 0
          for j in range(len(localReadItemsNextLoop)):
            latencyLeft -= localReadItemsNextLoop[j].issueLatency()*2
            readLeftLROPT += 1 if latencyLeft >= 0 else 0
          # at least 1 instruction
          readLeftLROPT = max(readLeftLROPT,1)
          # evenly schedule localread with each mfma
          readLeftLREven = numReadsInst // numMfmaPerIter
          if (numReadsInst % (numMfmaPerIter)) > i:
            readLeftLREven += 1
          # we want no localreads at barrier mfma
          if (iteration == isBarrier) and numMfmaPerIter != 1:
            numMfmaForLR = self.numMfmaForNextLoopLR
            if i < numMfmaPerIter - numMfmaForLR:
              readLeftLREven = 0
              readLeftLROPT = 0
            # rest mfma help to schedule those localReads
            else:
              readLeftLREven = numReadsInst // (numMfmaPerIter-1)
              if (numReadsInst % (numMfmaPerIter-1)) >= i:
                readLeftLREven += 1
          # if there are too many localreads, change strategy to even.
          readLeft = max(readLeftLREven,readLeftLROPT)
        for j in range(readLeft):
          if localReadItemsNextLoop:
            item = localReadItemsNextLoop.pop(0)
            iterCode.add(item)
            if (i == 0):
              localReadsWaitcnt += 1

        ####
        # scheduled wait localReads
        ####
        if i == 0:
          iterCode.add(waitCode)

        ####
        # scheduled pack
        ####
        if packItems:
          # how many pack have to be done
          # calculate the data index of this mfma used for A and B
          # if i // kernel["MIWaveTile"][0]==0, mfma will use new A (need to take iu into account)
          # if i % kernel["MIWaveTile"][0]==0, mfma will use new B
          packAIdx += instPerPack if i//(kernel["MIWaveTileA"]+kernel["MIWaveTileA"]*kernel["MIWaveTileB"]*(i//(kernel["MIWaveTileA"]*kernel["MIWaveTileB"]))) == 0 else 0
          packBIdx += instPerPack if i % kernel["MIWaveTileA"] == 0 else 0
          # blockWidth < 1, means 0.5 or 0.25 (BF,H,Int8)
          packAIdx = packAIdx if tPA["localReadInstruction"].blockWidth < 1 else 0
          packBIdx = packBIdx if tPB["localReadInstruction"].blockWidth < 1 else 0
          numPack = (packAIdx + packBIdx)
          iterCode.addComment0("pack scheduling: packAIdx:%u, packBIdx:%u" %(packAIdx,packBIdx))
          # we put 2 pack in each mfma
          if packItems:
            for j in range(instPerPack):
              iterCode.add(packItems.pop(0))
              curPackIdx += 1
          if packItems:
            for j in range(instPerPack):
              iterCode.add(packItems.pop(0))
              curPackIdx += 1
          # since packed register need to wait 2 quad cycle to finish packing
          # we insert pack instruction if we can, or s_nop
          while curPackIdx < numPack+2:
            if packItems:
              for j in range(instPerPack):
                iterCode.add(packItems.pop(0))
                curPackIdx += 1
            else:
              iterCode.add(SNop(waitState=0, comment="VALU packing writes to be consumed by matrix instruction"))
              curPackIdx += 1
        if i == numMfmaPerIter - 1:
          while packItems:
            iterCode.add(packItems.pop(0))

        ####
        # scheduled mfma
        ####
        iterCode.add(macIterItems.pop(0) if macIterItems else Module())

        ####
        # scheduled global read for DirectToVgpr (PGR=2 only)
        ####
        numLoadVgpr = len(list(globalReadCodeDTV.items()))
        if numLoadVgpr > 0:
          interval = roundUp(numMfmaPerIter / origLenGlobalReadCodeDTV)
          tileIndex = 0 if kernel["DirectToVgprA"] else 1
          if (kernel["MIWaveTile"][tileIndex] // kernel["VectorWidth"]) > 1:
            if kernel["ProblemType"]["DataType"].isDoubleComplex():
              # adjustment for double complex
              # limit the max of interval up to 4 if (kernel["MIWaveTile"][0] // kernel["VectorWidth"]) > 1
              interval = min(4, interval)
            elif kernel["ProblemType"]["DataType"].isDouble():
              # adjustment for double
              # in this case, interval must be 1 to avoid overwritting vreg by global read
              interval = 1
          # DirectToVgprA + TLU=False + VW > 1 case, need to use interval = 1
          if kernel["DirectToVgprA"] and (not kernel["ProblemType"]["TLUA"]) and kernel["VectorWidth"] > 1:
            interval = 1
          # if number of mfma after self.grEndMfmaIndex is smaller than numMfmaPerIter, we need to use smaller interval to insert DTV load.
          # this is to ensure DTV load is generated after lwStartMfmaIndex
          intervalAfterGrEnd = kernel["LoopIters"] * numMfmaPerIter - self.lwStartMfmaIndex
          intervalMfma = min(numMfmaPerIter, intervalAfterGrEnd)
          numInstToInsert = roundUp(origLenGlobalReadCodeDTV / intervalMfma)
          remainingTimesToInsert = roundUp(numLoadVgpr / numInstToInsert)
          insertMfmaIndex = kernel["LoopIters"] * numMfmaPerIter - 1 - interval * (remainingTimesToInsert - 1)
          # avoid insertMfmaIndex getting smaller than (kernel["LoopIters"] - 1) * numMfmaPerIter
          insertMfmaIndex = max(insertMfmaIndex, (kernel["LoopIters"] - 1) * numMfmaPerIter)
          if mfmaIndex == insertMfmaIndex:
            for i in range(min(numLoadVgpr, numInstToInsert)):
              loadDTVModule = globalReadCodeDTV.items().pop(0)
              if isDTVodd:
                # need to swap Vgpr set for odd code
                loadDTVModule = self.flipVregSetForDirectToVgprInGlobalRead(kernel, loadDTVModule)
              iterCode.add(loadDTVModule)

        if kernel["StorePriorityOpt"]:
          flagInsert = False
          if kernel["PrefetchGlobalRead"] == 2:
            #  if (mfmaIndex == self.barrierMfmaIndex or mfmaIndex == (kernel["LoopIters"] * numMfmaPerIter - 1)):
            if (mfmaIndex == self.barrierMfmaIndex - 1 or (not NLLlast) and mfmaIndex == (kernel["LoopIters"] * numMfmaPerIter - 1)) :
                flagInsert = True
          elif kernel["PrefetchGlobalRead"] == 1 and numMfmaPerIter >= 4:
            # this setting is good for fixed clock, but not good for auto clock
            #if (mfmaIndex == mfmaIndex == self.barrierMfmaIndex - 1 or mfmaIndex == (kernel["LoopIters"] * numMfmaPerIter - 1)) :
            insertPos1 = self.grEndMfmaIndex
            if not kernel["NoLdsWriteCode"]:
              insertPos1 = self.lwStartMfmaIndex - 1
            withGL = (not NLLlast)
            if withGL and (mfmaIndex == insertPos1 or (not NLLlast) and mfmaIndex == (kernel["LoopIters"] * numMfmaPerIter - 1)) or \
               (not withGL) and mfmaIndex == (kernel["LoopIters"] * numMfmaPerIter // 2 - 1):
              flagInsert = True
          if flagInsert:
            iterCode.add(SSetPrior(prior=0, comment="store optimization"))
    else:
      assert 0, "Unsupported scheduleIterAlg=%u"%self.scheduleIterAlg

    if isinstance(waitCode, SWaitCnt):

      # Set the waitCount, based on the new iter schedule
      lgkmcnt = waitCode.lgkmcnt
      localReads = 0
      localWrites = 0
      if kernel["EnableMatrixInstruction"]:
        # dataAtIter      : the data we wait is read at which iteration
        # numReadsIter    : in this loop, number of iteration we have read (data used in current loop)
        dataAtIterA = iteration//self.numIterPerCoalescedReadA - self.numItersPLR
        dataAtIterB = iteration//self.numIterPerCoalescedReadB - self.numItersPLR
        numReadsIterA = min(iteration+1, kernel["LoopIters"]//self.numIterPerCoalescedReadA - self.numItersPLR)
        numReadsIterB = min(iteration+1, kernel["LoopIters"]//self.numIterPerCoalescedReadB - self.numItersPLR)
        skipReadsIterA = numReadsIterA - dataAtIterA - 1 if not dataAtIterA < max(dataAtIterA,dataAtIterB) else 0
        skipReadsIterB = numReadsIterB - dataAtIterB - 1 if not dataAtIterB < max(dataAtIterA,dataAtIterB) else 0
        # numPrefetchIter : in this loop, number of prefetch iteration we have read (data used in next loop)
        # currently we have localReadA and localReadB if iteration >= isBarrier
        # some case will not have localReads if PGR=0 or NoLoadLoop
        # known bug: wider localread + numItersPLR>1 may have chance to fail.
        numPrefetchIter = (iteration//(kernel["LoopIters"]-self.numItersPLR))*((iteration+1)-(kernel["LoopIters"]-self.numItersPLR)) if kernel["PrefetchGlobalRead"] else 0
        numPrefetchIter = 0 if iteration >= isBarrier and not hasLocalRead else numPrefetchIter
        skipReadsIterA += numPrefetchIter
        skipReadsIterB += numPrefetchIter
        # here the reads are prefetches so can skip them in the waitcnt
        # how many localreads can skip is based on how many iterations we prefetch.
        localReads += self.numReadsPerIterA * skipReadsIterA + localReads + self.numReadsPerIterB * skipReadsIterB
        # some of localReads is interleaved after waitcnt in SIA3
        if kernel["ScheduleIterAlg"] == 3 and self.numItersPLR and\
          (iteration < numReadsIterA or iteration < numReadsIterB or numPrefetchIter):
          if (iteration < numReadsIterA and not dataAtIterA < max(dataAtIterA,dataAtIterB)) or numPrefetchIter:
            localReads -= self.numReadsPerIterA
          if (iteration < numReadsIterB and not dataAtIterB < max(dataAtIterA,dataAtIterB)) or numPrefetchIter:
            localReads -= self.numReadsPerIterB
          localReads += localReadsWaitcnt
        lgkmcnt += localReads
        iterCode.addComment0("numPrefetchIter=%u" % numPrefetchIter)
        iterCode.addComment0("dataAtIterA=%u numReadsIterA=%u skipReadsIterA=%u readsPerIterA=%u" % (dataAtIterA, numReadsIterA, skipReadsIterA, self.numReadsPerIterA))
        iterCode.addComment0("dataAtIterB=%u numReadsIterB=%u skipReadsIterB=%u readsPerIterB=%u" % (dataAtIterB, numReadsIterB, skipReadsIterB, self.numReadsPerIterB))
        if kernel["ScheduleIterAlg"] == 0 or kernel["ScheduleIterAlg"] == 1:
          for i in range (max(dataAtIterA,dataAtIterB),iteration+1):
            localWrites += self.perIterLocalWriteCode[i].countType(LocalWriteInstruction)
        # ScheduleIterAlg=2, localwrite is after waitCnt, no need to count it's current iteration.
        if kernel["ScheduleIterAlg"] == 3:
          for i in range (max(dataAtIterA,dataAtIterB)+1,iteration):
            localWrites += self.perIterLocalWriteCode[i].countType(LocalWriteInstruction)
          if kernel["ScheduleLocalWrite"] > 0:
            # current iteration localWrite count
            localWrites += skipLocalWriteWaitcnt
            # dataAtIter iteration localWrite count
            if self.numItersPLR:
              skipPreIterLW = self.perIterLocalWriteCanSkip[max(dataAtIterA,dataAtIterB)]
              if kernel["PrefetchGlobalRead"] == 2 and kernel["LocalReadVectorWidth"] == 2 and \
                 (kernel["DirectToVgprA"] or kernel["DirectToVgprB"]):
                # PGR==2 and LRVW==2 and DirectToVgpr enabled case, count local write before max(dataAtIterA,dataAtIterB)
                # NOTE: This logic assumes that local write is scheduled after local read.
                for up in range(max(dataAtIterA,dataAtIterB)):
                  skipPreIterLW += self.perIterLocalWriteCanSkip[up]
              localWrites += skipPreIterLW
        lgkmcnt += localWrites
      else:
        for item in list(iterCode.items()):
          localReads  = item.countType(LocalReadInstruction)
          localWrites = item.countType(LocalWriteInstruction)
          if self.numVgprBuffer:
            # SQ: If PrefetchLocalRead = 1 and DepthU == LocalSplitU, then there is no double
            #  buffering and we must wait for all localReads but not localWrites.
            #  In that case, LoopIters == 1:
            if kernel["LoopIters"] > 1:
              # here the reads are prefetches so can skip them in the waitcnt
              lgkmcnt += localReads
            # and the writes are targetting another section of LDS and are
            # synchronized through a different waitcnt than this one
            # (which is always just before the macs)
            lgkmcnt += localWrites
          else:
            # if UnrollLoopEfficiencyEnable == True  use waitCode passed lgkmCnt
            # else:
            # we need to wait for all preceding reads before the macs
            # so only opportunity for optimization is if the writes are at the end
            if globalParameters["UnrollLoopEfficiencyEnable"]:
              lgkmcnt = waitCode.lgkmcnt
            else:
              if localReads:
                lgkmcnt = 0 # reset to wait for all reads
              else:
                lgkmcnt = localWrites  # this only survives if writes are at the end

      waitCode.comment += " old=%u, new=%u newLW=%u newLR=%u" % (waitCode.lgkmcnt, lgkmcnt,localWrites,localReads)
      waitCode.lgkmcnt = lgkmcnt
      # This line is added for backward compatibility
      waitCode.vscnt = waitCode.vmcnt if waitCode.lgkmcnt != -1 and waitCode.vmcnt != -1 and self.archCaps["SeparateVscnt"] else -1

    return iterCode

  ##############################################################################
  # returns list of modules or text
  ##############################################################################
  def setupNewTile(self, kernel, tensorParametersA, tensorParametersB, isOptNLL=False, forceNoTileCode=False, forceNoGRCode=False):
    module = Module("setupNewTile")

    ####################################
    # Global Read Addresses
    ####################################
    module.addComment2("Begin setupNewTile")

    # work-group assignments
    module.addComment1("global read addresses: work-group")
    if not forceNoTileCode:
      module.add(self.graWorkGroup(kernel))

    self.dontAppendCode = forceNoTileCode
    # tile assignments
    module.addComment1("global read addresses: tile offset assignment a")
    module.add(self.graTileAssignment(kernel, tensorParametersA))
    module.addComment1("global read addresses: tile offset assignment b")
    module.add(self.graTileAssignment(kernel, tensorParametersB))

    # unroll assignments
    module.addComment1("global read addresses: unroll assignment a")
    module.add(self.graUnrollAssignment(kernel, tensorParametersA))
    module.addComment1("global read addresses: unroll assignment b")
    module.add(self.graUnrollAssignment(kernel, tensorParametersB))

    # other free indices
    if kernel["ProblemType"]["NumIndicesC"] > 2:
      module.addComment1("global read addresses: other free assignments")
      module.add(self.graOtherFreeAssignments())

    # other summation indices
    if self.otherSummations:
      module.addComment1("global read addresses: other summation assignments")
      module.add(self.graOtherSummationAssignments(kernel))

    # tile offsets
    module.addComment1("global read addresses: tile offsets a")
    module.add(self.graTileOffsets(kernel, tensorParametersA))
    module.addComment1("global read addresses: tile offsets b")
    module.add(self.graTileOffsets(kernel, tensorParametersB))

    # unroll offsets
    module.addComment1("global read addresses: unroll offsets a")
    module.add(self.graUnrollOffsets(kernel, tensorParametersA))
    module.addComment1("global read addresses: unroll offsets b")
    module.add(self.graUnrollOffsets(kernel, tensorParametersB))

    # tile edges
    if kernel["EdgeType"] == "ShiftPtr":
      # Shift here has two purposes:
      #  1. Ensure the loads are in-bounds to prevent fault.
      #     BufferLoad uses the buffer limit hardware and does not require bounds checking for this case
      #  2. Shift-left a wide vector load to ensure it is completely in-bounds.
      #     If this occurs we need to 'unshift' the C values (see shiftVectorComponents)
      #     BufferLoad does support this shifting, but if GuaranteeNoPartial=1 then
      #     it can be guaranteed that no shifting is required.
      if not (kernel["BufferLoad"] and kernel["GuaranteeNoPartialA"]) and not forceNoTileCode:
        module.addComment1("global read addresses: shift a")
        module.add(self.graShift(kernel, tensorParametersA))
      if not (kernel["BufferLoad"] and  kernel["GuaranteeNoPartialB"]) and not forceNoTileCode:
        module.addComment1("global read addresses: shift b")
        module.add(self.graShift(kernel, tensorParametersB))

    # final offsets
    module.addComment1("global read addresses: final offsets a")
    module.add(self.graFinalOffsets(kernel, tensorParametersA))
    module.addComment1("global read addresses: final offsets b")
    module.add(self.graFinalOffsets(kernel, tensorParametersB))
    self.dontAppendCode = False
    self.dontAppendCode = self.dontAppendCode or forceNoTileCode

    # addresses
    if not forceNoTileCode:
      module.addComment1("global read addresses: addresses a")
      module.add(self.graAddresses(kernel, tensorParametersA))
      module.addComment1("global read addresses: addresses b")
      module.add(self.graAddresses(kernel, tensorParametersB))

    # increments
    module.addComment1("global read addresses: increments a")
    for i in reversed(range(kernel["ProblemType"]["NumIndicesSummation"])):
      module.add(self.graIncrements(kernel, i, tensorParametersA))
    module.addComment1("global read addresses: increments b")
    for i in reversed(range(kernel["ProblemType"]["NumIndicesSummation"])):
      module.add(self.graIncrements(kernel, i, tensorParametersB))

    ####################################
    # Local Write Addresses
    ####################################
    module.addComment2("Local Write Addresses")

    # tile assignments
    module.add(self.lwaTileAssignment(tensorParametersA))
    module.add(self.lwaTileAssignment(tensorParametersB))

    # unroll assignments
    module.add(self.lwaUnrollAssignment(kernel, tensorParametersA))
    module.add(self.lwaUnrollAssignment(kernel, tensorParametersB))

    # first offsets
    module.addComment1("local write addresses: first offset a")
    module.add(self.lwaFirstOffset(kernel, tensorParametersA))
    module.addComment1("local write addresses: first offset b")
    module.add(self.lwaFirstOffset(kernel, tensorParametersB))
    self.dontAppendCode = False
    self.dontAppendCode = self.dontAppendCode or forceNoTileCode

    ###########################################################################
    # summations loops: open
    ###########################################################################

    # declare loop num iter
    if not forceNoTileCode:
      module.addComment0("declare loop num iterations")

    # perform initC in the shadow of the prefetch
    # Prefetch occurs at start of unroll loop
    # If we have multiple summation indices (otherSummationLoops>0),
    # we can't init in shadow of this prefetch
    # since that would initC inside the other summation loops

    if self.doShadowInit != 2:
      module.add(self.initC(kernel))

    # open non-unrolled summation loops
    if not forceNoTileCode:
      for i in range(kernel["ProblemType"]["NumIndicesSummation"]-1):
        module.addComment1("summation loop %u"%i)
        module.add(self.calculateLoopNumIter(kernel, tensorParametersA, tensorParametersB, i))
        if self.actualSummationLoops>1:
          module.add(self.openLoop(kernel, tensorParametersA, tensorParametersB, i))
      module.add(self.calculateLoopNumIter(kernel, tensorParametersA, tensorParametersB, self.unrollIdx))

    if not forceNoTileCode:
      if self.staggerU:
        module.add(self.declareStaggerParms(kernel))
        module.add(self.calculateStagger(kernel, tensorParametersA))
        module.add(self.calculateStagger(kernel, tensorParametersB))

    # LRO and LWA as assigned
    # init lds read pointers before each unrolled loop
    module.addComment0("local read addresses: init pointers a")
    module.add(self.localReadInitPointers(kernel, tensorParametersA, tensorParametersA))
    module.addComment0("local read addresses: init pointers b")
    module.add(self.localReadInitPointers(kernel, tensorParametersA, tensorParametersB))

    ####################################
    # prefetch: unrolled loop prefix
    ####################################
    if kernel["PrefetchGlobalRead"]:
      pfi = 1
      module.addComment1("prefetch: global -> local")
      module.add(self.openSumAtLeastUnroll(kernel, prefetch=True, isOptNLL=isOptNLL))
      # if DirectToVgprA is enabled, swap the order of global read (B->A)
      tensorParameters1st = tensorParametersA
      tensorParameters2nd = tensorParametersB
      if kernel["DirectToVgprA"]:
        tensorParameters1st, tensorParameters2nd = tensorParameters2nd, tensorParameters1st
      moduleTmp = self.directToLdsM0Update(kernel, 0, tensorParameters1st, usePlaceHolder=False)
      module.add(replaceHolder(moduleTmp, 0))
      module.add(self.globalReadDo(kernel, 0, tensorParameters1st, 0))
      moduleTmp = self.directToLdsM0Update(kernel, 0, tensorParameters2nd, usePlaceHolder=False)
      module.add(replaceHolder(moduleTmp, 0))
      module.add(self.globalReadDo(kernel, 0, tensorParameters2nd, 0))
      module.add(self.globalReadIncrementAB(kernel, tensorParametersA, tensorParametersB, self.unrollIdx, pfi))

    module.addComment2("End setupNewTile")

    return module


  ##############################################################################
  # get conditions to skip local write wait
  ##############################################################################
  def getConditionToSkipLocalWriteWait( self, kernel, u):
    cond1 = not (u == 0 and kernel["PrefetchLocalRead"] != 0 and \
       (kernel["DirectToVgprA"] and kernel["DirectToLdsB"] or kernel["DirectToVgprB"] and kernel["DirectToLdsA"])) \
      or kernel["PrefetchGlobalRead"]==2
    # no need local read wait if LocalReadVectorWidth==2 and u is odd.
    # In that case, Prefetch local read covers both u = 0 and 1 (limit to MFMA+double+DirectToVgpr only)
    # (The other side of numReadsIterCoalesced must be 0 to skip local read wait)
    condSkip = kernel["LocalReadVectorWidth"]==2 and (u%2 != 0) and kernel["EnableMatrixInstruction"] and \
               kernel["ProblemType"]["DataType"].isDouble() and \
              (kernel["DirectToVgprA"] and self.numReadsIterCoalescedB % 2 == 0 or \
               kernel["DirectToVgprB"] and self.numReadsIterCoalescedA % 2 == 0)
    return cond1 and (not condSkip)

  ##############################################################################
  # No Load Loop Body
  ##############################################################################
  def noLoadLoopBody( self, kernel, tensorParametersA, tensorParametersB, pack, isOptNLL, isNGLL, NLLfirst, NLLlast, isDTVodd=False):
    module = Module("noLoadLoopBody")
    expand = kernel["ExpandPointerSwap"]
    lastuIdx = False
    pflr     = self.numItersPLR
    localWriteEndIter = kernel["LoopIters"] - self.numItersPLR - 1

    for uIdx in range(0, kernel["LoopIters"]*kernel["DepthULdsDivisor"]):
      u = uIdx % kernel["LoopIters"]    #   u: index in compute loop (in contrast to the notion of global read loop)
      uDu = uIdx // kernel["LoopIters"] # uDu: index of compute loop
      isLastLoop = (uDu == kernel["DepthULdsDivisor"] -1 ) and not isNGLL
      if u == 0:
        if uDu > 0:
          assert len(self.globalReadACode.items()) > 0 and len(self.globalReadBCode.items()) > 0 # already issued in first uDu
          self.globalReadACode = StructuredModule() # empty
          self.globalReadBCode = StructuredModule() # empty
          self.globalReadIncrements = Module() # empty
          self.globalReadIncrements.add(Module("globalReadIncrementA"))
          self.globalReadIncrements.add(Module("globalReadIncrementB"))
        if not isLastLoop:
          self.localWriteACode = self.localWriteDo(kernel, tensorParametersA, (uDu+1)%kernel["DepthULdsDivisor"])  # local write in loopcnt N targets data for loopcnt N+1
          self.localWriteBCode = self.localWriteDo(kernel, tensorParametersB, (uDu+1)%kernel["DepthULdsDivisor"])
        else:
          self.localWriteACode = Module()
          self.localWriteBCode = Module()

        # TODO schedule waitcnt/barrier in makeSubIterSchedule()
        if kernel["PrefetchGlobalRead"] and kernel["LoopIters"] in [1, 2] and uDu > 0:
          module.add(self.wait(kernel, tensorParametersA, tensorParametersB, 1, 0, -1, "wait for local write"))
          module.add(self.syncThreads(kernel, "sync for local read after write"))

        if not isNGLL:
          # PAP would have GlobalRead and GlobalInc, but no localWrite
          # Get the perIterGlobalReadCode code for PAP (if PAP=On), else would be empty
          self.makeSchedule(kernel, tensorParametersA, tensorParametersB, localWriteEndIter, uDu, skipGlobalReadInc=False, lastLoop=NLLlast)
          module.add(self.unrollLoopHeaderCode)

      # which loop iteration to reset the LRO,
      # note if PLR=0, isResetLroIter is False for all u
      isResetLroIter = (u == localWriteEndIter)
      isSwapAndResetLwoIter = isResetLroIter
      isSwapLroIter = isResetLroIter
      if kernel["ScheduleIterAlg"] == 3:
          isSwapAndResetLwoIter = (u == self.lwEndMfmaIndex//(self.numMfmaPerIter))

      extraComment = ""
      if isLastLoop:
        extraComment += " (last unrolled loop)"
      else:
        if kernel.enabledSplitLDS:
            extraComment += f" (uDu={uDu}) "
        if isResetLroIter:
            extraComment += " (reset local read pointers iteration) "
        if isSwapAndResetLwoIter:
            extraComment += " (swap and reset local write pointers iteration) "
        if isSwapLroIter:
            extraComment += " (swap local read pointers iteration) "

      module.addComment1("iter %u%s"%(u,extraComment))
      plrIdx = ((u+pflr) % (self.numVgprBuffer+1)) % kernel["LoopIters"]
      localReads = Module()

      pointerLWCode = Module()
      pointerLRCode = Module()
      waitCode = Module()  # may be overwritten (not added to) below
      macIterCode = Module()
      waitLWCode = Module()
      syncCode = Module()

      hasLiveLdsData = kernel["PrefetchGlobalRead"] or (uDu < kernel["DepthULdsDivisor"]-1)
      hasLiveLdsData = hasLiveLdsData and not isLastLoop
      # reads for current loop are done in previous iteration because of wider local read
      doReadA = (u < kernel["LoopIters"]/self.numIterPerCoalescedReadA - self.numItersPLR)
      doReadB = (u < kernel["LoopIters"]/self.numIterPerCoalescedReadB - self.numItersPLR)
      # reads for next loop
      doReadA = doReadA or (hasLiveLdsData and u > localWriteEndIter)
      doReadB = doReadB or (hasLiveLdsData and u > localWriteEndIter)
      # disable LocalRead if DirectToVgpr is enabled
      doReadA = doReadA and (not kernel["DirectToVgprA"])
      doReadB = doReadB and (not kernel["DirectToVgprB"])
      for iui in range(0,kernel["InnerUnroll"]):
        doReadA = doReadA and iui*self.numReadsIterCoalescedA < kernel["InnerUnroll"]
        doReadB = doReadB and iui*self.numReadsIterCoalescedB < kernel["InnerUnroll"]
        if doReadA:
          localReads.addComment1("local read a")
          localReadCodeA, packCodeA = self.localReadDo(kernel, plrIdx*self.numIterPerCoalescedReadA, iui*self.numReadsIterCoalescedA, 0, tensorParametersA)
          localReads.add(localReadCodeA)
          pack[plrIdx*self.numIterPerCoalescedReadA].add(packCodeA)
        if doReadB:
          localReads.addComment1("local read b")
          localReadCodeB, packCodeB = self.localReadDo(kernel, plrIdx*self.numIterPerCoalescedReadB, iui*self.numReadsIterCoalescedB, 0, tensorParametersB)
          localReads.add(localReadCodeB)
          pack[plrIdx*self.numIterPerCoalescedReadB].add(packCodeB)
        if (not isResetLroIter or iui != kernel["InnerUnroll"]-1):
          if doReadA:
            localReads.addComment1("local read increment a")
            localReads.add(self.localReadInc(kernel, iui, tensorParametersA))
          if doReadB:
            localReads.addComment1("local read increment b")
            localReads.add(self.localReadInc(kernel, iui, tensorParametersB))

      if not isLastLoop:
        if kernel["PrefetchGlobalRead"]:
          # put barrier at localWriteEndIter+1
          if u == localWriteEndIter+1 or (u == (localWriteEndIter+1)%kernel["LoopIters"] and kernel["ScheduleIterAlg"] == 2):
            # skip local write wait if DirectToVgpr + DirectToLds is enabled
            if not kernel["NoLdsWriteCode"]:
              waitLWCode.add(self.wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "3wait for local write"))
            if (kernel["DirectToVgprA"] or kernel["DirectToVgprB"]) and (kernel["DirectToLdsA"] or kernel["DirectToLdsB"]):
              # DirectToVgpr + DirectToLds case, add waitcnt vmcnt before s_barrier
              retInst = self.getWaitcntCodeForDirectToVgpr(kernel, localWriteEndIter, u, firstIter=False, beforeBarrier=True)
              waitLWCode.add(retInst)
            syncCode.add(self.syncThreads(kernel))

          if isSwapAndResetLwoIter: # ResetLroIter
            # local write for next iter, used to have local writes here
            pointerLWCode.addComment1("local write swap offsets a")
            pointerLWCode.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersA))
            pointerLWCode.addComment1("local write swap offsets b")
            pointerLWCode.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersB))

          if isSwapLroIter: # ResetLroIter
            # Swap, reset, or increment the LRO:
            # force internalPointerSwap = False in NGLL case
            internalPointerSwap = expand and not isNGLL
            pointerLRCode.addComment1("local read swap offsets a")
            pointerLRCode.add(self.localReadSwapOffsets(kernel, internalPointerSwap, tensorParametersA))
            pointerLRCode.addComment1("local read swap offsets b")
            pointerLRCode.add(self.localReadSwapOffsets(kernel, internalPointerSwap, tensorParametersB))

        if isResetLroIter: # ResetLroIter
          pointerLRCode.addComment1("local read init pointers a")
          pointerLRCode.add(self.localReadInitPointers(kernel, tensorParametersA, tensorParametersA))
          pointerLRCode.addComment1("local read init pointers b")
          pointerLRCode.add(self.localReadInitPointers(kernel, tensorParametersA, tensorParametersB))

      # we initiate lgkmcnt to 0, then assigning it correct value in makeSubIterSchedule()
      waitCode = self.wait(kernel, tensorParametersA, tensorParametersB, \
          -1, 0, 0, \
          "wait for prior local read local write")
      # DirectToVgpr case, wait for global read as well as local read/write
      if kernel["DirectToVgprA"] or kernel["DirectToVgprB"]:
        # not generate wait here
        #  1) local write code in previous u (u-1) has waitcnt vmcnt
        prevVmcnt = False
        prevLocalWrite = ""
        if (u > 0):
          prevLocalWrite = ' '.join([str(x) for x in self.perIterLocalWriteCode[u-1].flatitems()])
          prevVmcnt = "vmcnt" in prevLocalWrite
        if not prevVmcnt:
          retInst = self.getWaitcntCodeForDirectToVgpr(kernel, localWriteEndIter, u, False, isNGLL, NLLlast=NLLlast)
          module.add(retInst)

      luIdx = (u) % (self.numVgprBuffer+1) # local to use for MACs
      if kernel["EnableMatrixInstruction"]:
        # NGLL case, use first set
        setId = 0 if isNGLL else 1
        # flip setId if isDTVodd is True
        if isDTVodd:
           setId = 1 - setId
        # use second set for DirectToVGPR
        vregSetIdxMFMA = setId # use first set for NGLL, second set for other cases
        macIterCode.add(self.mfmaIter(kernel, tensorParametersA, tensorParametersB, u, kernel["InnerUnroll"], vregSetIdxMFMA))
      else:
        printExit("TensileLite does not support MAC instructions.")

      subIterCode = self.makeSubIterSchedule(kernel, tensorParametersA, tensorParametersB, localReads, \
                      u, pointerLWCode, pointerLRCode, waitCode, macIterCode, waitLWCode, syncCode, pack[luIdx], isDTVodd, NLLlast)
      module.add(subIterCode)
      # vgpr.checkin for all the checked-out vgpr in LocalRead
      for item in list(pack[luIdx].items()):
        if item.tempVgpr != None:
          self.vgprPool.checkIn(item.tempVgpr)
          item.tempVgpr = None
      pack[luIdx] = Module()
    return module

  ##############################################################################
  # noLoadLoop
  # Create the no load loop (NLL)
  #
  # isOptNLL : the NLL is to be optimized for the alpha=1 and non-edge case
  ##############################################################################
  def noLoadLoop( self, kernel, tensorParametersA, tensorParametersB, isOptNLL, isNGLL, pack ):
    module = Module("noLoadLoop")
    LoopNameComment = "NoGlobalLoadLoop" if isNGLL else "NoLoadLoop"
    isOptNLLComment = "Opt" if isOptNLL else "Ord"
    startComment = "%s. %s - Begin " % (isOptNLLComment, LoopNameComment)
    module.addComment2(startComment)
    NLLfirst = True
    NLLlast = True
    if kernel["PrefetchGlobalRead"] == 2:
      # PGR=2 case NoLoadLoop(NLL) is generated twice
      # we need to distinguish them to generate proper code at each NLL
      if isNGLL:
        NLLlast = False
      else:
        # PGR=2 and not isNGLL means second NoLoadLoop for PGR2.
        # Need to avoid generating duplicated code which is already generated in NGLL(first NoLoadLoop for PGR=2)
        NLLfirst = False
    if isNGLL:
      self.perIterLocalWriteCode = self.perIterLocalWriteCodeNGLL
      self.perIterLocalWriteCanSkip = [ 0 for i in range (kernel["LoopIters"]) ]
    #else:
    if not isNGLL:
      self.dtlsM0UpdateACode = StructuredModule()
      self.globalReadACode = StructuredModule() # empty
      self.dtlsM0UpdateBCode = StructuredModule()
      self.globalReadBCode = StructuredModule() # empty
      self.globalReadIncrements = Module()
      self.globalReadIncrements.add(Module("globalReadIncrementA"))
      self.globalReadIncrements.add(Module("globalReadIncrementB"))
      self.localWriteACode = Module()
      self.localWriteBCode = Module()

    kStrOpenSum = self.openSumAtLeastUnroll(kernel, prefetch=False, isOptNLL=isOptNLL)

    # skip generating OpenSum code here for SingleNLLOpt
    if not (isOptNLL and self.enableSingleNLLOpt):
      module.add(kStrOpenSum)
      kStrOpenSum = "" # empty OpenSum str to avoid inserting it again

    if not self.numItersPLR:
      if kernel["DirectToLdsA"] or kernel["DirectToLdsB"]:
        module.add(self.wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, "10wait for global read"))
      # TODO: need to check if we correctly checked-in the temp VGPR used for Int8 LocalWrite (uDu, PGR=2)
      module.add(self.wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "4wait for local write"))
      module.add(self.syncThreads(kernel))

    # if DirectToVgpr and  ASEM is not multiple of DepthU*2, generate noLoadLoopBody twice for odd and even exit separately
    if ( kernel["DirectToVgprA"] or  kernel["DirectToVgprB"]) and (kernel["AssertSummationElementMultiple"] % (kernel["DepthU"] * 2) != 0):
      # generate additional No Load Loop Body code for odd case (to use the other Vreg set for DirectToVgpr)
      # 1. generate odd check
      name = ""
      if isNGLL:
        name += "NoGlobalLoadLoop"
      else:
        name += "NoLoadLoop"
      if isOptNLL:
        name += "Opt"
      else:
        name += "Ord"
      module.add(self.openOddNoLoadLoopForDTV(name))
      # 2. generate  no Load Loop Body code for odd
      # backup
      self.saveLocalPointers(kernel, tensorParametersA, tensorParametersB)
      # deepCopy packCode for OptNLL noLoadLoop
      deepCopyPack = copy.deepcopy(pack)
      module.add(self.noLoadLoopBody(kernel, tensorParametersA, tensorParametersB, deepCopyPack, isOptNLL, isNGLL, NLLfirst, NLLlast, isDTVodd=True))
      # restore
      self.restoreLocalPointers(kernel, tensorParametersA, tensorParametersB)
      # 3. generate even start label
      module.add(self.closeOddNoLoadLoopForDTV(name))
      # 4. generate  no Load Loop Body code for odd
      # need to re-initialize perIterLocalWriteCanSkip to avoid having incorrect lgkmcnt
      self.perIterLocalWriteCanSkip = [ 0 for i in range (kernel["LoopIters"]) ]
      module.add(self.noLoadLoopBody(kernel, tensorParametersA, tensorParametersB, pack, isOptNLL, isNGLL, NLLfirst, NLLlast))
      # 5. generate even end label
      module.add(self.generateEvenEndLabeNoLoadLoopForDTV(name))
    else:
      # generate no Load Loop Body code
      module.add(self.noLoadLoopBody(kernel, tensorParametersA, tensorParametersB, pack, isOptNLL, isNGLL, NLLfirst, NLLlast))

    # add OpenSum code here if it is not empty
    if kStrOpenSum != "":
      module.add(kStrOpenSum)

    # Close code is necessary for both first and last (NGLL case(=NLLfirst) needs label)
    module.add(self.closeSumAtLeastUnroll(kernel, tensorParametersA, tensorParametersB, prefetch=False, isOptNLL=isOptNLL, isNGLL=isNGLL))

    return module

  ##############################################################################
  # Loop Body
  ##############################################################################
  def loopBody( self, kernel, tensorParametersA, tensorParametersB, pack, lc, loopCopies, finalLoop, firstIter=False ):
    module = Module("loopBody")
    expand = kernel["ExpandPointerSwap"]

    # not generate openLoop for firstIter
    if not firstIter:
      module.addComment2("Unrolled Loop %u/%u - Begin" % (lc+1, loopCopies))
    if kernel["PrefetchGlobalRead"] and not self.numItersPLR and not kernel["ScheduleIterAlg"] == 2:
      if kernel["DirectToLdsA"] or kernel["DirectToLdsB"]:
        module.add(self.wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, "11wait for global read"))
      module.add(self.wait(kernel, tensorParametersA, tensorParametersB, 1, 0, -1, "1wait for local write"))
      module.add(self.syncThreads(kernel, "4sync for global read"))

    module.addComment1("Begin Each Unroll: Check VGPR.checkin for INT8 LW")

    # if DirectToVgprA is enabled, swap the order of global read (B->A)
    tensorParameters1st = tensorParametersA
    tensorParameters2nd = tensorParametersB
    tc1 = 'A'
    tc2 = 'B'
    if kernel["DirectToVgprA"]:
      tensorParameters1st, tensorParameters2nd = tensorParameters2nd, tensorParameters1st
      tc1, tc2 = tc2, tc1
    # unrolled loop: global read A, B
    # M0 update for directToLds
    vregSetIdxGR = 0
    if (kernel["DirectToVgpr%s"%tc1]):
      vregSetIdxGR = (kernel["PrefetchGlobalRead"] + lc ) % 2 # toggle vreg set for DirectToVgpr.
    self.dtlsM0UpdateACode = self.directToLdsM0Update(kernel, 1, tensorParameters1st, usePlaceHolder=True)
    self.globalReadACode  = self.globalReadDo(kernel, 1, tensorParameters1st, vregSetIdxGR)
    vregSetIdxGR = 0
    if (kernel["DirectToVgpr%s"%tc2]):
      vregSetIdxGR = (kernel["PrefetchGlobalRead"] + lc ) % 2 # toggle vreg set for DirectToVgpr.
    self.dtlsM0UpdateBCode = self.directToLdsM0Update(kernel, 1, tensorParameters2nd, usePlaceHolder=True)
    self.globalReadBCode = self.globalReadDo(kernel, 1, tensorParameters2nd, vregSetIdxGR)

    # unrolled loop: increment global read addresses
    self.globalReadIncrements = self.globalReadIncrementAB(kernel, tensorParametersA, tensorParametersB, self.unrollIdx, 0)

    if not kernel["NoLdsWriteCode"]:
      self.localWriteACode = self.localWriteDo(kernel, tensorParametersA)
      self.localWriteBCode = self.localWriteDo(kernel, tensorParametersB)
    else:
      self.localWriteACode = Module()
      self.localWriteBCode = Module()

    # localWriteEndIter is used to determine which iteration to put sync
    # if PGR=0, GR,LW,sync,LR will put at front of loop.
    localWriteEndIter = kernel["LoopIters"] - self.numItersPLR - 1

    # Schedule the global read, global read inc, and writes:
    unrollLoopHeaderCodeScheduled = False
    if not kernel["PrefetchGlobalRead"]:
      unrollLoopHeaderCodeScheduled = True
      self.makeSchedule(kernel, tensorParametersA, tensorParametersB, localWriteEndIter, firstIter=firstIter)
      module.add(self.unrollLoopHeaderCode)

    # if not prefetch global, localWrite before mac's
    if not kernel["PrefetchGlobalRead"]:
      # unrolled loop: local write A, B
      module.add(self.wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, "5wait for global read"))
      module.add(self.syncThreads(kernel, "PGR=0, prior iter done reading lds"))
      if not kernel["NoLdsWriteCode"]:
        module.addComment1("local write a")
        tempLWCodeModA = self.localWriteDo(kernel, tensorParametersA)
        module.add(tempLWCodeModA)
        module.addComment1("local write b")
        tempLWCodeModB = self.localWriteDo(kernel, tensorParametersB)
        module.add(tempLWCodeModB)
      module.add(self.wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "2prefetch wait for local write"))
      module.add(self.syncThreads(kernel))
      # debug Local state
      """
      module.add("    /* print Local state */" + self.endLine)
      module.add("    for (unsigned int i = serial; i < LDS_NUM_ELEMENTS; i+=NUM_THREADS) {%s" % self.endLine)
      module.add("      printf(\\\"localMemory[%%06u] = %%.0f\\\\n\\\", i, localMemory[i]);%s" )
          % self.endLine
      module.add("    }" + self.endLine)
      """

    # unrolled loop: prefetch local
    if self.numItersPLR and not kernel["PrefetchGlobalRead"]:
      for plrIdx in range(0, self.numItersPLR):
        pack[plrIdx] = Module()
        for iui in range(0,kernel["InnerUnroll"]):
          if iui*self.numReadsIterCoalescedA < kernel["InnerUnroll"] and (not kernel["DirectToVgprA"]) : # no local read code if DirectToVgpr is enabled
            module.addComment1("prefetch local a")
            localReadCodeA, packCodeA = self.localReadDo(kernel, plrIdx*self.numIterPerCoalescedReadA, iui*self.numReadsIterCoalescedA, 0, tensorParametersA)
            module.add(localReadCodeA)
            pack[plrIdx].add(packCodeA)
          if iui*self.numReadsIterCoalescedB < kernel["InnerUnroll"] and (not kernel["DirectToVgprB"]) : # no local read code if DirectToVgpr is enabled
            module.addComment1("prefetch local b")
            localReadCodeB, packCodeB = self.localReadDo(kernel, plrIdx*self.numIterPerCoalescedReadB, iui*self.numReadsIterCoalescedB, 0, tensorParametersB)
            module.add(localReadCodeB)
            pack[plrIdx].add(packCodeB)
          if iui*self.numReadsIterCoalescedA < kernel["InnerUnroll"] and (not kernel["DirectToVgprA"]) : # no local read code if DirectToVgpr is enabled
            module.addComment0("local read increment a")
            module.add(self.localReadInc(kernel, iui, tensorParametersA))
          if iui*self.numReadsIterCoalescedB < kernel["InnerUnroll"]  and (not kernel["DirectToVgprB"]) : # no local read code if DirectToVgpr is enabled
            module.addComment0("local read increment b")
            module.add(self.localReadInc(kernel, iui, tensorParametersB))

    pflr     = self.numItersPLR  # how many pf already done above

    ############################################################################
    # unrolled loop: mac iterations
    ############################################################################

    # double/quadruple the number of compute loop for each DepthU's worth of data read
    for uIdx in range(0, kernel["LoopIters"]*kernel["DepthULdsDivisor"]):
      u = uIdx % kernel["LoopIters"]    #   u: index in compute loop (in contrast to the notion of global read loop)
      uDu = uIdx // kernel["LoopIters"] # uDu: index of compute loop
      if u==0: # if at start of subloop...
        # ...update local write code
        if not kernel["NoLdsWriteCode"]:
          self.localWriteACode = self.localWriteDo(kernel, tensorParametersA, (uDu+1)%kernel["DepthULdsDivisor"])  # local write in loopcnt N targets data for loopcnt N+1
          self.localWriteBCode = self.localWriteDo(kernel, tensorParametersB, (uDu+1)%kernel["DepthULdsDivisor"])
        else:
          self.localWriteACode = Module()
          self.localWriteBCode = Module()

        # TODO schedule waitcnt/barrier in makeSubIterSchedule()
        if kernel["PrefetchGlobalRead"] and kernel["LoopIters"] in [1, 2] and uDu > 0:
          module.add(self.wait(kernel, tensorParametersA, tensorParametersB, 1, 0, -1, "wait for local write"))
          module.add(self.syncThreads(kernel, "sync for local read after write"))

        if not unrollLoopHeaderCodeScheduled:
          self.makeSchedule(kernel, tensorParametersA, tensorParametersB, localWriteEndIter, uDu, firstIter=firstIter, lastLoop=False, lastLc=(lc==loopCopies-1))
          module.add(self.unrollLoopHeaderCode)

      # for PGR=0 where generator can't schedule the instructions (yet),
      # we duplicate the local write codegen and append to string list directly
      if not kernel["PrefetchGlobalRead"]:
        doWrite = False
        if uDu<kernel["DepthULdsDivisor"]-1 and u==kernel["LoopIters"]-self.numItersPLR:
          doWrite = True
          writeForNextLoop = 1
        if uDu>0 and self.numItersPLR==0 and u==0:
          assert doWrite==False # should be exclusive with the previous condition
          doWrite = True
          writeForNextLoop = 0
        # unrolled loop: local write A, B
        if doWrite:
          module.add(self.wait(kernel, tensorParametersA, tensorParametersB, -1, -1, 0, "5wait for local read"))
          module.add(self.syncThreads(kernel, "PGR=0, prior iter done reading lds"))
          if not kernel["NoLdsWriteCode"]:
            module.addComment1("local write a")
            tempLWCodeModA = self.localWriteDo(kernel, tensorParametersA, (uDu+writeForNextLoop)%kernel["DepthULdsDivisor"])
            module.add(tempLWCodeModA)
            module.addComment1("local write b")
            tempLWCodeModB = self.localWriteDo(kernel, tensorParametersB, (uDu+writeForNextLoop)%kernel["DepthULdsDivisor"])
            module.add(tempLWCodeModB)
          module.add(self.wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "2prefetch wait for local write"))
          module.add(self.syncThreads(kernel))

      # which loop iteration to reset the LRO,
      # note if PLR=0, isResetLroIter is False for all u
      isResetLroIter = (u == localWriteEndIter)
      isSwapAndResetLwoIter = isResetLroIter
      isSwapLroIter = isResetLroIter
      if kernel["ScheduleIterAlg"] == 3:
        isSwapAndResetLwoIter = (u == self.lwEndMfmaIndex//(self.numMfmaPerIter))
      extraComment = ""
      if kernel.enabledSplitLDS:
        extraComment += f" (uDu={uDu}) "
      if isResetLroIter:
        extraComment += " (reset local read pointers iteration) "
      if isSwapAndResetLwoIter:
        extraComment += " (swap and reset local write pointers iteration) "
      if isSwapLroIter:
        extraComment += " (swap local read pointers iteration) "

      module.addComment1("iter %u%s"%(u,extraComment))
      plrIdx = ((u+pflr) % (self.numVgprBuffer+1)) % kernel["LoopIters"]

      localReads = Module()
      localReadsA = Module()
      localReadsB = Module()

      pointerLWCode = Module()
      pointerLRCode = Module()
      waitCode = Module()  # may be overwritten (not added to) below
      macIterCode = Module()
      waitLWCode = Module()
      syncCode = Module()

      hasLiveLdsData = kernel["PrefetchGlobalRead"] or (uDu < kernel["DepthULdsDivisor"]-1)
      # reads for current loop are done in previous iteration because of wider local read
      doReadA = (u < kernel["LoopIters"]/self.numIterPerCoalescedReadA - self.numItersPLR)
      doReadB = (u < kernel["LoopIters"]/self.numIterPerCoalescedReadB - self.numItersPLR)
      # reads for next loop
      doReadA = doReadA or (hasLiveLdsData and u > localWriteEndIter)
      doReadB = doReadB or (hasLiveLdsData and u > localWriteEndIter)
      # disable LocalRead if DirectToVgpr is enabled
      doReadA = doReadA and (not kernel["DirectToVgprA"])
      doReadB = doReadB and (not kernel["DirectToVgprB"])
      # double the number of VgprValu if self.vgprValuDouble is true
      plrIdxLR = plrIdx
      if self.vgprValuDouble and (lc & 1) == 0:
        # use the next buffer set (do not change the index of pack[])
        plrIdxLR += 1
      for iui in range(0,kernel["InnerUnroll"]):
        doReadA = doReadA and iui*self.numReadsIterCoalescedA < kernel["InnerUnroll"]
        doReadB = doReadB and iui*self.numReadsIterCoalescedB < kernel["InnerUnroll"]
        if doReadA:
          localReads.addComment1("local read a")
          localReadCodeA, packCodeA = self.localReadDo(kernel, plrIdxLR*self.numIterPerCoalescedReadA, iui*self.numReadsIterCoalescedA, 0, tensorParametersA)
          localReads.add(localReadCodeA)
          localReadsA.add(localReadCodeA)
          pack[plrIdx*self.numIterPerCoalescedReadA].add(packCodeA)
        if doReadB:
          localReads.addComment1("local read b")
          localReadCodeB, packCodeB = self.localReadDo(kernel, plrIdxLR*self.numIterPerCoalescedReadB, iui*self.numReadsIterCoalescedB, 0, tensorParametersB)
          localReads.add(localReadCodeB)
          localReadsB.add(localReadCodeB)
          pack[plrIdx*self.numIterPerCoalescedReadB].add(packCodeB)
        # Don't increment the LRO if we are going to reset them below:
        if not isResetLroIter or iui != kernel["InnerUnroll"]-1:
          if doReadA:
            localReads.addComment1("local read increment a")
            localReads.add(self.localReadInc(kernel, iui, tensorParametersA))
          if doReadB:
            localReads.addComment1("local read increment b")
            localReads.add(self.localReadInc(kernel, iui, tensorParametersB))

      if kernel["PrefetchGlobalRead"]:
        # wait code for DirectToVgpr
        if kernel["DirectToVgprA"] or kernel["DirectToVgprB"]:
          # not generate wait here
          #  1) local write code in previous u (u-1) has waitcnt vmcnt
          prevVmcnt = False
          prevLocalWrite = ""
          if (u > 0):
            for up in range(u):
              prevLocalWrite += ' '.join([str(x) for x in self.perIterLocalWriteCode[up].flatitems()])
            prevVmcnt = "vmcnt" in prevLocalWrite
          if not prevVmcnt:
            retInst = self.getWaitcntCodeForDirectToVgpr(kernel, localWriteEndIter, u, firstIter)
            module.add(retInst)
        # put barrier at localWriteEndIter+1
        if u == localWriteEndIter+1 or (u == (localWriteEndIter+1)%kernel["LoopIters"] and kernel["ScheduleIterAlg"] == 2):
          if kernel["DirectToLdsA"] or kernel["DirectToLdsB"]:
            # skip generating wait for global read again here in DirectToVgpr case
            if not(kernel["DirectToVgprA"] or kernel["DirectToVgprB"]):
              module.add(self.wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, "12wait for global read"))
            else:
              # DirectToVgpr + DirectToLds case, add waitcnt vmcnt before s_barrier
              retInst = self.getWaitcntCodeForDirectToVgpr(kernel, localWriteEndIter, u, firstIter, beforeBarrier=True)
              waitLWCode.add(retInst)
          # skip local write wait if DirectToVgpr + DirectToLds is enabled
          # (no local write code. Global read wait for DirectToLds is already done)
          if not kernel["NoLdsWriteCode"]:
            waitLWCode.add(self.wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "3wait for local write"))
          if kernel["DirectToVgprA"] or kernel["DirectToVgprB"]:
            # put only barrier for DirectToVgpr (to avoid generating waitcnt for global read)
            syncCode.add("s_barrier" + self.endLine)
          else:
            syncCode.add(self.syncThreads(kernel))

        if isSwapAndResetLwoIter: # ResetLroIter
          # local write for next iter, used to have local writes here
          pointerLWCode.addComment1("local write swap offsets a")
          pointerLWCode.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersA))
          pointerLWCode.addComment1("local write swap offsets b")
          pointerLWCode.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersB))

        if isSwapLroIter: # ResetLroIter
          # Swap, reset, or increment the LRO:
          pointerLRCode.addComment1("local read swap offsets a")
          pointerLRCode.add(self.localReadSwapOffsets(kernel, expand, tensorParametersA))
          pointerLRCode.addComment1("local read swap offsets b")
          pointerLRCode.add(self.localReadSwapOffsets(kernel, expand, tensorParametersB))

      if isResetLroIter: # ResetLroIter
        pointerLRCode.addComment1("local read init pointers a")
        pointerLRCode.add(self.localReadInitPointers(kernel, tensorParametersA, tensorParametersA))
        pointerLRCode.addComment1("local read init pointers b")
        pointerLRCode.add(self.localReadInitPointers(kernel, tensorParametersA, tensorParametersB))

      # we initiate lgkmcnt to 0, then assigning it correct value in makeSubIterSchedule()
      if self.getConditionToSkipLocalWriteWait(kernel, u):
        waitCode = self.wait(kernel, tensorParametersA, tensorParametersB, \
            -1, 0, 0, \
            "wait for prior local read local write")

      luIdx = (u) % (self.numVgprBuffer+1) # local to use for MACs
      if kernel["EnableMatrixInstruction"]:
        vregSetIdxMFMA = lc
        macIterCode.add(self.mfmaIter(kernel, tensorParametersA, tensorParametersB, u, kernel["InnerUnroll"], vregSetIdxMFMA))
      else:
        printExit("TensileLite does not support MAC instructions.")

      ###### unroll loop efficiency implementation######################################
      # unroll loop efficiency implementation
      ## split A&B fetch&MAC code into multiple groups
      ## splitting strategy   based on TT size
      ## 6x4 -> split  MAC blob(s) into group of 8(s) and 16 FMA instructions.
      ##        LDS fetch(es) into group of A{1-2)B(0) , A(3),B(1) (not implemented yet)
      ## 4x6 -> split  MAC blob(s) into group of 8(s) and 16 FMA instructions.
      ##        LDS fetch(es) into group of B{1-2)A(0) , B(3),A(1)
      ## 4x4 -> split into group of 8 and 8  MAC(s)
      ## 6x6 -> split into group of 12 MAC(s)
      ## 8x4/4x8 -> split into group of 16 and 16  MAC(s)
      ## 8x8 -> split into group of 16 MAC(s)
      ## supports only PLR=0
      ###############################################################################
      if self.numItersPLR or (not globalParameters["UnrollLoopEfficiencyEnable"]):
        subIterCode = self.makeSubIterSchedule(kernel, tensorParametersA, tensorParametersB, localReads, \
                        u, pointerLWCode, pointerLRCode, waitCode, macIterCode, waitLWCode, syncCode, pack[luIdx])
        module.add(subIterCode) # add scheduled "other", local reads, local writes
        for item in list(pack[luIdx].items()):
          if item.tempVgpr != None:
            self.vgprPool.checkIn(item.tempVgpr)
            item.tempVgpr = None
        pack[luIdx] = Module()
      else:
        printExit("TensileLite does not support MAC instructions.")

    # close unrolled loop
    if expand:
      if not finalLoop:
        module.addComment2("Unrolled Loop - End %u/%u"%(lc+1, loopCopies))
      else:
        module.addComment2("Unrolled Loop - End %u/%u (final)"%(lc+1, loopCopies))

    else:
      module.addComment2("Unrolled Loop - End")

    oddLabel = lc == 0
    module.add(self.closeLoop(kernel, tensorParametersA, tensorParametersB, self.unrollIdx, finalLoop, oddLabel=oddLabel))
    return module

  ##############################################################################
  # Kernel Body
  ##############################################################################
  def kernelBody( self, kernel, tensorParametersA, tensorParametersB ):
    expand = kernel["ExpandPointerSwap"]

    ####################################
    # Begin String
    moduleKernelBody = KernelBody("kernelBody")

    ####################################
    # Function Signature
    ####################################
    fs = self.functionSignature()
    moduleKernelBody.addSignature(fs)

    module = Module("body")
    module.add(self.defineAndResources(kernel, tensorParametersA, tensorParametersB))

    ####################################
    # Local Read Addresses
    ####################################
    module.addComment2("Local Read Addresses")

    # tile assignments
    module.addComment1("local read addresses: tile assignments a/b")
    module.add(self.lraTileAssignment(kernel, tensorParametersA, tensorParametersB))

    # final offsets
    module.addComment1("local read addresses: final offsets a")
    module.add(self.lraFinalOffset(kernel, tensorParametersA))
    module.addComment1("local read addresses: final offsets b")
    module.add(self.lraFinalOffset(kernel, tensorParametersB))

    # declare addresses
    module.addComment1("local read addresses: declare addresses a")
    module.add(self.lraDeclareAddresses(kernel, tensorParametersA))
    module.addComment1("local read addresses: declare addresses b")
    module.add(self.lraDeclareAddresses(kernel, tensorParametersB))

    # doShadowInit performs initialization in the 'shadow' of the global mem prefetch
    self.doShadowInit = 0
    if kernel["PrefetchGlobalRead"]:
      if self.actualSummationLoops == 1:
        self.doShadowInit = 2 # 2 is both store setup and initC
      else:
        # can't do shadow initC with multiple summation since this resets the ValuC counters
        # on each unroll iteration.
        self.doShadowInit = 1 # 1 is just store setup

    module.add(self.setupNewTile(kernel, tensorParametersA, tensorParametersB, isOptNLL=False))

    pack = [ Module() for i in range (self.numVgprBuffer+1) ]
    self.preLoopLocalWriteCode = None

    if kernel["PrefetchGlobalRead"]:
      if self.doShadowInit:
        module.add(self.openShadowInit())
        module.add(self.globalWriteWorkGroupInit(kernel))
        if self.doShadowInit == 2:
          module.add(self.initC(kernel)) # initC while waiting for global reads
        module.add(self.closeShadowInit(kernel))

      module.add(self.wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, "8wait for global read"))
      # These cases loop back and run the prefetch loop again
      # we need an extra barrier to ensure that the ds_reads (either for SR or MFMA) from previous iteration
      # have finished before we generate the prefetch for the next summation index.
      if self.actualSummationLoops>1:
        module.add(SBarrier())

      # local write
      self.preLoopLocalWriteCode = self.preLoopLocalWriteDo(kernel, tensorParametersA, tensorParametersB)
      module.add(self.preLoopLocalWriteCode)
      # swap local ptrs
      module.addComment1("local write swap a")
      module.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersA))
      module.addComment1("local write swap b")
      module.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersB))

      if kernel["PrefetchGlobalRead"] == 2:
        module.add(self.openPrefetchGlobalRead2(kernel))
        # if DirectToVgprA is enabled, swap the order of global read (B->A)
        tensorParameters1st = tensorParametersA
        tensorParameters2nd = tensorParametersB
        if kernel["DirectToVgprA"]:
          tensorParameters1st, tensorParameters2nd = tensorParameters2nd, tensorParameters1st
        module.add(self.directToLdsM0Update(kernel, 1, tensorParameters1st))
        module.add(self.globalReadDo(kernel, 0, tensorParameters1st, 1))
        module.add(self.directToLdsM0Update(kernel, 1, tensorParameters2nd))
        module.add(self.globalReadDo(kernel, 0, tensorParameters2nd, 1))

        # swap local ptrs again if DirectToLds is enabled
        if kernel["DirectToLdsA"]:
          module.addComment1("local write swap a")
          module.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersA))
        if kernel["DirectToLdsB"]:
          module.addComment1("local write swap b")
          module.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersB))

        module.add(self.closePrefetchGlobalRead2())

      # prefetch-local
      if self.numItersPLR:
        # not generate wait for local write if LDS write code is not generated
        if not kernel["NoLdsWriteCode"]:
          # TODO: need to check if we correctly checked-in the temp VGPR used for Int8 LocalWrite (uDu, PGR=2)
          module.add(self.wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "0prefetch wait for local write"))
        module.add(self.syncThreads(kernel))

        # in some cases need an extra copy of the LDS read with appropriate double buffer offsets
        for plrIdx in range(0, self.numItersPLR):
          pack[plrIdx] = Module()
          for espi in range(0, 1):
            for iui in range(0,kernel["InnerUnroll"]):
              if iui*self.numReadsIterCoalescedA < kernel["InnerUnroll"] and (not kernel["DirectToVgprA"]) : # no local read code if DirectToVgpr is enabled
                module.addComment1("local read prefetch a")
                localReadCodeA, packCodeA = self.localReadDo(kernel, plrIdx*self.numIterPerCoalescedReadA, iui*self.numReadsIterCoalescedA, espi, tensorParametersA)
                module.add(localReadCodeA)
                pack[plrIdx].add(packCodeA)
              if iui*self.numReadsIterCoalescedB < kernel["InnerUnroll"] and (not kernel["DirectToVgprB"]) : # no local read code if DirectToVgpr is enabled
                module.addComment1("local read prefetch b")
                localReadCodeB, packCodeB = self.localReadDo(kernel, plrIdx*self.numIterPerCoalescedReadB, iui*self.numReadsIterCoalescedB, espi, tensorParametersB)
                module.add(localReadCodeB)
                pack[plrIdx].add(packCodeB)
              if iui*self.numReadsIterCoalescedA < kernel["InnerUnroll"] and (not kernel["DirectToVgprA"]) : # no local read code if DirectToVgpr is enabled
                module.addComment1("local read inc a")
                module.add(self.localReadInc(kernel, iui, tensorParametersA))
              if iui*self.numReadsIterCoalescedB < kernel["InnerUnroll"] and (not kernel["DirectToVgprB"]) : # no local read code if DirectToVgpr is enabled
                module.addComment1("local read inc b")
                module.add(self.localReadInc(kernel, iui, tensorParametersB))
      module.add(self.closeSumAtLeastUnroll(kernel, tensorParametersA, tensorParametersB, prefetch=True, isOptNLL=False, isNGLL=False))

    loopCopies = 2 if expand else 1

    if self.useInitAccVgprOpt:
      # generate first iteration code for init accvgpr opt
      module.addComment2("First Unrolled Iter for InitAccVgprOpt - Begin")
      # open loop without Label
      module.add(self.openLoop(kernel, tensorParametersA, tensorParametersB, self.unrollIdx, noLabelGen=True))
      module.add(self.loopBody( kernel, tensorParametersA, tensorParametersB, pack, 0, loopCopies, False, firstIter=True ))

    # open unrolled summation loop
    module.addComment2("Unrolled Loop(s) - Begin")
    module.add(self.openLoop(kernel, tensorParametersA, tensorParametersB, self.unrollIdx, beginLabelOnly=False))

    lcStart = 0
    if self.useInitAccVgprOpt:
      lcStart = 1 if loopCopies == 2 else 0
    for lc in range(0, loopCopies):
      loopIndex = lcStart + lc
      if loopIndex >= loopCopies:
        loopIndex -= loopCopies
      # loop body code generation
      finalLoop = lc == loopCopies - 1
      module.add(self.loopBody( kernel, tensorParametersA, tensorParametersB, pack, loopIndex, loopCopies, finalLoop ))

    module.addComment1("Before NLL: Check VGPR.checkin for INT8 LW")

    # swap local write, read again before noLoadLoop if PrefetchGlobalRead and DirectToLds is enabled
    # In DirectToLds enabled case, local write address is necessary for prefetch global read (for m0).
    # However, even exit with DirectToLds will not pass with this code (limitation).
    # So far, this code is to make odd exit case (i.e. k is multiple of 2*depthU) pass for DirectToVgpr
    if not self.useInitAccVgprOpt and kernel["PrefetchGlobalRead"] and kernel["ExpandPointerSwap"]:
      # local write for next iter, used to have local writes here
      if(kernel["DirectToLdsA"]):
        module.addComment1("local write swap offsets a")
        module.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersA))
      if(kernel["DirectToLdsB"]):
        module.addComment1("local write swap offsets b")
        module.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersB))
    # swap local read point for self.useInitAccVgprOpt
    if self.useInitAccVgprOpt and kernel["ExpandPointerSwap"]:
      module.addComment1("local read swap offsets a")
      module.add(self.localReadSwapOffsets(kernel, expand, tensorParametersA))
      module.addComment1("local read swap offsets b")
      module.add(self.localReadSwapOffsets(kernel, expand, tensorParametersB))

    if kernel["PrefetchGlobalRead"] == 2:
      module.add(self.noLoadLoop(kernel, tensorParametersA, tensorParametersB, isOptNLL=False, isNGLL=True, pack=pack))

    # This "NoLoad" loop is a copy of the unroll loop but with global loads + LDS writes removed
    # doShadowInit is required since this pushes up the store SRD initialization before the NLL
    # OptNLL only allowed for single summation index  - for multiple summation we (currently)
    # execute the NLL inside each unroll iteration not just once at the end.
    if kernel["PrefetchGlobalRead"]:
      if not kernel["SuppressNoLoadLoop"]:

        firstNLLgenerated = False
        if kernel["KernelLanguage"] == "Assembly" and kernel["OptNoLoadLoop"] and \
           kernel["BufferLoad"] and kernel["BufferStore"] and self.doShadowInit and \
           kernel["LocalSplitU"]==1 and kernel["GlobalSplitU"] == 1 and \
           self.actualSummationLoops==1:

          firstNLLgenerated = True

          # two different noLoadLoops:
          # 1. OptNLL & PAP global-read interleaved (only for PAP=ON)
          # (2. OptNLL : No PAP global-read (For PAP=OFF, or PAP=ON but the last tile))
          #  -> this is unified with 1. global-read is invalidated at the last tile.
          # 3. OrdinaryNLL (Not Opt.)
          self.saveLocalPointers(kernel, tensorParametersA, tensorParametersB)
          # deepCopy packCode for OptNLL noLoadLoop
          deepCopyPack = copy.deepcopy(pack)
          module.add(self.noLoadLoop(kernel, tensorParametersA, tensorParametersB, isOptNLL=True, isNGLL=False, pack=deepCopyPack))
          self.restoreLocalPointers(kernel, tensorParametersA, tensorParametersB)

        # skip second NLL code if enableSingleNLLOpt
        if not (self.enableSingleNLLOpt and firstNLLgenerated):
          module.add(self.noLoadLoop(kernel, tensorParametersA, tensorParametersB, isOptNLL=False, isNGLL=False, pack=pack))
        else:
          # generate PrefetchGlobalLastIterEnd label
          module.add(self.closeSumAtLeastUnroll(kernel, tensorParametersA, tensorParametersB, prefetch=False, isOptNLL=False, isNGLL=False))

      # if PGR, last few iterations will have PLR,
      # and those PLR will not be used(register not checkIn) if without NoLoadLoop
      else:
        for i in range(self.numVgprBuffer):
          for item in list(pack[i].items()):
            if item.tempVgpr != None:
              self.vgprPool.checkIn(item.tempVgpr)
              item.tempVgpr = None

    if self.staggerU and self.actualSummationLoops>1:
      module.addComment1("remove stagger offsets")
      module.add(self.removeStagger(kernel, tensorParametersA))
      module.add(self.removeStagger(kernel, tensorParametersB))

    if not self.noTailLoop:
      ########################################
      # Tail Loop
      # which means tail loop not needed.
      ########################################
      self.inTailLoop = True
      module.addComment2("Tail Loop")

      # Update local write pointers in case the upcoming global reads are writing directly to LDS:
      if kernel["PrefetchGlobalRead"]:
        module.addComment1("local write reset offsets a")
        module.add(self.localWriteResetOffsets(kernel,  kernel["ExpandPointerSwap"], tensorParametersA))
        if kernel["ExpandPointerSwap"]:
          # reset local write offset in asm code as well
          module.add(self.localWriteResetOffsets(kernel, False, tensorParametersA))
        module.addComment1("local write reset offsets b")
        module.add(self.localWriteResetOffsets(kernel,  kernel["ExpandPointerSwap"], tensorParametersB))
        if kernel["ExpandPointerSwap"]:
          # reset local write offset in asm code as well
          module.add(self.localWriteResetOffsets(kernel, False, tensorParametersB))

      # tail: global read
      module.add(self.calculateLoopNumIter(kernel, tensorParametersA, tensorParametersB, -1))
      if self.staggerU and self.actualSummationLoops==1:
        module.addComment1("remove stagger offsets for tail loop")
        module.add(self.removeStagger(kernel, tensorParametersA))
        module.add(self.removeStagger(kernel, tensorParametersB))

      # if DirectToVgprA is enabled, swap the order of global read (B->A)
      tensorParameters1st = tensorParametersA
      tensorParameters2nd = tensorParametersB
      tc1 = 'a'
      tc2 = 'b'
      if kernel["DirectToVgprA"]:
        tensorParameters1st, tensorParameters2nd = tensorParameters2nd, tensorParameters1st
        tc1, tc2 = tc2, tc1
      module.addComment1("Update M0 for DTLDS")
      moduleTmp = self.directToLdsM0Update(kernel, 1, tensorParameters1st)
      module.add(replaceHolder(moduleTmp, 0))
      module.addComment1("global read %s"%tc1)
      vregSetIdx = 0
      module.add(self.globalReadDo(kernel, 2, tensorParameters1st, vregSetIdx))
      module.addComment1("Update M0 for DTLDS")
      moduleTmp = self.directToLdsM0Update(kernel, 1, tensorParameters2nd)
      module.add(replaceHolder(moduleTmp, 0))
      module.addComment1("global read %s"%tc2)
      vregSetIdx = 0
      module.add(self.globalReadDo(kernel, 2, tensorParameters2nd, vregSetIdx))
      module.add(self.wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, "2wait for global read"))
      module.add(self.syncThreads(kernel))

      # the following read/write addresses could be modified in recalcLocal(Read|Write)Addresses due to policy change
      self.oriLraA = None # back up original local read address vgpr
      self.oriLraB = None
      self.oriLwaA = None # back up original local write address vgpr
      self.oriLwaB = None
      for uDu in range(0, kernel["DepthULdsDivisor"]):
        if kernel.enabledSplitLDS:
          # change local write policy from interleave-K to fractional as tail loop
          # iterate LDS read address one unit of K at a time
          module.addComment1("Recalc local write offsets")
          module.add(self.recalcLocalWriteAddresses(kernel, tensorParametersA, uDu))
          module.add(self.recalcLocalWriteAddresses(kernel, tensorParametersB, uDu))
        if uDu > 0:
          module.addComment1("sync before local write")
          module.add(self.syncThreads(kernel))
        if not kernel["NoLdsWriteCode"]:
          # tail: local write
          module.addComment1("local write a")
          tempLWCodeModA = self.localWriteDo(kernel, tensorParametersA, None)
          module.add(tempLWCodeModA)
          module.addComment1("local write b")
          tempLWCodeModB = self.localWriteDo(kernel, tensorParametersB, None)
          module.add(tempLWCodeModB)
        # change local read policy from wider local read to one unit of K at a time
        module.addComment1("Recalc local read offsets")
        module.add(self.recalcLocalReadAddressesAB(kernel, tensorParametersA, tensorParametersB))
        # TODO: need to check if we correctly checked-in the temp VGPR used for Int8 LocalWrite (uDu, PGR=2)
        module.add(self.wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "5wait for local write"))
        module.add(self.syncThreads(kernel))
        #module.add(self.dumpLds(kernel, 0, 8))

        # tail: re-init local read addresses
        if kernel["PrefetchGlobalRead"]:
          module.addComment1("local read reset offsets a")
          module.add(self.localReadResetOffsets(kernel, tensorParametersA))
          module.addComment1("local read reset offsets b")
          module.add(self.localReadResetOffsets(kernel, tensorParametersB))
          module.addComment1("local read init pointers a")
          module.add(self.localReadInitPointers(kernel, tensorParametersA, tensorParametersA))
          module.addComment1("local read init pointers b")
          module.add(self.localReadInitPointers(kernel, tensorParametersA, tensorParametersB))
        # tail: macs
        module.addComment1("tail loop: macs")
        module.add(self.openLoop(kernel, tensorParametersA, tensorParametersB, -1, uDu if kernel.enabledSplitLDS else None))

        # Try to use InnerUnroll in the tail loop if allowed:
        KinInnerUnroll = kernel["InnerUnroll"]
        if kernel["EnableMatrixInstruction"]:
          KinInnerUnroll *= kernel["MatrixInstK"]

        tailLoopInnerUnroll = 1
        if (kernel["AssertSummationElementMultiple"] % KinInnerUnroll == 0):
          tailLoopInnerUnroll = kernel["InnerUnroll"]
        # need to unroll tail loop for the following cases
        mEnd = 1
        if kernel["DirectToVgprA"] or kernel["DirectToVgprB"]:
          mEnd = kernel["DepthU"]//KinInnerUnroll
        elif kernel["DirectToLds"] and kernel["EnableMatrixInstruction"] and kernel["InnerUnroll"] == 1 and\
             (kernel["GlobalLoadVectorWidthA"] * self.bpeAB > 4 or kernel["GlobalLoadVectorWidthB"] * self.bpeAB > 4) and \
             kernel["DepthU"] // kernel["MatrixInstK"] > 2:
          mEnd = kernel["DepthU"] // (kernel["MatrixInstK"] * 2)

        for mValue in range(mEnd):
          pack[0] = Module()
          for iui in range(0, tailLoopInnerUnroll):
            doReadA = not kernel["DirectToVgprA"]
            doReadB = not kernel["DirectToVgprB"]
            if doReadA:
              # Reading 16-bit data from LDS requires packing when ECC enabled
              module.addComment1("local read a")
              localReadCodeA, packCodeA = self.localReadDo(kernel, 0, iui, 0, tensorParametersA)
              module.add(localReadCodeA)
              pack[0].add(packCodeA)
            if doReadB:
              module.addComment1("local read b")
              localReadCodeB, packCodeB = self.localReadDo(kernel, 0, iui, 0, tensorParametersB)
              module.add(localReadCodeB)
              pack[0].add(packCodeB)
            # adjustment for DirectToLds case
            iuiParam = iui + tailLoopInnerUnroll * mValue
            if doReadA:
              module.addComment1("local read inc a")
              module.add(self.localReadInc(kernel, iuiParam, tensorParametersA))
            if doReadB:
              module.addComment1("local read inc b")
              module.add(self.localReadInc(kernel, iuiParam, tensorParametersB))
          module.add(self.wait(kernel, tensorParametersA, tensorParametersB, -1, -1, 0, "4wait for local read"))

          if kernel["EnableMatrixInstruction"]:
            module.add(pack[0])
            # vgpr.checkin for all the checked-out vgpr in LocalRead
            for item in list(pack[0].items()):
              if item.tempVgpr != None:
                self.vgprPool.checkIn(item.tempVgpr)
                item.tempVgpr = None
            pack[0] = Module()

          if kernel["EnableMatrixInstruction"]:
            # DirectToVgpr is not applicable for tail loop
            vregSetIdxMFMA = 0
            module.add(self.mfmaIter(kernel, tensorParametersA, tensorParametersB, 0, tailLoopInnerUnroll, vregSetIdxMFMA, True))
          else:
            printExit("TensileLite does not support MAC instructions.")

          finalLoop = mValue == mEnd - 1
          module.add(self.closeLoop(kernel, tensorParametersA, tensorParametersB, -1, finalLoop, uDu if kernel.enabledSplitLDS else None))
      # always emit the skip-tail-loop label
      module.add(self.closeLoop(kernel, tensorParametersA, tensorParametersB, -1, None, emitEndLabelOnly=True))
      # tail: close
      self.inTailLoop = False

    # extra summation loops: global increment and close
    for i in reversed(range(self.otherSummationLoops)):
      module.addComment1("global read inc AB")
      module.add(self.globalReadIncrementAB(kernel, tensorParametersA, tensorParametersB, i, 0))
      module.add(self.closeLoop(kernel, tensorParametersA, tensorParametersB, i, True))

    module.add(self.endSummation(kernel))
    if not self.doShadowInit:
      module.add(self.globalWriteWorkGroupInit(kernel))

    ####################################
    # Shift Vector Components
    ####################################
    if kernel["EdgeType"] == "ShiftPtr":
      # GuaranteeNoPartial means each component in the vector loads is always valid.  In this case we
      # don't need the unshift code

      # shift vector components d0
      if not kernel["GuaranteeNoPartialA"] and tensorParametersA["rtv"]:
        module.addComment1("shift vector components d0")
        module.add(self.shiftVectorComponents(kernel, tensorParametersA))

      # shift vector components d1, for MFMA version, B never entered this
      if not kernel["GuaranteeNoPartialB"] and tensorParametersB["rtv"]:
        module.addComment1("shift vector components d1")
        module.add(self.shiftVectorComponents(kernel, tensorParametersB))

    ####################################
    # LocalSplitU reduction
    ####################################
    #if kernel["NumThreads"]%kernel["MacroTile0"] == 0:
    if kernel["LocalSplitU"] > 1:
      module.addComment2("LocalSplitU Reduction")
      module.add(self.syncThreads(kernel))

      # LocalSplitU: local write
      module.addComment1("LocalSplitU: local write")
      module.add(self.localSplitULocalWrite(kernel))

      # LocalSplitU: local read
      module.addComment1("LocalSplitU: local read")
      module.add(self.localSplitULocalRead(kernel))

      # LocalSplitU: local read
      module.addComment1("LocalSplitU: reduction")
      module.add(self.localSplitUReduction(kernel))

      # LocalSplitU: global write indices
      module.addComment1("LocalSplitU: global write indices")
      module.add(self.localSplitUGlobalWriteIndices(kernel))

      # LocalSplitU: global write
      module.addComment1("LocalSplitU: global write")
      module.add(self.localSplitUGlobalWrite(kernel, tensorParametersA, tensorParametersB))


    else:
      ####################################
      # NOT LocalSplitU
      ####################################

      # global write indices
      module.addComment1("not-LocalSplitU: global write indices")
      module.add(self.notLocalSplitUGlobalWriteIndices(kernel))

      # global write
      module.addComment1("not-LocalSplitU: global write")
      module.add(self.notLocalSplitUGlobalWrite(kernel, tensorParametersA, tensorParametersB))

    # function suffix
    module.add(self.functionEnd(True))
    module.add(self.functionSuffix(kernel))

    # Tensile pass
    tpo = TensilePassOptions()
    tpo.removeDupActFunc = kernel["ActivationFuncCall"]
    TensilePass(module, tpo)

    moduleKernelBody.addBody(module)
    self.checkResources(moduleKernelBody)

    # Tensile instruction pass
    TensileInstructionsPass(moduleKernelBody)

    error = self.overflowedResources
    return (error, str(moduleKernelBody))

  ##############################################################################
  # Init Kernel
  ##############################################################################
  @abc.abstractmethod
  def initKernel(self, kernel, tensorParametersA, tensorParametersB ):
    # ISA version, such as 803
    self.version = tuple(kernel["ISA"])
    ti = TensileInstructions()
    ti.setKernelInfo(self.version, kernel["WavefrontSize"])
    self.staggerU = kernel["StaggerU"] and (kernel["KernelLanguage"]=="Source" or kernel["BufferLoad"])

    # Only assembly supports scheduling
    self.canSchedule = (kernel["KernelLanguage"] == "Assembly")

    if self.canSchedule:
      self.scheduleGlobalRead = kernel["ScheduleGlobalRead"] \
          and kernel["PrefetchGlobalRead"] \
          and kernel["BufferLoad"] # flat updates lgkmcnt counts = hard to schedule flat loads
    else:
      self.scheduleGlobalRead = 0

    if self.canSchedule:
      self.scheduleLocalWrite = kernel["ScheduleLocalWrite"] \
          and kernel["PrefetchGlobalRead"] \
          and kernel["BufferLoad"]  # flat updates lgkmcnt counts = hard to schedule writes and loads?
    else:
      self.scheduleLocalWrite = 0

    if self.canSchedule:
      self.scheduleIterAlg = kernel["ScheduleIterAlg"]
    else:
      self.scheduleIterAlg = 0

    self.noTailLoop = kernel["NoTailLoop"]

    self.actualSummationLoops = kernel["ProblemType"]["NumIndicesSummation"]
    self.otherSummationLoops  = self.actualSummationLoops-1
    self.otherSummations      = kernel["ProblemType"]["NumIndicesSummation"]-1 # not loops but summations vars

    if kernel["KernelLanguage"] == "Source":
      self.language = globalParameters["RuntimeLanguage"]
    else:
      self.language = "ASM"
    self.indexChars = []
    for i in range(0, len(globalParameters["IndexChars"])):
      self.indexChars.append(globalParameters["IndexChars"][i])
    self.indexChars[kernel["ProblemType"]["Index0"]] \
        = "0" + self.indexChars[kernel["ProblemType"]["Index0"]]
    self.indexChars[kernel["ProblemType"]["Index1"]] \
        = "1" + self.indexChars[kernel["ProblemType"]["Index1"]]
    self.unrollIdx = kernel["ProblemType"]["NumIndicesSummation"]-1
    self.unrollChar = \
        self.indexChars[kernel["ProblemType"]["IndicesSummation"][\
        self.unrollIdx]]
    self.tileChar0 = self.indexChars[kernel["ProblemType"]["Index0"]]
    self.tileChar1 = self.indexChars[kernel["ProblemType"]["Index1"]]
    self.tileCharA = self.tileChar0 if (kernel["ProblemType"]["Tensor0"]==0) \
        else self.tileChar1
    self.tileCharB = self.tileChar0 if (kernel["ProblemType"]["Tensor0"]==1) \
        else self.tileChar1

    """
    if kernel["ProblemType"]["Tensor0"]==0:
      kernel["ThreadTileA"] = kernel["ThreadTile0"]
      kernel["ThreadTileB"] = kernel["ThreadTile1"]
      kernel["SubGroupA"] = kernel["SubGroup0"]
      kernel["SubGroupB"] = kernel["SubGroup1"]
      kernel["MacroTileA"] = kernel["MacroTile0"]
      kernel["MacroTileB"] = kernel["MacroTile1"]
    else:
      kernel["ThreadTileB"] = kernel["ThreadTile0"]
      kernel["ThreadTileA"] = kernel["ThreadTile1"]
      kernel["SubGroupB"] = kernel["SubGroup0"]
      kernel["SubGroupA"] = kernel["SubGroup1"]
      kernel["MacroTileB"] = kernel["MacroTile0"]
      kernel["MacroTileA"] = kernel["MacroTile1"]
    """

    """
    # original parameters
    NumLoadsCoalesced -> NumLoadsPerpendicular
    # new intermediate parameters
    numReadsTile # nrt
    numReadsUnroll # nru
    numWritesCoal # nwc
    numWritesPerp # nwp
    numWritesCoalVecComp # nwvc
    numWritesPerpVecComp # nwvp
    """

    # TODO load sub-vector
    vwa = kernel["GlobalLoadVectorWidthA"]
    vwb = kernel["GlobalLoadVectorWidthB"]

    self.numItersPLR = kernel["PrefetchLocalRead"]%kernel["LoopIters"]
    self.numVgprBuffer = kernel["LoopIters"] if kernel["PrefetchLocalRead"] > kernel["LoopIters"] else kernel["PrefetchLocalRead"]
    # merge N iteration's read into 1 iteration if can't coalesce read
    # ex, A can coalesce read, B can't
    # MergeRead 0: ds_readAx1 ds_readBx1 mfma | ds_readAx1 ds_readBx1 mfma | => ds_readAx2 ds_readBx1 mfma | ds_readBx1 mfma |
    # MergeRead 1: ds_readAx1 ds_readBx1 mfma | ds_readAx1 ds_readAx1 mfma | => ds_readAx2 ds_readBx1 ds_readBx1 mfma | mfma |
    MergeRead = 0
    if not kernel["ProblemType"]["TLUA"] or MergeRead or kernel["allowLRVWforTLUandMI"]:
      if (not kernel["ProblemType"]["TLUA"]) and kernel["DirectToVgprA"]:
        # DirectToVgpr + TLU=False case, ignore LocalReadVectorWidth and use GlobalLoadVectorWidth instead.
        self.lrvwA = vwa
      else:
        self.lrvwA = kernel["LocalReadVectorWidth"]
    else:
      if kernel["EnableMatrixInstruction"]:
        self.lrvwA = kernel["MIInputPerThread"]
      else:
        self.lrvwA = 1
    if not kernel["ProblemType"]["TLUB"] or MergeRead or kernel["allowLRVWforTLUandMI"]:
      if (not kernel["ProblemType"]["TLUB"]) and kernel["DirectToVgprB"]:
        # DirectToVgpr + TLU=False case, ignore LocalReadVectorWidth and use GlobalLoadVectorWidth instead.
        self.lrvwB = vwb
      else:
        self.lrvwB = kernel["LocalReadVectorWidth"]
    else:
      if kernel["EnableMatrixInstruction"]:
        self.lrvwB = kernel["MIInputPerThread"]
      else:
        self.lrvwB = 1

    # DirectToVgprB + VW > 1 case, set lrvwB = VW
    # DirectToVgprB case, global load data directly goes to Vgpr.
    # If VW=2, it means lrwvB is 2.
    if kernel["DirectToVgprB"] and kernel["VectorWidth"] > 1:
      self.lrvwB = kernel["VectorWidth"]
    # DirectToVgpr + TLU=False case
    # set lrvw = VW
    self.vgprValuDouble = False
    #if kernel["DirectToVgprA"] and kernel["PrefetchLocalRead"] > 1 and (not kernel["ProblemType"]["TLUA"]) and kernel["VectorWidth"] > 1:
    if kernel["DirectToVgprA"] and (not kernel["ProblemType"]["TLUA"]) and (not kernel["ProblemType"]["TLUB"]) or \
       kernel["DirectToVgprB"] and (not kernel["ProblemType"]["TLUB"]) and (not kernel["ProblemType"]["TLUA"]):
      self.lrvwA = max(self.lrvwA, self.lrvwB)
      self.lrvwB = self.lrvwA
      if kernel["DepthU"] // kernel["MatrixInstK"] <= 2 and self.lrvwA > 1:
        # need to double vgprValu to avoid local read overwritting vgprValu registers
        self.vgprValuDouble = True

    # Wider LocalRead
    if kernel["EnableMatrixInstruction"]:
      self.numReadsIterCoalescedA = self.lrvwA // kernel["MIInputPerThread"]
      self.numReadsIterCoalescedB = self.lrvwB // kernel["MIInputPerThread"]
      if kernel["allowLRVWforTLUandMI"]:
        self.numReadsIterCoalescedA = 1
        self.numReadsIterCoalescedB = 1
    else:
      self.numReadsIterCoalescedA  = 1
      self.numReadsIterCoalescedB  = 1
    self.numIterPerCoalescedReadA = max(1,self.numReadsIterCoalescedA//kernel["InnerUnroll"])
    self.numIterPerCoalescedReadB = max(1,self.numReadsIterCoalescedB//kernel["InnerUnroll"])

    if kernel["ScheduleIterAlg"] == 3 or kernel["ScheduleIterAlg"] == 2:
      self.numMfmaPerIter = kernel["MIWaveTile"][0] * kernel["MIWaveTile"][1] * kernel["InnerUnroll"]
      if kernel["ProblemType"]["DataType"].isComplex(): self.numMfmaPerIter *= 4

    # NamedTuple is immutable
    class intermediateTPValues(NamedTuple):
      numReadsTile: int = -1
      numReadsUnroll: int = -1
      readTileDimVector: bool = False
      writeTileDimComponents: bool = False
      numWritesCoalVecComp: int = -1
      numWritesPerpVecComp: int = -1
      # convert tile/unroll to para/perp
      numReadsCoalVecComp: int = -1
      numReadsPerpVecComp: int = -1

    def readWriteVectors(mat, vw, kernel):
      ########################################
      # read vectors or vector components
      ########################################
      # Two dim: tile and unroll
      if kernel["ProblemType"]["TLU%s"%mat]: # NT no transpose
        numReadsTile = kernel["NumLoadsCoalesced%s"%mat]
        numReadsUnroll = kernel["NumLoadsPerpendicular%s"%mat]
        readTileDimVector = True # Vector
      else: # TN yes transpose
        numReadsTile = kernel["NumLoadsPerpendicular%s"%mat]
        numReadsUnroll = kernel["NumLoadsCoalesced%s"%mat]
        readTileDimVector = False # Scalar

      numReadsCoalVecComp   = vw
      numReadsPerpVecComp   = 1

      ########################################
      # write vectors or vector components
      ########################################
      if kernel["ProblemType"]["TLU%s"%mat] != kernel["UnrollMajorLDS%s"%mat]: # NT no transpose
        writeTileDimComponents = False # Vector
        # writeCoal indicates writes should be done in the coal dim or else perp
        numWritesCoalVecComp = vw // kernel["DepthULdsDivisor"]
        numWritesPerpVecComp = 1
      else: # TN yes transpose
        writeTileDimComponents = kernel["GlobalReadVectorWidth"] > 1 # Components
        numWritesCoalVecComp = 1
        numWritesPerpVecComp = vw

      return intermediateTPValues(numReadsTile, numReadsUnroll, readTileDimVector, \
        writeTileDimComponents, numWritesCoalVecComp, numWritesPerpVecComp, \
        numReadsCoalVecComp, numReadsPerpVecComp)

    itP = dict()
    itP["A"] = readWriteVectors("A", vwa, kernel)
    itP["B"] = readWriteVectors("B", vwb, kernel)

    self.getTensorParameters(tensorParametersA, kernel, itP, True)
    self.getTensorParameters(tensorParametersB, kernel, itP, False)

    tensorParametersA["PackedIndices"] = kernel["PackedC%uIndicesX"%tensorParametersA["tile01Idx"]]
    tensorParametersB["PackedIndices"] = kernel["PackedC%uIndicesX"%tensorParametersB["tile01Idx"]]

    # condition(s) to enable init accvgpr opt (initialize only the last set of accvgpr instead of whole accvgpr)
    self.useInitAccVgprOpt = False
    # enable for the following conditions
    if (kernel["DirectToVgprA"] or kernel["DirectToVgprB"]):
      self.useInitAccVgprOpt = True
    # force to disable for the following conditions
    if self.useInitAccVgprOpt:
      if kernel["PrefetchGlobalRead"] == 2:
        # PGR=2 case, K > DepthU * 2 is necessary ( if not noTailLoop, need > DepthU * 3)
        # (kernel["AssertSizeGreaterThan"][3] > DepthU * 2 (or 3)
        minDUnum = 2 if self.noTailLoop else 3
        if not (3 in kernel["AssertSizeGreaterThan"].keys() and kernel["AssertSizeGreaterThan"][3] >= kernel["DepthU"] * minDUnum):
          print2("InitAccVgprOpt is disabled because AssertSizeGreaterThan for K is not greater than DepthU * %u"%minDUnum)
          self.useInitAccVgprOpt = False
      if kernel["PrefetchGlobalRead"] == 1:
        # PGR=1 case, K > DepthU * 1 is necessary ( if not noTailLoop, need > DepthU * 2)
        # (kernel["AssertSizeGreaterThan"][3] > DepthU * 2 (or 3)
        minDUnum = 1 if self.noTailLoop else 2
        if not (3 in kernel["AssertSizeGreaterThan"].keys() and kernel["AssertSizeGreaterThan"][3] >= kernel["DepthU"] * minDUnum):
          print2("InitAccVgprOpt is disabled because AssertSizeGreaterThan for K is not greater than DepthU * %u"%minDUnum)
          self.useInitAccVgprOpt = False

    # condition(s) to enable singleNLL opt
    self.enableSingleNLLOpt = False
    if self.noTailLoop:
      pass
      # so far, not enabled for DirectToVgpr
      # Performance is better with Tensile, but does not perform better with HPL
      #if kernel["DirectToVgprA"] or kernel["DirectToVgprB"]:
      #  self.enableSingleNLLOpt = True

  ##############################################################################
  # Function Signature
  ##############################################################################
  @abc.abstractmethod
  def functionSignature(self):
    return ""

  ##############################################################################
  # Allocate Resources
  ##############################################################################
  @abc.abstractmethod
  def defineAndResources(self, kernel, tPA, tPB):
    return ""

  ##############################################################################
  # Check Resources
  ##############################################################################
  @abc.abstractmethod
  def checkResources(self, mkb) -> None:
    pass

  ##############################################################################
  # Global Read Addresses: Work-Group
  ##############################################################################
  @abc.abstractmethod
  def graWorkGroup(self, kernel):
    return ""

  ##############################################################################
  # Get Params For Tensor A/B
  ##############################################################################
  def getTensorParameters(self, tP, kernel, itP, tA):
    cM = "A" if tA else "B"
    tP["mirror"] = bool(kernel["ProblemType"]["MirrorDims%s" % ("A" if tA else "B")])
    if tA: # A
      tP["tensorIdx"] = 0                                   # tensor index A=0, B=1
      tP["tileChar"] = self.tileCharA                       # tile char I0 or J1
    else: # B
      tP["tensorIdx"] = 1
      tP["tileChar"] = self.tileCharB

    tP["isA"] = tA                                               # is this tensor A
    tP["isB"] = (not tA)                                         # is this tensor B
    tP["bpe"] = int(4*kernel["ProblemType"]["DataType"].numRegisters())
    tP["tensorChar"] = cM                                        # tensor character A/B
    tP["tileIdx"] = kernel["ProblemType"]["Index01%s"%cM]        # is the tile dimension of A the 0th or 1th index, i.e. Aki, tileIdx=0
    tP["tile01Idx"] = 1 if tP["tileIdx"] else 0
    tP["lsc"] = "LSC%s"%cM                                       # load size coalesced A/B, number of elements that get loaded along coalesced dimension with each load
    tP["lsp"] = "LSP%s"%cM                                       # load size perpendicular A/B, number of elements that get loaded along non-coalesced dimension with each load
    tP["lvc"] = "LVC%s"%cM                                       # "load size" in terms of number of short-vectors and not elements
    tP["lvp"] = "LVP%s"%cM                                       # "load size" in terms of number of short-vectors and not elements
    tP["rtv"] = itP[cM].readTileDimVector                        # bool in the tile dimension, reads will read vectors
    tP["wg"] = "WorkGroup%u" % (tP["tile01Idx"])                 # these are storing the actual strong to lookup the number from kernel dictionary
    tP["sg"] = "SubGroup%u" % (tP["tile01Idx"])
    tP["tt"] = "ThreadTile%u" % (tP["tile01Idx"])
    tP["mt"] = "MacroTile%u" % (tP["tile01Idx"])
    tP["tlu"] = kernel["ProblemType"]["TLU%s"%cM]                # thread stride is less than unroll stride, i.e., not transposing matrix
    tP["ia"] = kernel["ProblemType"]["IndexAssignments%s"%cM]    # array of index assignments
    tP["nrt"] = itP[cM].numReadsTile                             # number of reads along tile dimension
    tP["nru"] = itP[cM].numReadsUnroll                           # number of reads along unroll dimension
    tP["nrc"] = kernel["NumLoadsCoalesced%s"%cM]                 # number of reads along coalesced dimension
    tP["nrcv"] = itP[cM].numReadsCoalVecComp                     # number of vector components along coalesced dimension
    tP["nrp"] = kernel["NumLoadsPerpendicular%s"%cM]             # number of reads along perpendicular dimension
    tP["nrpv"] = itP[cM].numReadsPerpVecComp                     # number of vector components along perpendicular dimension
    tP["nwcv"] = itP[cM].numWritesCoalVecComp                    # number of vector component writes along coalesced dimension
    tP["nwpv"] = itP[cM].numWritesPerpVecComp                    # number of vector component writes along perpendicular dimension
    tP["glvw"] = kernel["GlobalLoadVectorWidth%s"%cM]
    tP["wtc"] = itP[cM].writeTileDimComponents                   # write vector components along tile dimension
    tP["idx"] = kernel["ProblemType"]["Index%d"%tP["tensorIdx"]] # index 0 is tile dimension belonging to A. Note 'idx' may not be in tP['ia'].
    tP["NonTemporal"] = kernel["NonTemporal%s"%cM]               # non-temporal read type

    # KernelWriterAssembly
    tP["localReadSwapByteOffset"]  = 0
    tP["localWriteSwapByteOffset"] = 0
    tP["gpr"] = {}

  ##############################################################################
  # Global Read Addresses: Tile Assignment A/B
  ##############################################################################
  @abc.abstractmethod
  def graTileAssignment(self, kernel, tP):
    return ""

  ##############################################################################
  # Global Read Addresses: Unroll Assignment A/B
  ##############################################################################
  @abc.abstractmethod
  def graUnrollAssignment(self, kernel, tP):
    return ""

  ##############################################################################
  # Global Read Addresses: Other Free Assignments
  ##############################################################################
  @abc.abstractmethod
  def graOtherFreeAssignments(self):
    return ""

  ##############################################################################
  # Global Read Addresses: Other Summation Assignments
  ##############################################################################
  @abc.abstractmethod
  def graOtherSummationAssignments(self, kernel):
    return ""

  ##############################################################################
  # Global Read Addresses: Tile Offsets A/B
  ##############################################################################
  @abc.abstractmethod
  def graTileOffsets(self, kernel, tP):
    return ""

  ##############################################################################
  # Global Read Addresses: Unroll Offsets A/B
  ##############################################################################
  @abc.abstractmethod
  def graUnrollOffsets(self, kernel, tP):
    return ""

  ##############################################################################
  # Global Read Addresses: Shift A/B
  ##############################################################################
  @abc.abstractmethod
  def graShift(self, kernel, tP):
    return ""

  ##############################################################################
  # Global Read Addresses: Final Offsets A/B
  ##############################################################################
  @abc.abstractmethod
  def graFinalOffsets(self, kernel, tP):
    return ""

  ##############################################################################
  # Global Read Addresses: Addresses A/B
  ##############################################################################
  @abc.abstractmethod
  def graAddresses(self, kernel, tP):
    return ""

  ##############################################################################
  # Global Read Addresses: Increments A/B
  # This function declares the increments
  ##############################################################################
  @abc.abstractmethod
  def graIncrements(self, kernel, loopIdx, tP):
    return ""

  ##############################################################################
  # Local Write Addresses: Tile Assignment A/B
  ##############################################################################
  @abc.abstractmethod
  def lwaTileAssignment(self, tP):
    return ""

  ##############################################################################
  # Local Write Addresses: Unroll Assignment A/B
  ##############################################################################
  @abc.abstractmethod
  def lwaUnrollAssignment(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Write Addresses: First Offset A/B
  ##############################################################################
  @abc.abstractmethod
  def lwaFirstOffset(self, kernel, tP, uDu):
    return ""

  ##############################################################################
  # Local Read Addresses: Tile Assignment
  ##############################################################################
  @abc.abstractmethod
  def lraTileAssignment(self, kernel, tPA, tPB):
    return ""

  ##############################################################################
  # Local Read Addresses: Final Offset A/B
  ##############################################################################
  @abc.abstractmethod
  def lraFinalOffset(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Read Addresses offset conversion for DTL + NLC > 1
  ##############################################################################
  @abc.abstractmethod
  def lraOffsetConversionForDTLandNLC(self, kernel, tP, offset_val, generateAsm=False, \
                                      finalVgpr=None, tmp1=None, tmp2=None):
    return ""

  ##############################################################################
  # Local Read Addresses: Declare Addresses A/B
  ##############################################################################
  @abc.abstractmethod
  def lraDeclareAddresses(self, kernel, tP):
    return ""

  ##############################################################################
  # Recalculate local read addresses A/B
  ##############################################################################
  @abc.abstractmethod
  def recalcLocalReadAddressesAB(self, kernel, tPA, tPB):
    return ""

  ##############################################################################
  # Recalculate local write addresses A/B
  ##############################################################################
  @abc.abstractmethod
  def recalcLocalWriteAddresses(self, kernel, tP, uDu):
    return ""

  ##############################################################################
  # Define stagger parms that will be used in calculateStagger
  ##############################################################################
  @abc.abstractmethod
  def declareStaggerParms(self, kernel):
    return ""


  ##############################################################################
  # Calculate and apply stagger offsets and edge
  ##############################################################################
  @abc.abstractmethod
  def calculateStagger(self, kernel, loopIdx):
    return ""

  ##############################################################################
  # Remove stagger offset (before tail loop)
  ##############################################################################
  @abc.abstractmethod
  def removeStagger(self, kernel):
    return ""

  ##############################################################################
  # Calculate Loop Num Iter
  ##############################################################################
  @abc.abstractmethod
  def calculateLoopNumIter(self, kernel, tPA, tPB, loopIdx):
    return ""


  ##############################################################################
  # openShadowInit:
  # Top of shadow init code
  ##############################################################################
  @abc.abstractmethod
  def openShadowInit(self):
    return ""

  ##############################################################################
  # closeShadowInit:
  # Top of shadow init code
  ##############################################################################
  @abc.abstractmethod
  def closeShadowInit(self, kernel):
    return ""

  ##############################################################################
  # Initialize C
  ##############################################################################
  @abc.abstractmethod
  def initC(self, kernel):
    return ""

  ##############################################################################
  # Open Loop
  # loopIdx<0 : tail loop
  ##############################################################################
  @abc.abstractmethod
  def openLoop(self, kernel, tPA, tPB, loopIdx, uDu, noLabelGen, beginLabelOnly):
    return ""

  ##############################################################################
  # Close Loop
  ##############################################################################
  @abc.abstractmethod
  def closeLoop(self, kernel, tPA, tPB, loopIdx, \
                finalLoop, uDu, emitEndLabelOnly, oddLabel=False):
    return ""

  ##############################################################################
  # End Summation
  ##############################################################################
  @abc.abstractmethod
  def endSummation(self, kernel, label = None):
    return ""

  ##############################################################################
  # At Least 1 Unroll
  ##############################################################################
  @abc.abstractmethod
  def openSumAtLeastUnroll(self, kernel, prefetch, isOptNLL):
    return ""

  @abc.abstractmethod
  def closeSumAtLeastUnroll(self, kernel, tPA, tPB, prefetch, isOptNLL, isNGLL):
    return ""

  ##############################################################################
  # Global Read: Increment A/B
  ##############################################################################
  @abc.abstractmethod
  def globalReadIncrementAB(self, kernel, tPA, tPB, loopIdx, prefetchIndex):
    return ""

  ##############################################################################
  # Global Read: Do It A/B
  # mode: 0=prefetch, 1=unroll loop, 2=guardK
  ##############################################################################
  @abc.abstractmethod
  def globalReadDo(self, kernel, mode, tP, vregSetIdx=0):
    return ""

  ##############################################################################
  # directToLds m0 update: Do It A/B
  # mode: 0=prefetch, 1=unroll loop, 2=guardK
  ##############################################################################
  @abc.abstractmethod
  def directToLdsM0Update(self, kernel, mode, tP, usePlaceHolder=False):
    return ""

  ##############################################################################
  # Local Write: Swap Offsets A/B
  ##############################################################################
  @abc.abstractmethod
  def localWriteSwapOffsets(self, kernel, internalPointerSwap, tP):
    return ""

  ##############################################################################
  # Local Write: Reset Offsets A/B
  ##############################################################################
  @abc.abstractmethod
  def localWriteResetOffsets(self, kernel, internalPointerSwap, tP):
    return ""

  ##############################################################################
  # Local Write in Prefetch Pass (PreLoop): Do It A/B
  ##############################################################################
  @abc.abstractmethod
  def preLoopLocalWriteDo(self, kernel, tPA, tPB):
    return ""

  ##############################################################################
  # Local Write: Do It A/B
  ##############################################################################
  @abc.abstractmethod
  def localWriteDo(self, kernel, tP, uDu):
    return ""

  ##############################################################################
  # Local Read: Swap Offsets A/B
  ##############################################################################
  @abc.abstractmethod
  def localReadSwapOffsets(self, kernel, internalPointerSwap, tP):
    return ""

  ##############################################################################
  # Local Read: Reset Offsets A/B
  ##############################################################################
  @abc.abstractmethod
  def localReadResetOffsets(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Read: Init Pointers A/B
  ##############################################################################
  @abc.abstractmethod
  def localReadInitPointers(self, kernel, tPA, tP):
    return ""

  ##############################################################################
  # Local Read: Increment A/B
  ##############################################################################
  @abc.abstractmethod
  def localReadInc(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Read: Do It A/B
  ##############################################################################
  @abc.abstractmethod
  def localReadDo(self, kernel, bufferIdx, innerUnrollIndex, epsi, tP):
    return ""

  ##############################################################################
  # Shift Vector Components d0/1
  ##############################################################################
  @abc.abstractmethod
  def shiftVectorComponents(self, kernel, tP):
    return ""

  ##############################################################################
  # LocalSplitU: Local Write
  ##############################################################################
  @abc.abstractmethod
  def localSplitULocalWrite(self, kernel):
    return ""

  ##############################################################################
  # LocalSplitU: Local Read
  ##############################################################################
  @abc.abstractmethod
  def localSplitULocalRead(self, kernel):
    return ""

  ##############################################################################
  # LocalSplitU: Reduction
  ##############################################################################
  @abc.abstractmethod
  def localSplitUReduction(self, kernel):
    return ""

  ##############################################################################
  # globalWriteWorkGroupInit:
  # Perform work-group granularity init
  ##############################################################################
  @abc.abstractmethod
  def globalWriteWorkGroupInit(self, kernel):
    return ""

  ##############################################################################
  # LocalSplitU: Global Write Indices
  ##############################################################################
  @abc.abstractmethod
  def localSplitUGlobalWriteIndices(self, kernel):
    return ""

  ##############################################################################
  # LocalSplitU: Global Write
  ##############################################################################
  @abc.abstractmethod
  def localSplitUGlobalWrite(self, kernel, tPA, tPB):
    return ""

  ##############################################################################
  # Not LocalSplitU: Global Write Indices
  ##############################################################################
  @abc.abstractmethod
  def notLocalSplitUGlobalWriteIndices(self, kernel):
    return ""

  ##############################################################################
  # Not LocalSplitU: Global Write
  ##############################################################################
  @abc.abstractmethod
  def notLocalSplitUGlobalWrite(self, kernel, tPA, tPB):
    return ""

  ##############################################################################
  # openOddNoLoadLoopForDTV
  # generate open code for DirectToVgpr + odd exit case in noLoadLoop code
  ##############################################################################
  @abc.abstractmethod
  def openOddNoLoadLoopForDTV(self, name):
    return ""

  ##############################################################################
  # closeOddNoLoadLoopForDTV
  # generate close code for DirectToVgpr + odd exit case in noLoadLoop code
  ##############################################################################
  @abc.abstractmethod
  def closeOddNoLoadLoopForDTV(self, name):
    return ""

  ##############################################################################
  # generateEvenEndLabeNoLoadLoopForDTV
  # generate even end label for DirectToVgpr
  ##############################################################################
  @abc.abstractmethod
  def generateEvenEndLabeNoLoadLoopForDTV(self, name):
    return ""

  ##############################################################################
  # generateOddEndVgprCopyForDTV
  # generate odd end vgpr copy for DirectToVgpr
  ##############################################################################
  @abc.abstractmethod
  def generateOddEndVgprCopyForDTV(self, kernel):
    return ""

  ##############################################################################
  # PrefetchGlobalRead2
  ##############################################################################
  @abc.abstractmethod
  def openPrefetchGlobalRead2(self, kernel):
    return ""

  @abc.abstractmethod
  def closePrefetchGlobalRead2(self):
    return ""

  ##############################################################################
  # Function End
  ##############################################################################
  @abc.abstractmethod
  def functionEnd(self, addLabel=True):
    return ""

  ##############################################################################
  # Function Suffix
  ##############################################################################
  @abc.abstractmethod
  def functionSuffix(self, kernel):
    return ""

  ##############################################################################
  # WaitCnt
  ##############################################################################
  @abc.abstractmethod
  def wait(self, kernel, tPA, tPB, globalRead, localWrite, localRead, comment):
    return ""

  ##############################################################################
  # SyncThreads
  ##############################################################################
  @abc.abstractmethod
  def syncThreads(self, kernel):
    return ""

  ##############################################################################
  # MapAcctoArch
  ##############################################################################
  @abc.abstractmethod
  def MapAcctoArchRegs(self, kernel):
    return ""

  ##############################################################################
  #
  #   Entry Functions
  #
  ##############################################################################


  ##############################################################################
  # get kernel name
  ##############################################################################
  def getKernelFileBase(self, kernel):
    if isCustomKernelConfig(kernel):
      fileBase = kernel["CustomKernelName"]
    elif globalParameters["ShortNames"]:
      fileBase = Solution.getNameSerial(kernel, self.kernelSerialNaming)
    else:
      fileBase = self.shortenFileBase(kernel)
    return fileBase

  def getKernelName(self, kernel):
    kernelName = Solution.getNameMin(kernel, self.kernelMinNaming)
    return kernelName

  def getKernelSource(self, kernel):
    """
    Returns the source of the kernel, either C++ or assembly.
    """


    fileString = ""
    tensorParametersA = {}
    tensorParametersB = {}
    self.initKernel(kernel, tensorParametersA, tensorParametersB )
    self.stringIdx = 0
    (error, kb) = self.kernelBody( kernel, tensorParametersA, tensorParametersB)
    fileString += str(kb)

    if error != 0:
      if globalParameters["ForceGenerateKernel"]:
        print ("warning: Generating kernel source resulted in error {}, but ForceGenerateKernel=1 so saving source".format(error))
      else:
        raise RuntimeError("Generating kernel source resulted in error {}".format(error))
    return fileString

  def getAssemblyDirectory(self):
      return Common.ensurePath(os.path.join(globalParameters["WorkingPath"], "assembly"))

  def byteArrayScriptSource(self):
    return """
#!/usr/bin/env python

fileString = ""
fileString += "/*******************************************************************************\\n"
fileString += "* Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.\\n"
fileString += "*\\n"
fileString += "* Permission is hereby granted, free of charge, to any person obtaining a copy\\n"
fileString += '* of this software and associated documentation files (the \"Software\"), to deal\\n'
fileString += "* in the Software without restriction, including without limitation the rights\\n"
fileString += "* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-\\n"
fileString += "* ies of the Software, and to permit persons to whom the Software is furnished\\n"
fileString += "* to do so, subject to the following conditions:\\n"
fileString += "*\\n"
fileString += "* The above copyright notice and this permission notice shall be included in all\\n"
fileString += "* copies or substantial portions of the Software.\\n"
fileString += "*\\n"
fileString += '* THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-\\n'
fileString += "* PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS\\n"
fileString += "* FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR\\n"
fileString += "* COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER\\n"
fileString += "* IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-\\n"
fileString += "* CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.\\n"
fileString += "*******************************************************************************/\\n\\n"
fileString += "/**************************************************\\n"
fileString += "* This file was generated by Tensile:             *\\n"
fileString += "* https://github.com/ROCmSoftwarePlatform/Tensile *\\n"
fileString += "**************************************************/\\n\\n\\n"
import os.path
fileString += '#include "Kernels.h"\\n\\n'
fileString += "/* code object byte array */\\n\\n"
codeObjectFileNames = [f for f in os.listdir(".") if (os.path.isfile(f) and f.endswith(".co"))]
for codeObjectFileName in codeObjectFileNames:
  print codeObjectFileName
  print "\\n"
  kernelName=os.path.splitext(codeObjectFileName)[0]
  codeObjectFile = open(codeObjectFileName, "r")
  codeObjectByteArray = bytearray(codeObjectFile.read())
  codeObjectFile.close()
# write code object byte array for asm
  fileString += "const unsigned char %s_coba[%u] = {\\n" % (kernelName, len(codeObjectByteArray))
  for byteIdx in range(0, len(codeObjectByteArray)):
    byte = codeObjectByteArray[byteIdx]
    fileString += "0x%02x" % byte
    if byteIdx < len(codeObjectByteArray)-1:
      fileString += ","
    else:
      fileString += "};\\n"
    if byteIdx % 16 == 15:
      fileString += "\\n"
  text_file = open("Kernels.cpp", "w")
  text_file.write("%s" % fileString)
  text_file.close()
"""

  def writeByteArrayScript(self):
    asmPath = self.getAssemblyDirectory()

    bytearrayFileName = os.path.join(asmPath,"insert_byte_array.py")
    if not os.path.isfile(bytearrayFileName):
      with open(bytearrayFileName, 'w') as bytearrayFile:
        bytearrayFile.write(self.byteArrayScriptSource())
      os.chmod(bytearrayFileName, 0o777)
    return bytearrayFileName

  def shortenFileBase(self, kernel):
    base = self.getKernelName(kernel)
    if len(base) <= globalParameters["MaxFileName"]:
      return base

    import hashlib
    import base64

    pivot = globalParameters["MaxFileName"] * 3 // 4
    firstPart = base[:pivot]
    secondPart = base[pivot:]

    secondHash = hashlib.sha256(secondPart.encode()).digest()
    secondPart = base64.b64encode(secondHash, b'_-').decode()

    return firstPart + secondPart

  def getKernelObjectAssemblyFile(self, kernel):
    asmPath = self.getAssemblyDirectory()
    # write assembly file to assembly directory
    kernelName = self.getKernelFileBase(kernel)
    fileBase = os.path.join(asmPath, kernelName )
    assemblyFileName = "%s.s" % fileBase

    kernelSource = self.getKernelSource(kernel)

    if globalParameters["PrintLevel"] >= 2:
      print("write_assemblyFilename %s" % assemblyFileName)

    with open(assemblyFileName, 'w') as assemblyFile:
      assemblyFile.write(kernelSource)

    return assemblyFileName

  def getAssembledKernelObjectFile(self, kernel):
    assemblyFileName = self.getKernelObjectAssemblyFile(kernel)

    base, ext = os.path.splitext(assemblyFileName)
    objectFileName = base + '.o'

    args = self.getCompileArgs(assemblyFileName, objectFileName)
    if globalParameters["PrintCodeCommands"]:
      print (' '.join(args), " && ")

    subprocess.check_call(args, cwd=self.getAssemblyDirectory())

    return objectFileName

  def getSingleCodeObjectFile(self, kernel):
    objectFileName = self.getAssembledKernelObjectFile(kernel)

    base, ext = os.path.splitext(objectFileName)
    coFileName = base + '.co'

    args = self.getLinkCodeObjectArgs([objectFileName], coFileName)
    if globalParameters["PrintCodeCommands"]:
      print (' '.join(args))

    subprocess.check_call(args, cwd=self.getAssemblyDirectory())

    return coFileName

  ##############################################################################
  def getSourceFileString(self, kernel):
    """
    Returns a string suitable for placing in Kernels.cpp.  This means the actual kernel source in the case
    of a source kernel, or an assembled code object byte array definition in the case of an assembly kernel,
    or an empty string in the case that CodeFromFiles is true.

    In the case of an assembly kernel, this function has the side effect of creating the following files:
     * An assembly source file
     * An object file
     * A code object file
     * A Python script which can create byte array variable definitions.
    """

    try:
      if kernel["KernelLanguage"] == "Assembly":
        # asmPath = self.getAssemblyDirectory()
        # kernelName = self.getKernelName(kernel)

        if globalParameters["GenerateSourcesAndExit"]:
          # only create the assembly file.
          self.getKernelObjectAssemblyFile(kernel)
          return (0, "")
        else:
          self.writeByteArrayScript()
          self.getSingleCodeObjectFile(kernel)

          # I guess in this case we are making sure that the code object file exists by executing the code
          # above but we aren't placing it into the source.
          return (0, "")

      else:
        return (0, self.getKernelSource(kernel))

    except subprocess.CalledProcessError as exc:
      print(exc)
      return (-1, "")
    except RuntimeError as exc:
      if globalParameters["PrintSolutionRejectionReason"]:
        print(exc)
      return (-2, "")

  ##############################################################################
  # header file string
  ##############################################################################
  def getHeaderFileString(self, kernel):
    kernelName = self.getKernelName(kernel)
    fileString = "" # CHeader
    if self.language == "HIP" or self.language == "OCL":
      if not globalParameters["MergeFiles"]:
        fileString += CHeader
        fileString += "#pragma once\n\n"
        if self.language == "HIP":
          fileString += "#include <hip/hip_runtime.h>\n"
          fileString += "#include <hip/hip_fp16.h>\n"
          fileString += "#include <KernelHeader.h>\n"
          fileString += "\n"
        else:
          fileString += "#include <string>\n"
      if self.language == "OCL":
        fileString += "extern const char * const %s_src;\n" % kernelName
      else:
        # Asm should not reach here
        fileString += self.functionSignature()
        fileString += ";\n"
    else:
      if not globalParameters["MergeFiles"] or globalParameters["NumMergedFiles"] > 1:
        fileString += "#pragma once\n\n"
      if not globalParameters["CodeFromFiles"]:
        fileString += "extern const unsigned char %s_coba[]; // code object byte array\n" % kernelName

    return fileString

  ##############################################################################
  # flip Vreg set for DirectToVgpr in global read
  ##############################################################################
  def replaceSet(module: Item, srcStr, dst):
    if isinstance(module, Module):
      for item in module.items():
        replaceSet(item, srcStr, dst)
    elif isinstance(item, GlobalReadInstruction):
      if isinstance(item, FLATReadInstruction):
        item.dst.replaceRegName(srcStr, dst)
      elif isinstance(item, MUBUFReadInstruction):
        item.dst.replaceRegName(srcStr, dst)
    elif isinstance(item, GlobalWriteInstruction):
      if isinstance(item, FLATStoreInstruction):
        item.srcData.replaceRegName(srcStr, dst)
      elif isinstance(item, MUBUFStoreInstruction):
        item.srcData.replaceRegName(srcStr, dst)
    elif isinstance(module, SWaitCnt):
      assert(isinstance(dst, int))
      if module.vmcnt == srcStr:
        module.vmcnt = dst
      if module.lgkmcnt == srcStr:
        module.lgkmcnt = dst
      if module.vscnt == srcStr:
        module.vscnt = dst

  def flipVregSetForDirectToVgprInGlobalRead(self, kernel, item):
    # need to swap VGPR register set for odd code
    baseName = "G2LA" if kernel["DirectToVgprA"] else "G2LB" # only one of them is enabled
    set0 = baseName + "0"
    set1 = baseName + "1"
    itemStr = str(item)
    if set0 in itemStr:
      # replace set0 with set1
      replaceSet(item, set0, set1)
    elif set1 in itemStr:
      # replace set1 with set0
      replaceSet(item, set1, set0)
    return item

  ##############################################################################
  # waitcnt code for DirectToVgpr
  ##############################################################################
  def getWaitcntCodeForDirectToVgpr(self, kernel, localWriteEndIter, u, firstIter, beforeBarrier=False, NLLlast=False, oddLast=False):
    inst = TextBlock(slash50("No need to wait."))
    # generate wait only if BufferLoad is True (this logic does not work with FlatLoad)
    if (kernel["DirectToVgprA"] or kernel["DirectToVgprB"]) and kernel["BufferLoad"]:
      pgr2 = kernel["PrefetchGlobalRead"] == 2
      numGlobalReadA = kernel["NumLoadsPerpendicularA"] * kernel["NumLoadsCoalescedA"]
      numGlobalReadB = kernel["NumLoadsPerpendicularB"] * kernel["NumLoadsCoalescedB"]
      numGlobalRead = numGlobalReadA if kernel["DirectToVgprA"] else numGlobalReadB
      numGlobalReadAll = numGlobalReadA + numGlobalReadB
      numGlobalStoreC = 0
      numReadsIterCoalesced = self.numReadsIterCoalescedA if kernel["DirectToVgprA"] else self.numReadsIterCoalescedB
      waitComment = "global read wait for DirectToVgpr"
      # delay DirectToVgpr global read (from previous iteration) which is not referred yet
      numRegsIn1set = (numGlobalRead // kernel["LoopIters"]) * numReadsIterCoalesced
      numSet = (u + numReadsIterCoalesced) // numReadsIterCoalesced
      numSetMod = (u + numReadsIterCoalesced) % numReadsIterCoalesced
      if numSetMod > 0:
        # if mod > 0, wait is already done by mod == 0 case and no need to wait for same set of global read
        return inst
      needToWait = numGlobalRead - numSet * numRegsIn1set
      # No global load A, B in no load loop. Reset numGlobalReadAll and numGlobalRead
      numGlobalReadAll = 0
      numGlobalRead = 0
      if pgr2:
        # PGR=2 case, add numGlobalReadAll for second set of prefetch
        needToWait += numGlobalReadAll
      if u > 0:
        # count number of global read for i < u
        count = 0
        for i in range(u):
          globalReadStr = ' '.join([str(x) for x in self.perIterGlobalReadCode[i].flatitems()])
          count += globalReadStr.count("_buffer_load")
          # PGR=2 case, global read is in LocalWriteCode
          localWriteStr = ' '.join([str(x) for x in self.perIterLocalWriteCode[i].flatitems()])
          count += localWriteStr.count("_buffer_load")
        needToWait += count
        if u == localWriteEndIter + 1 and beforeBarrier:
          # beforeBarrier case, reduce the amount of non-Vgpr global read
          needToWait -= (numGlobalReadAll - numGlobalRead)
      # adjustment for oddLast
      # oddLast case, ignore all of above and set 0
      if oddLast:
        needToWait = 0
      inst = SWaitCnt(vmcnt=needToWait, comment=waitComment)
    return inst
