# ########################################################################
# Copyright (C) 2024 Advanced Micro Devices, Inc.
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################


# ------------------------------------------------------------------------
#
# Usage:
# 
# Assuming one runs the following Tensile command:
#
#   Tensile/bin/Tensile <yaml> tensile-out
#
# where tensile-out has the following directory structure:
#
# tensile-out
# ├── 0_Build
# ├── 1_BenchmarkProblems
# │   ├── Cijk_Ailk_Bjlk_1
# |   ├── Cijk_Ailk_Bjlk_2
# |   ...
# |   ├── Cijk_Ailk_Bjlk_N
#
# One can invoke make with this Makefile as follows:
#
# make co PROBLEM_DIR=tensile-out/1_BenchmarkProblems/Cijk_Ailk_Bjlk_1
#
# where PROBLEM_DIR is the relative path to one problem directory.
#
# The Makefile will set the target based on the name of the co file and sets
# a default wavefront flag but each of these can be customized as follows:
#
# make co PROBLEM_DIR=tensile-out/1_BenchmarkProblems/Cijk_Ailk_Bjlk_DB_UserArgs_00 \
#         ARCH="gfx942:xnack-"\
#         WAVEFRONTSIZE="-mno-wavefrontsize64"
#
# In addition, we provide `ASM_ARGS` and `LINK_ARGS` as additional customization
# points for the assemble and link step respectively.
# ------------------------------------------------------------------------

AS := /opt/rocm/bin/amdclang++
LDD := /opt/rocm/bin/amdclang++
SRCFILES := $(shell find ${PROBLEM_DIR}/00_Final/source/build_tmp/SOURCE/assembly -name '*.s')
OBJFILES := $(patsubst %.s,%.o,$(SRCFILES))
COFILE := $(shell find ${PROBLEM_DIR}/00_Final/source/library -name '*.co')
ARCH := $(subst .co,,$(subst TensileLibrary_,,$(lastword $(subst /, ,$(COFILE)))))
WAVEFRONTSIZE := -mwavefrontsize64

co: $(COFILE)

$(COFILE): $(OBJFILES)
	@echo relinking $(lastword $(subst /, ,$@))
	@$(LDD) -target amdgcn-amd-amdhsa -Xlinker --build-id=sha1 ${LINK_ARGS} -o $@ $^

%.o: %.s
	@echo rebuilding $(lastword $(subst /, ,$@))
	@$(AS) -x assembler -target amdgcn-amd-amdhsa -mcode-object-version=4 -mcpu=${ARCH} ${WAVEFRONTSIZE} ${ASM_ARGS} -c -o $@ $^
