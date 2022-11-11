################################################################################
# Copyright 2022 Advanced Micro Devices, Inc. All rights reserved.
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

from Tensile.AsmRegisterPool import RegisterPool, allocTmpGpr

def test_alloc_tmp_gpr_simple():
    TEST_SIZE = 128
    pool = RegisterPool(TEST_SIZE, 's', True)
    assert pool.size() == TEST_SIZE
    pool.addRange(0, TEST_SIZE - 1)
    assert pool.available() == TEST_SIZE

    for k in range(1, 11):
        with allocTmpGpr(pool, k, pool.size(), 1, 'unittest_alloc'):
            assert pool.available() + k == TEST_SIZE
        assert pool.size() == TEST_SIZE
    assert pool.available() == TEST_SIZE

def test_alloc_tmp_gpr_nested():
    TEST_SIZE = 128
    pool = RegisterPool(TEST_SIZE, 's', True)
    assert pool.size() == TEST_SIZE
    pool.addRange(0, TEST_SIZE - 1)
    assert pool.available() == TEST_SIZE

    with allocTmpGpr(pool, 1, pool.size(), 1, 'unittest_alloc') as gpr0:
        assert pool.available() + 1 == TEST_SIZE
        with allocTmpGpr(pool, 1, pool.size(), 1, 'unittest_alloc') as gpr1:
            assert pool.available() + 2 == TEST_SIZE
            assert gpr0.idx != gpr1.idx
    assert pool.size() == TEST_SIZE

def test_alloc_tmp_gpr_overflow():
    TEST_SIZE = 4 
    pool = RegisterPool(TEST_SIZE, 's', True)
    assert pool.size() == TEST_SIZE
    pool.addRange(0, TEST_SIZE - 1)
    assert pool.available() == TEST_SIZE

    try:
        with allocTmpGpr(pool, pool.size() + 1, pool.size(), 1, None):
            pass
    except RegisterPool.ResourceOverflowException:
        return

    # should not reach here
    assert False

def test_alloc_tmp_gpr_nested_overflow():
    TEST_SIZE = 4 
    pool = RegisterPool(TEST_SIZE, 's', True)
    assert pool.size() == TEST_SIZE
    pool.addRange(0, TEST_SIZE - 1)
    assert pool.available() == TEST_SIZE

    with allocTmpGpr(pool, 2, pool.size(), 1, None):
        assert pool.available() == pool.size() - 2
        try:
            with allocTmpGpr(pool, 3, pool.size(), 1, None):
                pass
        except RegisterPool.ResourceOverflowException:
            return

    # should not reach here
    assert False

def test_wrapped_gpr_overflow():
    overflowed = False
    class SimpleWrapper:
        def __init__(self):
            TEST_SIZE = 4 
            self.pool = RegisterPool(TEST_SIZE, 's', True)
            assert self.pool.size() == TEST_SIZE
            self.pool.addRange(0, TEST_SIZE - 1)
            assert self.pool.available() == TEST_SIZE

        def alloc(self, num: int, aligmnent, opt):
            def overflow_listener(e):
                nonlocal overflowed
                overflowed = True
            return allocTmpGpr(self.pool, num, self.pool.size(), aligmnent, opt, overflow_listener)

    test_wrapper = SimpleWrapper()
    with test_wrapper.alloc(6, 1, None) as _:
        assert overflowed is True
