# Copyright (C) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Bench launch utils."""

import logging
import pathlib
import asyncio
import sys
from collections import defaultdict
from asyncio.subprocess import PIPE, STDOUT


def run_yaml(bench,
             yamlfile,
             verbose=False,
             timeout=10):
    """Run hipblaslt-bench"""
    cmd = [pathlib.Path(bench).resolve()]
    cmd += ['--yaml', yamlfile]

    cmd = [str(x) for x in cmd]
    logging.info('hipblaslt-bench: ' + ' '.join(cmd))
    if verbose:
        print('hipblaslt-bench: ' + ' '.join(cmd))

    startingToken = "["
    csvKeys = ''
    benchResultsList = []
    capturingValues = False

    async def run_command(*args, timeout=None):

        process = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE)

        nonlocal startingToken
        nonlocal csvKeys
        nonlocal benchResultsList
        nonlocal capturingValues

        while True:
            try:
                line = await asyncio.wait_for(process.stdout.readline(),
                                              timeout)
            except asyncio.TimeoutError:
                logging.info(
                    "timeout expired. killed. Please check the process.")
                print("timeout expired. killed. Please check the process.")
                process.kill()  # Timeout or some criterion is not satisfied
                break

            if not line:
                break
            else:
                line = line.decode('utf-8').rstrip('\n')
                line = line.strip()
                if capturingValues:
                    print(line)
                    dd_output = defaultdict(str, zip(csvKeys, line.split(',')))
                    benchResultsList += [dd_output]
                    capturingValues = False
                elif line.startswith(startingToken):
                    line = line.replace('hipblaslt-Gflops', 'gflops')
                    line = line.split(':')[1]
                    print(f'\n{line}')
                    csvKeys = line.split(',')
                    capturingValues = True
        return await process.wait()  # Wait for the child process to exit

    if sys.platform == "win32":
        loop = asyncio.ProactorEventLoop()  # For subprocess' pipes on Windows
        asyncio.set_event_loop(loop)
    else:
        loop = asyncio.new_event_loop()

    returncode = loop.run_until_complete(run_command(*cmd, timeout=timeout))
    success = returncode == 0

    loop.close()

    return csvKeys, benchResultsList, success