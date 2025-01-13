/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "argument_model.hpp"
#include "performance_monitor.hpp"

// this should have been a member variable but due to the complex variadic template this singleton allows global control

static bool log_function_name = false;

void ArgumentModel_set_log_function_name(bool f)
{
    log_function_name = f;
}

bool ArgumentModel_get_log_function_name()
{
    return log_function_name;
}

void ArgumentModel_log_performance(hipblaslt_internal_ostream& name_line,
                                   hipblaslt_internal_ostream& val_line)
{

    PerformanceMonitor& performance_monitor = getPerformanceMonitor();
    if(!performance_monitor.enabled())
        return;

    name_line << ",Total Granularity";
    val_line << "," << performance_monitor.getTotalGranularityValue();

    name_line << ",Tiles Per-CU";
    val_line << "," << performance_monitor.getTilesPerCuValue();

    name_line << ",Tile-0 Granularity";
    val_line << "," << performance_monitor.getTile0Granularity();

    name_line << ",Tile-1 Granularity";
    val_line << "," << performance_monitor.getTile1Granularity();

    name_line << ",CU granularity";
    val_line << "," << performance_monitor.getCuGranularity();

    name_line << ",Wave granularity";
    val_line << "," << performance_monitor.getWaveGranularity();

    name_line << ",#CU's";
    val_line << "," << performance_monitor.getCUs();

    name_line << ",mem-read-bytes";
    val_line << "," << performance_monitor.getMemReadBytes();

    name_line << ",mem-write-bytes";
    val_line << "," << performance_monitor.getMemWriteBytesD();

    if(!performance_monitor.detailedReport())
    {
        name_line << ",lowest-avg-freq";
        val_line << "," << performance_monitor.getLowestAverageSYSCLK();

        name_line << ",lowest-median-freq";
        val_line << "," << performance_monitor.getLowestMedianSYSCLK();
    }
    else
    {
        auto allAvgSYSCLK = performance_monitor.getAllAverageSYSCLK();
        for(int i = 0; i < allAvgSYSCLK.size(); i++)
        {
            name_line << ",avg-freq_" << i;
            val_line << "," << allAvgSYSCLK[i];
        }

        auto allMedianSYSCLK = performance_monitor.getAllMedianSYSCLK();
        for(int i = 0; i < allMedianSYSCLK.size(); i++)
        {
            name_line << ",median-freq_" << i;
            val_line << "," << allMedianSYSCLK[i];
        }
    }

    name_line << ",avg-MCLK";
    val_line << "," << performance_monitor.getAverageMEMCLK();

    name_line << ",median-MCLK";
    val_line << "," << performance_monitor.getMedianMEMCLK();
}
