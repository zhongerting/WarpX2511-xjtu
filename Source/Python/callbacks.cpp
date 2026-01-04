/* Copyright 2019-2022 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: David Grote, Maxence Thevenet, Weiqun Zhang, Roelof Groenewald, Axel Huebl
 *
 * License: BSD-3-Clause-LBNL
 */
#include "callbacks.H"

#include <cstdlib>
#include <exception>
#include <iostream>


std::map< std::string, std::function<void()> > warpx_callback_py_map;

void InstallPythonCallback ( const std::string& name, std::function<void()> callback )
{
    warpx_callback_py_map[name] = std::move(callback);
}

bool IsPythonCallbackInstalled ( const std::string& name )
{
    return (warpx_callback_py_map.count(name) == 1u);
}

// Execute Python callbacks of the type given by the input string
/**
 * @brief 执行指定名称的Python回调函数
 * 
 * 该函数用于在WarpX模拟过程中的特定时间点执行用户定义的Python回调函数。
 * 它提供了C++代码与Python脚本之间的桥梁，允许用户在模拟的关键阶段插入自定义逻辑。
 * 
 * @param name 回调函数的名称标识符，用于在回调映射表中查找对应的Python函数
 * 
 * 功能流程：
 * 1. 检查指定名称的回调是否已安装（注册）
 * 2. 如果已安装，启动性能分析
 * 3. 在安全异常处理环境中执行Python回调
 * 4. 捕获并处理可能的异常，避免程序崩溃
 * 
 * 异常处理：
 * - 如果Python回调执行失败，会输出错误信息到标准错误流
 * - 使用std::exit(3)终止程序，避免MPI环境下的挂起问题
 * - 注意：不使用amrex::Abort()是为了防止MPI进程挂起
 * 
 * 性能分析：
 * - 使用WARPX_PROFILE宏记录回调执行时间
 * - 分析标签格式为"warpx_py_" + 回调名称
 */
void ExecutePythonCallback ( const std::string& name )
{
    // 检查指定名称的Python回调是否已注册
    if ( IsPythonCallbackInstalled(name) ) {
        
        // 启动性能分析，用于监控回调执行时间
        // 分析标签包含回调名称，便于识别不同的回调
        WARPX_PROFILE("warpx_py_" + name);
        
        try {
            // 从全局回调映射表中获取并执行对应的Python函数
            // warpx_callback_py_map存储了所有已注册的Python回调
            warpx_callback_py_map[name]();
            
        } catch (std::exception &e) {
            // 捕获Python回调抛出的异常
            std::cerr << "Python callback '" << name << "' failed!" << "\n";
            std::cerr << e.what() << "\n";
            
            // 使用exit(3)终止程序，避免MPI环境下的进程挂起
            // 注意：这里不使用amrex::Abort()，因为它可能导致MPI进程挂起
            std::exit(3);  // note: NOT amrex::Abort(), to avoid hangs with MPI

            // 未来改进说明：
            // 如果我们希望将Python异常重新抛出到管理Python解释器，
            // 需要先清除py::error_already_set中的Python错误状态。
            // 否则MPI运行会挂起，Python会保持错误状态。
            // 参考：https://pybind11.readthedocs.io/en/stable/advanced/exceptions.html#handling-unraisable-exceptions
        }
    }
}


void ClearPythonCallback ( const std::string& name )
{
    warpx_callback_py_map.erase(name);
}
