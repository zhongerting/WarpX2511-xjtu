/* Copyright 2016-2020 Andrew Myers, Ann Almgren, Axel Huebl
 *                     David Grote, Jean-Luc Vay, Remi Lehe
 *                     Revathi Jambunathan, Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "WarpX.H"

#include "Initialization/WarpXInit.H"
#include "Utils/WarpXProfilerWrapper.H"

#include <ablastr/utils/timer/Timer.H>

#include <AMReX_Print.H>

int
main (int argc, char* argv[]) {
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        WARPX_PROFILE_VAR("main()", pmain);

        auto timer = ablastr::utils::timer::Timer{};
        timer.record_start_time();

        // 初始化操作：
        // 1. 移动窗口参数；2. 相对论坐标系换算；3. 初始化场边界数组；
        // 4. 判定周期性边界条件；5. 初始化粒子边界数组；6. （RZ坐标系额外检查）
        auto& warpx = WarpX::GetInstance();
        
        // 
        warpx.InitData();
        // 
        warpx.Evolve();
        const auto is_warpx_verbose = warpx.Verbose();
        WarpX::Finalize();

        timer.record_stop_time();
        if (is_warpx_verbose) {
            amrex::Print() << "Total Time                     : "
                           << timer.get_global_duration() << '\n';
        }

        WARPX_PROFILE_VAR_STOP(pmain);
    }
    warpx::initialization::finalize_external_libraries();
}
