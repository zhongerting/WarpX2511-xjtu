/* Copyright 2019-2020 Andrew Myers, Ann Almgren, Aurore Blelly
 *                     Axel Huebl, Burlen Loring, David Grote
 *                     Glenn Richardson, Jean-Luc Vay, Luca Fedeli
 *                     Maxence Thevenet, Remi Lehe, Revathi Jambunathan
 *                     Weiqun Zhang, Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "WarpX.H"

#include "BoundaryConditions/PML.H"
#include "Diagnostics/MultiDiagnostics.H"
#include "Diagnostics/ReducedDiags/MultiReducedDiags.H"
#include "EmbeddedBoundary/Enabled.H"
#include "Fields.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#ifdef WARPX_USE_FFT
#   ifdef WARPX_DIM_RZ
#       include "FieldSolver/SpectralSolver/SpectralSolverRZ.H"
#   else
#       include "FieldSolver/SpectralSolver/SpectralSolver.H"
#   endif
#endif
#include "Parallelization/GuardCellManager.H"
#include "Particles/MultiParticleContainer.H"
#include "Fluids/MultiFluidContainer.H"
#include "Fluids/WarpXFluidContainer.H"
#include "Particles/ParticleBoundaryBuffer.H"
#include "Python/callbacks.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXUtil.H"
#include "Utils/WarpXConst.H"
#include "Utils/WarpXProfilerWrapper.H"

#include <ablastr/utils/SignalHandling.H>
#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_BLassert.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_LayoutData.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_Utility.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <array>
#include <memory>
#include <ostream>
#include <vector>

using namespace amrex;
using ablastr::utils::SignalHandling;

namespace
{
    /** Print Unused Parameter Warnings after Step 1
     *
     * Instead of waiting for a simulation to end, we already do an early "unused parameter check"
     * after step 1 to inform users early of potential issues with their simulation setup.
     */
    void checkEarlyUnusedParams ()
    {
        amrex::Print() << "\n"; // better: conditional \n based on return value
        amrex::ParmParse::QueryUnusedInputs();

        // Print the warning list right after the first step.
        amrex::Print() << ablastr::warn_manager::GetWMInstance().PrintGlobalWarnings("FIRST STEP");
    }

    void StoreCurrent (int lev, ablastr::fields::MultiFabRegister& fields)
    {
        using ablastr::fields::Direction;
        using warpx::fields::FieldType;

        for (int idim = 0; idim < 3; ++idim) {
            const auto dir = Direction{idim};
            if (fields.has(FieldType::current_store, dir,lev)) {
                MultiFab::Copy(*fields.get(FieldType::current_store, dir, lev),
                               *fields.get(FieldType::current_fp, dir, lev),
                               0, 0, 1, fields.get(FieldType::current_store, dir, lev)->nGrowVect());
            }
        }
    }

    void RestoreCurrent (int lev, ablastr::fields::MultiFabRegister& fields)
    {
        using ablastr::fields::Direction;
        using warpx::fields::FieldType;

        for (int idim = 0; idim < 3; ++idim) {
            const auto dir = Direction{idim};
            if (fields.has(FieldType::current_store, dir, lev)) {
                std::swap(
                    *fields.get(FieldType::current_fp, dir, lev),
                    *fields.get(FieldType::current_store, dir, lev)
                );
            }
        }
    }
}

void
WarpX::SynchronizeVelocityWithPosition () {
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    if (!m_is_synchronized) {
        // This assumes that the particle boundary conditions have been checked
        // so that the field gather in PushP will be correct.
        FillBoundaryE(guard_cells.ng_FieldGather);
        FillBoundaryB(guard_cells.ng_FieldGather);
        if (fft_do_time_averaging)
        {
            FillBoundaryE_avg(guard_cells.ng_FieldGather);
            FillBoundaryB_avg(guard_cells.ng_FieldGather);
        }
        UpdateAuxilaryData();
        FillBoundaryAux(guard_cells.ng_UpdateAux);
        for (int lev = 0; lev <= finest_level; ++lev) {
            mypc->PushP(
                lev,
                0.5_rt*dt[lev],
                *m_fields.get(FieldType::Efield_aux, Direction{0}, lev),
                *m_fields.get(FieldType::Efield_aux, Direction{1}, lev),
                *m_fields.get(FieldType::Efield_aux, Direction{2}, lev),
                *m_fields.get(FieldType::Bfield_aux, Direction{0}, lev),
                *m_fields.get(FieldType::Bfield_aux, Direction{1}, lev),
                *m_fields.get(FieldType::Bfield_aux, Direction{2}, lev)
            );
        }
        m_is_synchronized = true;
    }
}

void
WarpX::Evolve (int numsteps)
{
    WARPX_PROFILE_REGION("WarpX::Evolve()");
    WARPX_PROFILE("WarpX::Evolve()");

    using ablastr::fields::Direction;

    Real cur_time = t_new[0];

    // Note that the default argument is numsteps = -1
    const int numsteps_max = (numsteps < 0)?(max_step):(istep[0] + numsteps);

    // check typos in inputs after step 1
    bool early_params_checked = false;

    static Real evolve_time = 0;

    const int step_begin = istep[0];

    // 主函数循环，执行numsteps_max次迭代或直到模拟结束
    for (int step = istep[0]; step < numsteps_max && cur_time < stop_time; ++step)
    {
        WARPX_PROFILE("WarpX::Evolve::step");
        const auto evolve_time_beg_step = static_cast<Real>(amrex::second());

        // Check and clear signal flags and asynchronously broadcast them from process 0
        // 确定是否有信号被接收，如Ctrl+C中断信号
        SignalHandling::CheckSignals();

        // 诊断数据输出
        multi_diags->NewIteration();

        // 输出频率判定和设置
        bool verbose_step = (bool)verbose;
        if (verbose && m_limit_verbose_step) {

            int verbose_step_interval = 1;
            if (step<10) { verbose_step_interval = 1; }
            else if (step<100) { verbose_step_interval = 10; }
            else { verbose_step_interval = 100; }

            verbose_step = !((step+1)%verbose_step_interval);

        }

        // Start loop on time steps
        if (verbose_step) {
            amrex::Print() << "STEP " << step+1 << " starts ...\n";
        }

        // 找寻python中的beforestep的回调函数(callback)，并执行
        ExecutePythonCallback("beforestep");

        // 负载平衡函数，包含LoadBalance和ResetCosts
        // 用于保证每个MPI进程的计算负载均衡
        CheckLoadBalance(step);

        // Update timestep for electrostatic solver if a constant dt is not provided
        // This first synchronizes the position and velocity before setting the new timestep
        // 动态调整时间步长（静电求解器+无设置步长+模拟步骤符合预设更新时间间隔）
        if (electromagnetic_solver_id == ElectromagneticSolverAlgo::None &&
            !m_const_dt.has_value() && m_dt_update_interval.contains(step+1)) {
            if (verbose_step) {
                amrex::Print() << Utils::TextMsg::Info("updating timestep");
            }
            SynchronizeVelocityWithPosition();
            UpdateDtFromParticleSpeeds();
        }

        // If position and velocity are synchronized, push velocity backward one half step
        // 若显式求解，则针对电磁场边界条件进行处理
        if (evolve_scheme == EvolveScheme::Explicit)
        {
            ExplicitFillBoundaryEBUpdateAux();
        }

        // If needed, deposit the initial ion charge and current densities that
        // will be used to update the E-field in Ohm's law.

        // 首步计算，且电磁求解器采用混合PIC算法
        // 额外进行一次RhoJ和B的初始化计算（仅首步）
        if (step == step_begin &&
            electromagnetic_solver_id == ElectromagneticSolverAlgo::HybridPIC
        ) {
            HybridPICInitializeRhoJandB();
        }

        // multi-physics: field ionization
        // 多物理场扩展计算——场电离（需要强大的外部电场）
        doFieldIonization();

#ifdef WARPX_QED
        // multi-physics: QED effects
        doQEDEvents();
        mypc->doQEDSchwinger();
#endif

        // perform particle injection
        // 找寻python中的particleinjection的回调函数(callback)，并执行
        // 用于粒子的注入
        ExecutePythonCallback("particleinjection");

        // perform collisions and advance fields and particles by one time step
        // 多物理场扩展计算——碰撞和推进粒子
        // 参数：当前时间，时间步长，计算步数

        // 最核心的步骤，用于推动粒子运动和电流沉积以及碰撞。
        OneStep(cur_time, dt[0], step);

        // Resample particles
        // +1 is necessary here because value of step seen by user (first step is 1) is different than
        // value of step in code (first step is 0)
        // 对粒子进行重采样，更新粒子位置和属性
        // 这是为了确保粒子在计算中不会被跳过或重复计算
        mypc->doResampling(Geom(), istep[0]+1, verbose_step);

        // 显式求解需要
        // 应用镜像边界条件，确保粒子在边界处正确处理
        if (evolve_scheme == EvolveScheme::Explicit) {
            applyMirrors(cur_time);
            // E : guard cells are NOT up-to-date
            // B : guard cells are NOT up-to-date
        }

        // 更新迭代步数
        // 这是为了记录当前模拟的迭代次数，用于后续分析和控制
        for (int lev = 0; lev <= max_level; ++lev) {
            ++istep[lev];
        }

        // 更新当前物理时间
        // 这是为了记录当前模拟的物理时间，用于后续分析和控制
        cur_time += dt[0];

        // 对粒子进行伽利略变换边界处理
        // 这是为了确保粒子在计算中不会被跳过或重复计算
        ShiftGalileanBoundary();

        // sync up time
        // 同步物理时间
        // 这是为了确保所有MPI进程的物理时间保持一致，避免时间上的不匹配
        for (int i = 0; i <= max_level; ++i) {
            t_old[i] = t_new[i];
            t_new[i] = cur_time;
        }
        multi_diags->FilterComputePackFlush( step, false, true );

        const bool move_j = m_is_synchronized;
        // If m_is_synchronized we need to shift j too so that next step we can evolve E by dt/2.
        // We might need to move j because we are going to make a plotfile.
        const int num_moved = MoveWindow(step+1, move_j);

        // Update the accelerator lattice element finder if the window has moved,
        // from either a moving window or a boosted frame
        if (num_moved != 0 || gamma_boost > 1) {
            for (int lev = 0; lev <= finest_level; ++lev) {
                m_accelerator_lattice[lev]->UpdateElementFinder(lev, gett_new());
            }
        }

        // 处理粒子在边界处的行为，包括粒子删除、反射或重新分布等操作
        // 该函数确保粒子不会超出计算域边界，维护粒子分布的合理性
        HandleParticlesAtBoundaries(step, cur_time, num_moved);

        // 静电或混合PIC求解器的场求解步骤
        // 当使用静电求解器或混合PIC求解器时，需要在此阶段求解电磁场
        if( electrostatic_solver_id != ElectrostaticSolverAlgo::None ||
            electromagnetic_solver_id == ElectromagneticSolverAlgo::HybridPIC )
        {
            // 执行场求解前的Python回调函数，允许用户自定义操作
            ExecutePythonCallback("beforeEsolve");

            if (electrostatic_solver_id != ElectrostaticSolverAlgo::None) {
                // 静电求解器处理流程：
                // 对每个粒子物种：沉积电荷并将相关的空间电荷电场和磁场添加到网格上
                // 此操作在PIC循环结束时执行（即在Redistribute之后、下次位置推进之前）
                // 这样可以避免粒子越界沉积，并确保输出中的场处于正确的时间点
                bool const reset_fields = true;
                // 计算空间电荷场，重置现有场并求解泊松方程得到电场
                ComputeSpaceChargeField( reset_fields );
                
                if (electrostatic_solver_id == ElectrostaticSolverAlgo::LabFrameElectroMagnetostatic) {
                    // 对于电磁静态求解器：调用静磁求解器求解矢量势A并计算B场
                    // 忽略A随时间变化对E场的贡献，目前仅在实验室坐标系中计算
                    ComputeMagnetostaticField();
                }
                
                // 由于上面重置了场，需要重新添加外部场
                // 这样净场就是场解和外部场的总和
                for (int lev = 0; lev <= max_level; ++lev) {
                    AddExternalFields(lev);
                }
            } else if (electromagnetic_solver_id == ElectromagneticSolverAlgo::HybridPIC) {
                // 混合PIC情况：
                // 粒子当前处于p^{n+1/2}和x^{n+1}状态
                // 根据混合PIC方案（欧姆定律和安培定律）更新场
                HybridPICEvolveFields();
            }
            
            // 执行场求解后的Python回调函数
            ExecutePythonCallback("afterEsolve");
        }


        bool const do_diagnostic = (multi_diags->DoComputeAndPack(step) || reduced_diags->DoDiags(step));
        bool const end_of_step_loop = (step == numsteps_max - 1) || (cur_time + dt[0] >= stop_time - 1.e-3*dt[0]);
        if (synchronize_velocity_for_diagnostics &&
            (do_diagnostic || end_of_step_loop)) {
            // When the diagnostics require synchronization, push p by 0.5*dt to synchronize.
            // Note that this will be undone at the start of the next step by the half v-push
            // backwards.
            SynchronizeVelocityWithPosition();
        }

        // afterstep callback runs with the updated global time. It is included
        // in the evolve timing.
        ExecutePythonCallback("afterstep");

        /// reduced diags
        if (reduced_diags->m_plot_rd != 0)
        {
            reduced_diags->LoadBalance();
            reduced_diags->ComputeDiags(step);
            reduced_diags->WriteToFile(step);
        }
        multi_diags->FilterComputePackFlush( step );

        // execute afterdiagnostic callbacks
        ExecutePythonCallback("afterdiagnostics");

        // inputs: unused parameters (e.g. typos) check after step 1 has finished
        if (!early_params_checked) {
            ::checkEarlyUnusedParams();
            early_params_checked = true;
        }

        // create ending time stamp for calculating elapsed time each iteration
        const auto evolve_time_end_step = static_cast<Real>(amrex::second());
        evolve_time += evolve_time_end_step - evolve_time_beg_step;

        HandleSignals();

        if (verbose_step) {
            amrex::Print()<< "STEP " << step+1 << " ends." << " TIME = " << cur_time
                        << " DT = " << dt[0] << "\n";
            amrex::Print()<< "Evolve time = " << evolve_time
                      << " s; This step = " << evolve_time_end_step-evolve_time_beg_step
                      << " s; Avg. per step = " << evolve_time/(step-step_begin+1) << " s\n\n";
        }

        if (checkStopSimulation(cur_time)) {
            break;
        }
    } // End loop on time steps

    // This if statement is needed for PICMI, which allows the Evolve routine to be
    // called multiple times, otherwise diagnostics will be done at every call,
    // regardless of the diagnostic period parameter provided in the inputs.
    bool const final_time_step = (istep[0] == max_step)
                                || (cur_time >= stop_time - 1.e-3*dt[0]
                                 && cur_time < stop_time + dt[0]);
    if (final_time_step || m_exit_loop_due_to_interrupt_signal) {
        multi_diags->FilterComputePackFlushLastTimestep( istep[0] );
        if (m_exit_loop_due_to_interrupt_signal) { ExecutePythonCallback("onbreaksignal"); }
    }

    amrex::Print() <<
        ablastr::warn_manager::GetWMInstance().PrintGlobalWarnings("THE END");
}

void WarpX::OneStep (
    amrex::Real a_cur_time,
    amrex::Real a_dt,
    int a_step
)
{
    WARPX_PROFILE("WarpX::OneStep()");

    // implicit solver
    // 隐式求解方法粒子推进过程：
    // 1. 执行beforecollisions回调函数，用于在碰撞前进行自定义操作
    // 2. 调用mypc->doCollisions()进行碰撞计算
    // 3. 执行aftercollisions回调函数，用于在碰撞后进行自定义操作
    // 4. 调用m_implicit_solver->OneStep()进行隐式求解方法的粒子推进
    if (m_implicit_solver) {
        // perform particle collisions
        ExecutePythonCallback("beforecollisions");
        mypc->doCollisions(a_step, a_cur_time, a_dt);
        ExecutePythonCallback("aftercollisions");

        // advance fields and particles by one time step
        m_implicit_solver->OneStep(a_cur_time, a_dt, a_step);
    }
    // explicit solver
    else {
        // electrostatic solver or hybrid solver
        // 若采用静电求解器或混合PIC求解器
        if (electromagnetic_solver_id == ElectromagneticSolverAlgo::None ||
            electromagnetic_solver_id == ElectromagneticSolverAlgo::HybridPIC) {
            // with collisions placed in the middle of the position push and after the momentum push
            // 判定碰撞发生时间，在位置推进和动量推进之间进行。
            // m_collisions_split_position_push参数用于表征显式求解并且没有 embedded 边界
            if (m_collisions_split_position_push) {
                // push particles (half position and full momentum)
                // 推进粒子到新的时间步位置，并将粒子运动产生的电流密度沉积到网格上。
                PushParticlesandDeposit(
                    a_cur_time,
                    /*skip_current=*/true,
                    PositionPushType::FirstHalf,
                    MomentumPushType::Full
                );

                // communicate particle data
                // 传输粒子数据，确保不同网格层级上的粒子数据同步。
                mypc->Redistribute();

                // 至此，进行了一次位置推进，并计算了由于粒子运动引起的电流密度

                // perform particle collisions
                // 执行碰撞前回调函数，用于在碰撞前进行自定义操作
                ExecutePythonCallback("beforecollisions");
                // 执行粒子碰撞计算
                mypc->doCollisions(a_step, a_cur_time, a_dt);
                // 执行碰撞后回调函数，用于在碰撞后进行自定义操作
                ExecutePythonCallback("aftercollisions");

                // push particles (half position)
                // 推进粒子到新的时间步位置，并将粒子运动产生的电流密度沉积到网格上。
                PushParticlesandDeposit(
                    a_cur_time,
                    /*skip_current=*/true,
                    PositionPushType::SecondHalf,
                    MomentumPushType::None
                );
            }
            // with collisions placed before the position and momentum push, or without collisions
            // 隐式求解
            else {
                // perform particle collisions
                ExecutePythonCallback("beforecollisions");
                mypc->doCollisions(a_step, a_cur_time, a_dt);
                ExecutePythonCallback("aftercollisions");

                // push particles (half position)
                PushParticlesandDeposit(
                    a_cur_time,
                    /*skip_current=*/true,
                    PositionPushType::Full,
                    MomentumPushType::Full
                );
            }
        }
        // electromagnetic solver
        // 电磁求解器
        else {
            // perform particle collisions
            ExecutePythonCallback("beforecollisions");
            mypc->doCollisions(a_step, a_cur_time, a_dt);
            ExecutePythonCallback("aftercollisions");

            // without mesh refinement
            if (finest_level == 0) {
                // standard PIC loop
                if (!m_JRhom) {
                    OneStep_nosub(a_cur_time);
                }
                // JRhom PIC loop
                else {
                    OneStep_JRhom(a_cur_time);
                }
            }
            // with mesh refinement
            else {
                // without subcycling
                if (!m_do_subcycling) {
                    OneStep_nosub(a_cur_time);
                }
                // with subcycling
                else {
                    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                        finest_level == 1,
                        "Subcycling not implemented with more than 1 mesh refinement level"
                    );
                    OneStep_sub1(a_cur_time);
                }
            }
        }
    }
}

/**
 * \brief Perform one PIC iteration, without subcycling
 * i.e. all levels/patches use the same timestep (that of the finest level)
 * for the field advance and particle pusher.
 */
void
WarpX::OneStep_nosub (Real cur_time)
{
    WARPX_PROFILE("WarpX::OneStep_nosub()");

    // Push particle from x^{n} to x^{n+1}
    //               from p^{n-1/2} to p^{n+1/2}
    // Deposit current j^{n+1/2}
    // Deposit charge density rho^{n}

    ExecutePythonCallback("particlescraper");
    ExecutePythonCallback("beforedeposition");

    PushParticlesandDeposit(cur_time);

    ExecutePythonCallback("afterdeposition");

    // Synchronize J and rho:
    // filter (if used), exchange guard cells, interpolate across MR levels
    // and apply boundary conditions
    SyncCurrentAndRho();

    // At this point, J is up-to-date inside the domain, and E and B are
    // up-to-date including enough guard cells for first step of the field
    // solve.

    // For extended PML: copy J from regular grid to PML, and damp J in PML
    if (do_pml && pml_has_particles) { CopyJPML(); }
    if (do_pml && do_pml_j_damping) { DampJPML(); }

    ExecutePythonCallback("beforeEsolve");

    // Push E and B from {n} to {n+1}
    // (And update guard cells immediately afterwards)
    if (WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD) {
        if (use_hybrid_QED)
        {
            WarpX::Hybrid_QED_Push(dt);
            FillBoundaryE(guard_cells.ng_alloc_EB);
        }
        PushPSATD(cur_time);

        if (do_pml) {
            DampPML();
        }

        if (use_hybrid_QED) {
            FillBoundaryE(guard_cells.ng_alloc_EB);
            FillBoundaryB(guard_cells.ng_alloc_EB, WarpX::sync_nodal_points);
            WarpX::Hybrid_QED_Push(dt);
            FillBoundaryE(guard_cells.ng_afterPushPSATD, WarpX::sync_nodal_points);
        }
        else {
            FillBoundaryE(guard_cells.ng_afterPushPSATD, WarpX::sync_nodal_points);
            FillBoundaryB(guard_cells.ng_afterPushPSATD, WarpX::sync_nodal_points);
            if (WarpX::do_dive_cleaning || WarpX::do_pml_dive_cleaning) {
                FillBoundaryF(guard_cells.ng_alloc_F, WarpX::sync_nodal_points);
            }
            if (WarpX::do_divb_cleaning || WarpX::do_pml_divb_cleaning) {
                FillBoundaryG(guard_cells.ng_alloc_G, WarpX::sync_nodal_points);
            }
        }
    } else {
        EvolveF(0.5_rt * dt[0], /*rho_comp=*/0);
        EvolveG(0.5_rt * dt[0]);
        FillBoundaryF(guard_cells.ng_FieldSolverF);
        FillBoundaryG(guard_cells.ng_FieldSolverG);

        EvolveB(0.5_rt * dt[0], SubcyclingHalf::FirstHalf, cur_time); // We now have B^{n+1/2}
        FillBoundaryB(guard_cells.ng_FieldSolver, WarpX::sync_nodal_points);

        if (m_em_solver_medium == MediumForEM::Vacuum) {
            // vacuum medium
            EvolveE(dt[0], cur_time); // We now have E^{n+1}
        } else if (m_em_solver_medium == MediumForEM::Macroscopic) {
            // macroscopic medium
            MacroscopicEvolveE(dt[0], cur_time); // We now have E^{n+1}
        } else {
            WARPX_ABORT_WITH_MESSAGE("Medium for EM is unknown");
        }
        FillBoundaryE(guard_cells.ng_FieldSolver, WarpX::sync_nodal_points);

        EvolveF(0.5_rt * dt[0], /*rho_comp=*/1);
        EvolveG(0.5_rt * dt[0]);
        EvolveB(0.5_rt * dt[0], SubcyclingHalf::SecondHalf, cur_time + 0.5_rt * dt[0]); // We now have B^{n+1}

        if (do_pml) {
            DampPML();
            FillBoundaryE(guard_cells.ng_MovingWindow, WarpX::sync_nodal_points);
            FillBoundaryB(guard_cells.ng_MovingWindow, WarpX::sync_nodal_points);
            FillBoundaryF(guard_cells.ng_MovingWindow, WarpX::sync_nodal_points);
            FillBoundaryG(guard_cells.ng_MovingWindow, WarpX::sync_nodal_points);
        }

        // E and B are up-to-date in the domain, but all guard cells are
        // outdated.
        if (m_safe_guard_cells) {
            FillBoundaryB(guard_cells.ng_alloc_EB);
        }
    } // !PSATD

    ExecutePythonCallback("afterEsolve");
}

bool WarpX::checkStopSimulation (amrex::Real cur_time)
{
    m_exit_loop_due_to_interrupt_signal = SignalHandling::TestAndResetActionRequestFlag(SignalHandling::SIGNAL_REQUESTS_BREAK);
    return (cur_time >= stop_time - 1.e-3*dt[0])  ||
        m_exit_loop_due_to_interrupt_signal;
}

void WarpX::ExplicitFillBoundaryEBUpdateAux ()
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(evolve_scheme == EvolveScheme::Explicit,
        "Cannot call WarpX::ExplicitFillBoundaryEBUpdateAux without Explicit evolve scheme set!");

    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    // At the beginning, we have B^{n} and E^{n}.
    // Particles have p^{n} and x^{n}.
    // m_is_synchronized is true.

    if (m_is_synchronized) {
        // Not called at each iteration, so exchange all guard cells
        FillBoundaryE(guard_cells.ng_alloc_EB);
        FillBoundaryB(guard_cells.ng_alloc_EB);

        UpdateAuxilaryData();
        FillBoundaryAux(guard_cells.ng_UpdateAux);
        // on first step, push p by -0.5*dt
        for (int lev = 0; lev <= finest_level; ++lev)
        {
            mypc->PushP(
                lev,
                -0.5_rt*dt[lev],
                *m_fields.get(FieldType::Efield_aux, Direction{0}, lev),
                *m_fields.get(FieldType::Efield_aux, Direction{1}, lev),
                *m_fields.get(FieldType::Efield_aux, Direction{2}, lev),
                *m_fields.get(FieldType::Bfield_aux, Direction{0}, lev),
                *m_fields.get(FieldType::Bfield_aux, Direction{1}, lev),
                *m_fields.get(FieldType::Bfield_aux, Direction{2}, lev)
            );
        }
        m_is_synchronized = false;

    } else {
        // Beyond one step, we have E^{n} and B^{n}.
        // Particles have p^{n-1/2} and x^{n}.
        // E and B: enough guard cells to update Aux or call Field Gather in fp and cp
        // Need to update Aux on lower levels, to interpolate to higher levels.

        // E and B are up-to-date inside the domain only
        FillBoundaryE(guard_cells.ng_FieldGather);
        FillBoundaryB(guard_cells.ng_FieldGather);
        if (electrostatic_solver_id == ElectrostaticSolverAlgo::None) {
            if (fft_do_time_averaging)
            {
                FillBoundaryE_avg(guard_cells.ng_FieldGather);
                FillBoundaryB_avg(guard_cells.ng_FieldGather);
            }
            // TODO Remove call to FillBoundaryAux before UpdateAuxilaryData?
            if (WarpX::electromagnetic_solver_id != ElectromagneticSolverAlgo::PSATD) {
                FillBoundaryAux(guard_cells.ng_UpdateAux);
            }
        }
        UpdateAuxilaryData();
        FillBoundaryAux(guard_cells.ng_UpdateAux);
    }
}

/**
 * @brief 处理粒子在边界处的行为，包括连续注入、边界条件应用、粒子重分布和EB壁面交互
 * 
 * 该函数是WarpX主时间推进循环中处理粒子边界行为的核心函数，负责：
 * 1. 处理连续通量注入（如激光等离子体加速中的连续注入）
 * 2. 应用粒子边界条件（吸收、反射、周期性等）
 * 3. 管理粒子在网格间的重分布
 * 4. 处理粒子与嵌入式边界（EB）的交互
 * 5. 根据需要对粒子进行重新排序以优化性能
 * 
 * @param step 当前时间步数
 * @param cur_time 当前物理时间
 * @param num_moved 本时间步移动的粒子数量（用于优化重分布）
 */
void WarpX::HandleParticlesAtBoundaries (int step, amrex::Real cur_time, int num_moved)
{
    // 处理连续通量注入：在指定边界处持续注入粒子
    // 常用于激光等离子体加速等需要持续粒子注入的场景
    mypc->ContinuousFluxInjection(cur_time, dt[0]);

    // 应用粒子边界条件：处理到达边界的粒子
    // 包括吸收、反射、周期性边界等不同处理方式
    mypc->ApplyBoundaryConditions();
    
    // 从域边界收集粒子到缓冲区：将到达计算域边界的粒子收集到专门的缓冲区
    m_particle_boundary_buffer->gatherParticlesFromDomainBoundaries(*mypc);

    // 非麦克斯韦求解器（静电或混合PIC）：粒子可以任意移动多个网格单元
    
    if( electromagnetic_solver_id == ElectromagneticSolverAlgo::None ||
        electromagnetic_solver_id == ElectromagneticSolverAlgo::HybridPIC )
    {
        // 执行全局粒子重分布：将所有粒子重新分配到正确的网格位置
        mypc->Redistribute();
    }
    else
    {
        // 电磁求解器：由于CFL条件限制，粒子每时间步只能移动1-2个网格单元
        // 隐式方案允许额外的网格跨越，由particle_max_grid_crossings参数控制
        if (max_level == 0) {
            // 无网格细化时，计算局部重分布所需的幽灵层厚度
            int num_redistribute_ghost = num_moved;
            
            if ((m_v_galilean[0]!=0) or (m_v_galilean[1]!=0) or (m_v_galilean[2]!=0)) {
                // 伽利略算法：粒子可以比最大数量多移动一个额外的网格单元
                // 因为网格本身也在以伽利略速度移动
                num_redistribute_ghost += particle_max_grid_crossings + 1;
            } else {
                // 标准算法：粒子最多可以移动particle_max_grid_crossings个网格单元
                num_redistribute_ghost += particle_max_grid_crossings;
            }
            
            // 执行局部粒子重分布：只重分布指定幽灵层厚度内的粒子
            // 比全局重分布更高效，特别适用于粒子移动范围有限的情况
            mypc->RedistributeLocal(num_redistribute_ghost);
        }
        else {
            // 有网格细化时，执行全局粒子重分布
            mypc->Redistribute();
        }
    }

    // 处理粒子与EB（嵌入式边界）壁面的交互（如果存在EB）
    if (EB::enabled()) {
        using warpx::fields::FieldType;
        
        // 在EB壁面处刮削粒子：移除或反射与EB壁面接触的粒子
        mypc->ScrapeParticlesAtEB(m_fields.get_mr_levels(FieldType::distance_to_eb, finest_level));
        
        // 从嵌入式边界收集粒子：将到达EB边界的粒子收集到缓冲区
        m_particle_boundary_buffer->gatherParticlesFromEmbeddedBoundaries(
            *mypc, m_fields.get_mr_levels(FieldType::distance_to_eb, finest_level));
        
        // 删除无效粒子：移除因EB交互而标记为无效的粒子
        mypc->deleteInvalidParticles();
    }

    // 根据指定的时间间隔对粒子进行重新排序
    // 重新排序可以改善缓存局部性，提高沉积和场收集的性能
    if (sort_intervals.contains(step+1)) {
        if (verbose) {
            amrex::Print() << Utils::TextMsg::Info("re-sorting particles");
        }
        
        // 按网格单元箱对粒子进行排序
        mypc->SortParticlesByBin(
            sort_bin_size, m_sort_particles_for_deposition, m_sort_idx_type);
    }
}


void WarpX::SyncCurrentAndRho ()
{
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    if (electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD)
    {
        if (fft_periodic_single_box)
        {
            // With periodic single box, synchronize J and rho here,
            // even with current correction or Vay deposition
            std::string const current_fp_string = (current_deposition_algo == CurrentDepositionAlgo::Vay)
                ? "current_fp_vay" : "current_fp";
            // TODO Replace current_cp with current_cp_vay once Vay deposition is implemented with MR

            SyncCurrent(current_fp_string);
            SyncRho();

        }
        else // no periodic single box
        {
            // Without periodic single box, synchronize J and rho here,
            // except with current correction or Vay deposition:
            // in these cases, synchronize later (in WarpX::PushPSATD)
            if (!current_correction &&
                current_deposition_algo != CurrentDepositionAlgo::Vay)
            {
                SyncCurrent("current_fp");
                SyncRho();
            }

            if (current_deposition_algo == CurrentDepositionAlgo::Vay)
            {
                // TODO This works only without mesh refinement
                const int lev = 0;
                if (use_filter) {
                    ApplyFilterMF(m_fields.get_mr_levels_alldirs(FieldType::current_fp_vay, finest_level), lev);
                }
            }
        }
    }
    else // FDTD
    {
        SyncCurrent("current_fp");
        SyncRho();
    }

    // Reflect charge and current density over PEC boundaries, if needed.
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        if (m_fields.has(FieldType::rho_fp, lev)) {
            ApplyRhofieldBoundary(lev, m_fields.get(FieldType::rho_fp,lev), PatchType::fine);
        }
        ApplyJfieldBoundary(lev,
            m_fields.get(FieldType::current_fp, Direction{0}, lev),
            m_fields.get(FieldType::current_fp, Direction{1}, lev),
            m_fields.get(FieldType::current_fp, Direction{2}, lev),
            PatchType::fine);
        if (lev > 0) {
            if (m_fields.has(FieldType::rho_cp, lev)) {
                ApplyRhofieldBoundary(lev, m_fields.get(FieldType::rho_cp,lev), PatchType::coarse);
            }
            ApplyJfieldBoundary(lev,
                m_fields.get(FieldType::current_cp, Direction{0}, lev),
                m_fields.get(FieldType::current_cp, Direction{1}, lev),
                m_fields.get(FieldType::current_cp, Direction{2}, lev),
                PatchType::coarse);
        }
    }
}

void
WarpX::OneStep_JRhom (const amrex::Real cur_time)
{
#ifdef WARPX_USE_FFT

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD,
        "JRhom algorithm not implemented with the FDTD solver"
    );

    using warpx::fields::FieldType;

    bool const skip_lev0_coarse_patch = true;

    const int rho_mid = spectral_solver_fp[0]->m_spectral_index.rho_mid;
    const int rho_new = spectral_solver_fp[0]->m_spectral_index.rho_new;

    // Push particle from x^{n} to x^{n+1}
    //               from p^{n-1/2} to p^{n+1/2}
    const bool skip_deposition = true;
    PushParticlesandDeposit(cur_time, skip_deposition);

    // Initialize PSATD-JRhom loop:

    // 1) Prepare E,B,F,G fields in spectral space
    PSATDForwardTransformEB();
    if (WarpX::do_dive_cleaning) { PSATDForwardTransformF(); }
    if (WarpX::do_divb_cleaning) { PSATDForwardTransformG(); }

    // 2) Set the averaged fields to zero
    if (WarpX::fft_do_time_averaging) { PSATDEraseAverageFields(); }

    // 3) Deposit rho (in rho_new, since it will be moved during the loop)
    //    (after checking that pointer to rho_fp on MR level 0 is not null)
    if (m_fields.has(FieldType::rho_fp, 0) && time_dependency_rho != TimeDependencyRho::Constant)
    {
        ablastr::fields::MultiLevelScalarField const rho_fp = m_fields.get_mr_levels(FieldType::rho_fp, finest_level);

        std::string const rho_fp_string = "rho_fp";
        std::string const rho_cp_string = "rho_cp";

        // Deposit rho at relative time -dt
        // (dt[0] denotes the time step on mesh refinement level 0)
        mypc->DepositCharge(rho_fp, -dt[0]);
        // Filter, exchange boundary, and interpolate across levels
        SyncRho();
        // Forward FFT of rho
        PSATDForwardTransformRho(rho_fp_string, rho_cp_string, 0, rho_new);
    }

    // 4) Deposit J at relative time -dt with time step dt
    //    (dt[0] denotes the time step on mesh refinement level 0)
    if (time_dependency_J != TimeDependencyJ::Constant)
    {
        std::string const current_string = (do_current_centering) ? "current_fp_nodal" : "current_fp";
        mypc->DepositCurrent( m_fields.get_mr_levels_alldirs(current_string, finest_level), dt[0], -dt[0]);
        // Synchronize J: filter, exchange boundary, and interpolate across levels.
        // With current centering, the nodal current is deposited in 'current',
        // namely 'current_fp_nodal': SyncCurrent stores the result of its centering
        // into 'current_fp' and then performs both filtering, if used, and exchange
        // of guard cells.
        SyncCurrent("current_fp");
        // Forward FFT of J
        PSATDForwardTransformJ("current_fp", "current_cp");
    }

    // Number of depositions for multi-J scheme
    const int n_deposit = WarpX::m_JRhom_subintervals;
    // Time sub-step for each multi-J deposition
    const amrex::Real sub_dt = dt[0] / static_cast<amrex::Real>(n_deposit);
    // Whether to perform PSATD-JRhom depositions on a time interval that spans
    // one or two full time steps (from n*dt to (n+1)*dt, or from n*dt to (n+2)*dt)
    const int n_loop = (WarpX::fft_do_time_averaging) ? 2*n_deposit : n_deposit;

    // Loop over PSATD-JRhom depositions
    for (int i_deposit = 0; i_deposit < n_loop; i_deposit++)
    {
        // Move J from new to old if J is linear or quadratic in time
        if (time_dependency_J != TimeDependencyJ::Constant) { PSATDMoveJNewToJOld(); }

        const amrex::Real t_deposit_current = (time_dependency_J == TimeDependencyJ::Linear) ?
            (i_deposit-n_deposit+1)*sub_dt : (i_deposit-n_deposit+0.5_rt)*sub_dt;

        const amrex::Real t_deposit_charge = (time_dependency_rho == TimeDependencyRho::Linear) ?
            (i_deposit-n_deposit+1)*sub_dt : (i_deposit-n_deposit+0.5_rt)*sub_dt;

        // Deposit new J at relative time t_deposit_current with time step dt
        // (dt[0] denotes the time step on mesh refinement level 0)
        std::string const current_string = (do_current_centering) ? "current_fp_nodal" : "current_fp";
        mypc->DepositCurrent( m_fields.get_mr_levels_alldirs(current_string, finest_level), dt[0], t_deposit_current);
        // Synchronize J: filter, exchange boundary, and interpolate across levels.
        // With current centering, the nodal current is deposited in 'current',
        // namely 'current_fp_nodal': SyncCurrent stores the result of its centering
        // into 'current_fp' and then performs both filtering, if used, and exchange
        // of guard cells.
        SyncCurrent("current_fp");
        // Forward FFT of J
        PSATDForwardTransformJ("current_fp", "current_cp");

        if (time_dependency_J == TimeDependencyJ::Quadratic)
        {
            PSATDMoveJNewToJMid();
            mypc->DepositCurrent( m_fields.get_mr_levels_alldirs(current_string, finest_level),  dt[0], t_deposit_current + 0.5_rt*sub_dt);
            SyncCurrent("current_fp");
            PSATDForwardTransformJ("current_fp", "current_cp");
        }

        // Deposit new rho
        // (after checking that pointer to rho_fp on MR level 0 is not null)
        if (m_fields.has(FieldType::rho_fp, 0))
        {
            ablastr::fields::MultiLevelScalarField const rho_fp = m_fields.get_mr_levels(FieldType::rho_fp, finest_level);

            std::string const rho_fp_string = "rho_fp";
            std::string const rho_cp_string = "rho_cp";

            // Move rho from new to old if rho is linear in time
            if (time_dependency_rho != TimeDependencyRho::Constant) { PSATDMoveRhoNewToRhoOld(); }

            // Deposit rho at relative time t_deposit_charge
            mypc->DepositCharge(rho_fp, t_deposit_charge);
            // Filter, exchange boundary, and interpolate across levels
            SyncRho();
            // Forward FFT of rho
            const int rho_idx = (time_dependency_rho != TimeDependencyRho::Constant) ? rho_new : rho_mid;
            PSATDForwardTransformRho(rho_fp_string, rho_cp_string, 0, rho_idx);

            if (time_dependency_rho == TimeDependencyRho::Quadratic)
            {
                PSATDMoveRhoNewToRhoMid();
                mypc->DepositCharge(rho_fp, t_deposit_charge + 0.5_rt*sub_dt);
                SyncRho();
                PSATDForwardTransformRho(rho_fp_string, rho_cp_string, 0, rho_new);
            }
        }

        if (WarpX::current_correction)
        {
            WARPX_ABORT_WITH_MESSAGE(
                "Current correction not implemented for PSATD-JRhom algorithm.");
        }

        // Advance E,B,F,G fields in time and update the average fields
        PSATDPushSpectralFields();

        // Transform non-average fields E,B,F,G after n_deposit pushes
        // (the relative time reached here coincides with an integer full time step)
        if (i_deposit == n_deposit-1)
        {
            PSATDBackwardTransformEB();
            if (WarpX::do_dive_cleaning) { PSATDBackwardTransformF(); }
            if (WarpX::do_divb_cleaning) { PSATDBackwardTransformG(); }
        }
    }

    // Transform fields back to real space
    if (WarpX::fft_do_time_averaging)
    {
        // We summed the integral of the field over 2*dt
        PSATDScaleAverageFields(1._rt / (2._rt*dt[0]));
        PSATDBackwardTransformEBavg(
            m_fields.get_mr_levels_alldirs(FieldType::Efield_avg_fp, finest_level),
            m_fields.get_mr_levels_alldirs(FieldType::Bfield_avg_fp, finest_level),
            m_fields.get_mr_levels_alldirs(FieldType::Efield_avg_cp, finest_level, skip_lev0_coarse_patch),
            m_fields.get_mr_levels_alldirs(FieldType::Bfield_avg_cp, finest_level, skip_lev0_coarse_patch)
        );
    }

    // Evolve fields in PML
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        if (do_pml && pml[lev]->ok())
        {
            pml[lev]->PushPSATD(m_fields, lev);
        }
        ApplyEfieldBoundary(lev, PatchType::fine, cur_time + dt[0]);
        if (lev > 0) { ApplyEfieldBoundary(lev, PatchType::coarse, cur_time + dt[0]); }
        ApplyBfieldBoundary(lev, PatchType::fine, SubcyclingHalf::FirstHalf, cur_time + dt[0]);
        if (lev > 0) { ApplyBfieldBoundary(lev, PatchType::coarse, SubcyclingHalf::FirstHalf, cur_time + dt[0]); }
    }

    // Damp fields in PML before exchanging guard cells
    if (do_pml)
    {
        DampPML();
    }

    // Exchange guard cells and synchronize nodal points
    FillBoundaryE(guard_cells.ng_alloc_EB, WarpX::sync_nodal_points);
    FillBoundaryB(guard_cells.ng_alloc_EB, WarpX::sync_nodal_points);
    if (WarpX::do_dive_cleaning || WarpX::do_pml_dive_cleaning) {
        FillBoundaryF(guard_cells.ng_alloc_F, WarpX::sync_nodal_points);
    }
    if (WarpX::do_divb_cleaning || WarpX::do_pml_divb_cleaning) {
        FillBoundaryG(guard_cells.ng_alloc_G, WarpX::sync_nodal_points);
    }

#else
    amrex::ignore_unused(cur_time);
    WARPX_ABORT_WITH_MESSAGE(
        "JRhom algorithm not implemented with the FDTD solver");
#endif // WARPX_USE_FFT
}

/**
 *  \brief Perform one PIC iteration, with subcycling
 *  i.e. The fine patch uses a smaller timestep (and steps more often)
 *  than the coarse patch, for the field advance and particle pusher.
 *
 * This version of subcycling only works for 2 levels and with a refinement
 * ratio of 2.
 * The particles and fields of the fine patch are pushed twice
 * (with dt[coarse]/2) in this routine.
 * The particles of the coarse patch and mother grid are pushed only once
 * (with dt[coarse]). The fields on the coarse patch and mother grid
 * are pushed in a way which is equivalent to pushing once only, with
 * a current which is the average of the coarse + fine current at the 2
 * steps of the fine grid.
 *
 */
void
WarpX::OneStep_sub1 (Real cur_time)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        electrostatic_solver_id == ElectrostaticSolverAlgo::None,
        "Electrostatic solver cannot be used with sub-cycling."
    );

    // TODO: we could save some charge depositions

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(finest_level == 1, "Must have exactly two levels");
    const int fine_lev = 1;
    const int coarse_lev = 0;

    using warpx::fields::FieldType;

    bool const skip_lev0_coarse_patch = true;

    // i) Push particles and fields on the fine patch (first fine step)
    PushParticlesandDeposit(fine_lev, cur_time, SubcyclingHalf::FirstHalf);
    RestrictCurrentFromFineToCoarsePatch(
        m_fields.get_mr_levels_alldirs(FieldType::current_fp, finest_level),
        m_fields.get_mr_levels_alldirs(FieldType::current_cp, finest_level, skip_lev0_coarse_patch), fine_lev);
    RestrictRhoFromFineToCoarsePatch(fine_lev);
    if (use_filter) {
        ApplyFilterMF( m_fields.get_mr_levels_alldirs(FieldType::current_fp, finest_level), fine_lev);
    }
    SumBoundaryJ(
        m_fields.get_mr_levels_alldirs(FieldType::current_fp, finest_level),
        fine_lev, Geom(fine_lev).periodicity());

    if (m_fields.has(FieldType::rho_fp, finest_level) &&
        m_fields.has(FieldType::rho_cp, finest_level)) {
        ApplyFilterandSumBoundaryRho(
            m_fields.get_mr_levels(FieldType::rho_fp, finest_level),
            m_fields.get_mr_levels(FieldType::rho_cp, finest_level, skip_lev0_coarse_patch),
            fine_lev, PatchType::fine, 0, 2*ncomps);
    }

    EvolveB(fine_lev, PatchType::fine, 0.5_rt*dt[fine_lev], SubcyclingHalf::FirstHalf, cur_time);
    EvolveF(fine_lev, PatchType::fine, 0.5_rt*dt[fine_lev], /*rho_comp=*/0);
    FillBoundaryB(fine_lev, PatchType::fine, guard_cells.ng_FieldSolver,
                  WarpX::sync_nodal_points);
    FillBoundaryF(fine_lev, PatchType::fine, guard_cells.ng_alloc_F,
                  WarpX::sync_nodal_points);

    EvolveE(fine_lev, PatchType::fine, dt[fine_lev], cur_time);
    FillBoundaryE(fine_lev, PatchType::fine, guard_cells.ng_FieldGather);

    EvolveB(fine_lev, PatchType::fine, 0.5_rt*dt[fine_lev], SubcyclingHalf::SecondHalf, cur_time + 0.5_rt * dt[fine_lev]);
    EvolveF(fine_lev, PatchType::fine, 0.5_rt*dt[fine_lev], /*rho_comp=*/1);

    if (do_pml) {
        FillBoundaryF(fine_lev, PatchType::fine, guard_cells.ng_alloc_F);
        DampPML(fine_lev, PatchType::fine);
        FillBoundaryE(fine_lev, PatchType::fine, guard_cells.ng_FieldGather);
    }

    FillBoundaryB(fine_lev, PatchType::fine, guard_cells.ng_FieldGather);

    // ii) Push particles on the coarse patch and mother grid.
    // Push the fields on the coarse patch and mother grid
    // by only half a coarse step (first half)
    PushParticlesandDeposit(coarse_lev, cur_time, SubcyclingHalf::None);
    ::StoreCurrent(coarse_lev, m_fields);
    AddCurrentFromFineLevelandSumBoundary(
        m_fields.get_mr_levels_alldirs(FieldType::current_fp, finest_level),
        m_fields.get_mr_levels_alldirs(FieldType::current_cp, finest_level, skip_lev0_coarse_patch),
        m_fields.get_mr_levels_alldirs(FieldType::current_buf, finest_level, skip_lev0_coarse_patch), coarse_lev);

    if (m_fields.has(FieldType::rho_fp, finest_level) &&
        m_fields.has(FieldType::rho_cp, finest_level) &&
        m_fields.has(FieldType::rho_buf, finest_level)) {
        AddRhoFromFineLevelandSumBoundary(
            m_fields.get_mr_levels(FieldType::rho_fp, finest_level),
            m_fields.get_mr_levels(FieldType::rho_cp, finest_level, skip_lev0_coarse_patch),
            m_fields.get_mr_levels(FieldType::rho_buf, finest_level, skip_lev0_coarse_patch),
            coarse_lev, 0, ncomps);
    }

    EvolveB(fine_lev, PatchType::coarse, dt[fine_lev], SubcyclingHalf::FirstHalf, cur_time);
    EvolveF(fine_lev, PatchType::coarse, dt[fine_lev], /*rho_comp=*/0);
    FillBoundaryB(fine_lev, PatchType::coarse, guard_cells.ng_FieldGather);
    FillBoundaryF(fine_lev, PatchType::coarse, guard_cells.ng_FieldSolverF);

    EvolveE(fine_lev, PatchType::coarse, dt[fine_lev], cur_time);
    FillBoundaryE(fine_lev, PatchType::coarse, guard_cells.ng_FieldGather);

    EvolveB(coarse_lev, PatchType::fine, 0.5_rt*dt[coarse_lev], SubcyclingHalf::FirstHalf, cur_time);
    EvolveF(coarse_lev, PatchType::fine, 0.5_rt*dt[coarse_lev], /*rho_comp=*/0);
    FillBoundaryB(coarse_lev, PatchType::fine, guard_cells.ng_FieldGather,
                    WarpX::sync_nodal_points);
    FillBoundaryF(coarse_lev, PatchType::fine, guard_cells.ng_FieldSolverF,
                    WarpX::sync_nodal_points);

    EvolveE(coarse_lev, PatchType::fine, 0.5_rt*dt[coarse_lev], cur_time);
    FillBoundaryE(coarse_lev, PatchType::fine, guard_cells.ng_FieldGather);

    // TODO Remove call to FillBoundaryAux before UpdateAuxilaryData?
    FillBoundaryAux(guard_cells.ng_UpdateAux);
    // iii) Get auxiliary fields on the fine grid, at dt[fine_lev]
    UpdateAuxilaryData();
    FillBoundaryAux(guard_cells.ng_UpdateAux);

    // iv) Push particles and fields on the fine patch (second fine step)
    PushParticlesandDeposit(fine_lev, cur_time + dt[fine_lev], SubcyclingHalf::SecondHalf);
    RestrictCurrentFromFineToCoarsePatch(
        m_fields.get_mr_levels_alldirs(FieldType::current_fp, finest_level),
        m_fields.get_mr_levels_alldirs(FieldType::current_cp, finest_level, skip_lev0_coarse_patch), fine_lev);
    RestrictRhoFromFineToCoarsePatch(fine_lev);
    if (use_filter) {
        ApplyFilterMF( m_fields.get_mr_levels_alldirs(FieldType::current_fp, finest_level), fine_lev);
    }
    SumBoundaryJ( m_fields.get_mr_levels_alldirs(FieldType::current_fp, finest_level), fine_lev, Geom(fine_lev).periodicity());

    if (m_fields.has(FieldType::rho_fp, finest_level) &&
        m_fields.has(FieldType::rho_cp, finest_level)) {
        ApplyFilterandSumBoundaryRho(
            m_fields.get_mr_levels(FieldType::rho_fp, finest_level),
            m_fields.get_mr_levels(FieldType::rho_cp, finest_level, skip_lev0_coarse_patch),
            fine_lev, PatchType::fine, 0, ncomps);
    }

    EvolveB(fine_lev, PatchType::fine, 0.5_rt*dt[fine_lev], SubcyclingHalf::FirstHalf, cur_time + dt[fine_lev]);
    EvolveF(fine_lev, PatchType::fine, 0.5_rt*dt[fine_lev], /*rho_comp=*/0);
    FillBoundaryB(fine_lev, PatchType::fine, guard_cells.ng_FieldSolver);
    FillBoundaryF(fine_lev, PatchType::fine, guard_cells.ng_FieldSolverF);

    EvolveE(fine_lev, PatchType::fine, dt[fine_lev], cur_time + dt[fine_lev]);
    FillBoundaryE(fine_lev, PatchType::fine, guard_cells.ng_FieldSolver,
                    WarpX::sync_nodal_points);

    EvolveB(fine_lev, PatchType::fine, 0.5_rt*dt[fine_lev], SubcyclingHalf::SecondHalf, cur_time + 1.5_rt*dt[fine_lev]);
    EvolveF(fine_lev, PatchType::fine, 0.5_rt*dt[fine_lev], /*rho_comp=*/1);

    if (do_pml) {
        DampPML(fine_lev, PatchType::fine);
        FillBoundaryE(fine_lev, PatchType::fine, guard_cells.ng_FieldSolver);
    }

    if ( m_safe_guard_cells ) {
        FillBoundaryF(fine_lev, PatchType::fine, guard_cells.ng_FieldSolver);
    }
    FillBoundaryB(fine_lev, PatchType::fine, guard_cells.ng_FieldSolver);

    // v) Push the fields on the coarse patch and mother grid
    // by only half a coarse step (second half)
    ::RestoreCurrent(coarse_lev, m_fields);
    AddCurrentFromFineLevelandSumBoundary(
        m_fields.get_mr_levels_alldirs(FieldType::current_fp, finest_level),
        m_fields.get_mr_levels_alldirs(FieldType::current_cp, finest_level, skip_lev0_coarse_patch),
        m_fields.get_mr_levels_alldirs(FieldType::current_buf, finest_level, skip_lev0_coarse_patch),
        coarse_lev);

    if (m_fields.has(FieldType::rho_fp, finest_level) &&
        m_fields.has(FieldType::rho_cp, finest_level) &&
        m_fields.has(FieldType::rho_buf, finest_level)) {
        AddRhoFromFineLevelandSumBoundary(
            m_fields.get_mr_levels(FieldType::rho_fp, finest_level),
            m_fields.get_mr_levels(FieldType::rho_cp, finest_level, skip_lev0_coarse_patch),
            m_fields.get_mr_levels(FieldType::rho_buf, finest_level, skip_lev0_coarse_patch),
            coarse_lev, ncomps, ncomps);
    }

    EvolveE(fine_lev, PatchType::coarse, dt[fine_lev], cur_time + 0.5_rt * dt[fine_lev]);
    FillBoundaryE(fine_lev, PatchType::coarse, guard_cells.ng_FieldSolver,
                  WarpX::sync_nodal_points);

    EvolveB(fine_lev, PatchType::coarse, dt[fine_lev], SubcyclingHalf::SecondHalf, cur_time + 0.5_rt * dt[fine_lev]);
    EvolveF(fine_lev, PatchType::coarse, dt[fine_lev], /*rho_comp=*/1);

    if (do_pml) {
        FillBoundaryF(fine_lev, PatchType::fine, guard_cells.ng_FieldSolverF);
        DampPML(fine_lev, PatchType::coarse); // do it twice
        DampPML(fine_lev, PatchType::coarse);
        FillBoundaryE(fine_lev, PatchType::coarse, guard_cells.ng_alloc_EB);
    }

    FillBoundaryB(fine_lev, PatchType::coarse, guard_cells.ng_FieldSolver,
                  WarpX::sync_nodal_points);
    FillBoundaryF(fine_lev, PatchType::coarse, guard_cells.ng_FieldSolverF,
                  WarpX::sync_nodal_points);

    EvolveE(coarse_lev, PatchType::fine, 0.5_rt*dt[coarse_lev], cur_time + 0.5_rt*dt[coarse_lev]);
    FillBoundaryE(coarse_lev, PatchType::fine, guard_cells.ng_FieldSolver,
                  WarpX::sync_nodal_points);

    EvolveB(coarse_lev, PatchType::fine, 0.5_rt*dt[coarse_lev], SubcyclingHalf::SecondHalf, cur_time + 0.5_rt*dt[coarse_lev]);
    EvolveF(coarse_lev, PatchType::fine, 0.5_rt*dt[coarse_lev], /*rho_comp=*/1);

    if (do_pml) {
        if (moving_window_active(istep[0]+1)){
            // Exchange guard cells of PMLs only (0 cells are exchanged for the
            // regular B field MultiFab). This is required as B and F have just been
            // evolved.
            FillBoundaryB(coarse_lev, PatchType::fine, IntVect::TheZeroVector(),
                          WarpX::sync_nodal_points);
            FillBoundaryF(coarse_lev, PatchType::fine, IntVect::TheZeroVector(),
                          WarpX::sync_nodal_points);
        }
        DampPML(coarse_lev, PatchType::fine);
        if ( m_safe_guard_cells ) {
            FillBoundaryE(coarse_lev, PatchType::fine, guard_cells.ng_FieldSolver,
                          WarpX::sync_nodal_points);
        }
    }
    if ( m_safe_guard_cells ) {
        FillBoundaryB(coarse_lev, PatchType::fine, guard_cells.ng_FieldSolver,
                      WarpX::sync_nodal_points);
    }
}

void
WarpX::doFieldIonization ()
{
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    for (int lev = 0; lev <= finest_level; ++lev) {
        mypc->doFieldIonization(
            lev,
            *m_fields.get(FieldType::Efield_aux, Direction{0}, lev),
            *m_fields.get(FieldType::Efield_aux, Direction{1}, lev),
            *m_fields.get(FieldType::Efield_aux, Direction{2}, lev),
            *m_fields.get(FieldType::Bfield_aux, Direction{0}, lev),
            *m_fields.get(FieldType::Bfield_aux, Direction{1}, lev),
            *m_fields.get(FieldType::Bfield_aux, Direction{2}, lev)
        );
    }
}

#ifdef WARPX_QED
void
WarpX::doQEDEvents ()
{
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    for (int lev = 0; lev <= finest_level; ++lev) {
        mypc->doQedEvents(
            lev,
            *m_fields.get(FieldType::Efield_aux, Direction{0}, lev),
            *m_fields.get(FieldType::Efield_aux, Direction{1}, lev),
            *m_fields.get(FieldType::Efield_aux, Direction{2}, lev),
            *m_fields.get(FieldType::Bfield_aux, Direction{0}, lev),
            *m_fields.get(FieldType::Bfield_aux, Direction{1}, lev),
            *m_fields.get(FieldType::Bfield_aux, Direction{2}, lev)
        );
    }
}
#endif

void
WarpX::PushParticlesandDeposit (
    amrex::Real cur_time,
    bool skip_current,
    PositionPushType position_push_type,
    MomentumPushType momentum_push_type,
    ImplicitOptions const * implicit_options
)
// 推进粒子到新的时间步位置，并将粒子运动产生的电流密度沉积到网格上。
{
    // Evolve particles to p^{n+1/2} and x^{n+1}
    // Deposit current, j^{n+1/2}
    // 网格细化层级： finest_level
    // 每个网格层级上推进粒子并沉积电流密度
    for (int lev = 0; lev <= finest_level; ++lev) {
        PushParticlesandDeposit(
            lev,
            cur_time,
            SubcyclingHalf::None,
            skip_current,
            position_push_type,
            momentum_push_type,
            implicit_options
        );
    }
}

/**
 * @brief 推进粒子并沉积电流密度的核心函数
 * 
 * 该函数负责在单个网格层级上推进粒子到新的时间步位置，并将粒子运动产生的
 * 电流密度沉积到网格上。这是WarpX中粒子模拟的关键步骤，包含以下主要过程：
 * 
 * 1. 根据电流沉积算法选择适当的电流场标识符
 * 2. 调用粒子容器的Evolve函数执行粒子推进和电流沉积
 * 3. 在圆柱坐标系(RZ)下进行体积修正和边界处理
 * 4. 处理流体物种的演化（如果启用）
 * 
 * @param lev 当前网格层级
 * @param cur_time 当前物理时间
 * @param subcycling_half 子循环半步类型（None/FirstHalf/SecondHalf）
 * @param skip_current 是否跳过电流沉积
 * @param position_push_type 位置推进类型（Full/FirstHalf/SecondHalf/None）
 * @param momentum_push_type 动量推进类型（Full/FirstHalf/SecondHalf/None）
 * @param implicit_options 隐式求解选项（nullptr表示显式推进）
 */
void
WarpX::PushParticlesandDeposit (
    int lev,
    amrex::Real cur_time,
    SubcyclingHalf subcycling_half,
    bool skip_current,
    PositionPushType position_push_type,
    MomentumPushType momentum_push_type,
    ImplicitOptions const * implicit_options
)
{
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    std::string current_fp_string;

    // 确定当前密度的存储位置
    // 若采用节点中心存储，当前密度存储在current_fp_nodal
    // 若采用Vay deposition，当前密度存储在current_fp_vay
    // 否则，当前密度存储在current_fp
    // TEC 模拟电流沉积算法采用 direct
    if (WarpX::do_current_centering)
    {
        current_fp_string = "current_fp_nodal";
    }
    else if (WarpX::current_deposition_algo == CurrentDepositionAlgo::Vay)
    {
        current_fp_string = "current_fp_vay";
    }
    // TEC 计算对应这个
    else
    {
        current_fp_string = "current_fp";
    }

    // 计算粒子的运动，位矢变化、速度变化以及电流沉积
    mypc->Evolve(
        m_fields,
        lev,
        current_fp_string,
        cur_time,
        dt[lev],
        subcycling_half,
        skip_current,
        position_push_type,
        momentum_push_type,
        implicit_options
    );

    if (! skip_current) {
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        // This is called after all particles have deposited their current and charge.
        ApplyInverseVolumeScalingToCurrentDensity(
            m_fields.get(FieldType::current_fp, Direction{0}, lev),
            m_fields.get(FieldType::current_fp, Direction{1}, lev),
            m_fields.get(FieldType::current_fp, Direction{2}, lev),
            lev);
        if (m_fields.has_vector(FieldType::current_buf, lev)) {
            ApplyInverseVolumeScalingToCurrentDensity(
                m_fields.get(FieldType::current_buf, Direction{0}, lev),
                m_fields.get(FieldType::current_buf, Direction{1}, lev),
                m_fields.get(FieldType::current_buf, Direction{2}, lev),
                lev-1);
        }
        if (m_fields.has(FieldType::rho_fp, lev)) {
            ApplyInverseVolumeScalingToChargeDensity(m_fields.get(FieldType::rho_fp, lev), lev);
            if (m_fields.has(FieldType::rho_buf, lev)) {
                ApplyInverseVolumeScalingToChargeDensity(m_fields.get(FieldType::rho_buf, lev), lev-1);
            }
        }
// #else
        // I left this comment here as a reminder that currently the
        // boundary handling for cartesian grids are not matching the RZ handling
        // (done in the ApplyInverseScalingToChargeDensity function). The
        // Cartesian grid code had to be moved from here to after the application
        // of the filter to avoid incorrect results (moved to `SyncCurrentAndRho()`).
        // Might this be related to issue #1943?
#endif
        if (do_fluid_species) {
            myfl->Evolve(m_fields,
                         lev,
                         current_fp_string,
                         cur_time,
                         skip_current
            );
        }
    }
}

/* \brief Apply perfect mirror condition inside the box (not at a boundary).
 * In practice, set all fields to 0 on a section of the simulation domain
 * (as for a perfect conductor with a given thickness).
 * The mirror normal direction has to be parallel to the z axis.
 */
/**
 * @brief 在模拟域内部应用完美镜子边界条件（非边界处）
 * 
 * 该函数在模拟域的指定区域内将所有电磁场设置为零，模拟完美导体（PEC）的效果。
 * 镜子法线方向必须与z轴平行。此功能常用于模拟激光等离子体相互作用中的反射镜，
 * 或在特定区域抑制电磁场演化。
 * 
 * 主要功能：
 * 1. 检查是否有镜子需要处理
 * 2. 对每个镜子区域进行洛伦兹变换（用于加速坐标系）
 * 3. 在指定z范围内将所有电磁场分量清零
 * 4. 支持多网格层级和场清理场的处理
 * 
 * @param time 当前物理时间，用于加速坐标系下的坐标变换
 */
void
WarpX::applyMirrors (Real time)
{
    using ablastr::fields::Direction;

    // 检查是否有镜子需要处理，若无则直接返回
    if (m_num_mirrors == 0) {
        return;
    }

    // 循环处理每个镜子
    for(int i_mirror=0; i_mirror<m_num_mirrors; ++i_mirror)
    {
        // 获取镜子的z方向边界（下界和上界）
        amrex::Real z_min = m_mirror_z[i_mirror];
        amrex::Real z_max_tmp = z_min + m_mirror_z_width[i_mirror];

        // 对于加速坐标系模拟，对镜子坐标进行洛伦兹变换
        // 将实验室坐标系转换为加速坐标系
        if (gamma_boost>1)
        {
            z_min = z_min/gamma_boost - PhysConst::c*beta_boost*time;
            z_max_tmp = z_max_tmp/gamma_boost - PhysConst::c*beta_boost*time;
        }

        // 循环处理所有网格层级
        for(int lev=0; lev<=finest_level; lev++)
        {
            // 确保镜子区域至少包含指定数量的网格点
            const amrex::Real dz = WarpX::CellSize(lev)[2];
            const amrex::Real z_max = std::max(z_max_tmp, z_min+m_mirror_z_npoints[i_mirror]*dz);

            // 在细网格上将指定z范围内的电场分量清零
            NullifyMF(m_fields, "Efield_fp", Direction{0}, lev, z_min, z_max);
            NullifyMF(m_fields, "Efield_fp", Direction{1}, lev, z_min, z_max);
            NullifyMF(m_fields, "Efield_fp", Direction{2}, lev, z_min, z_max);
            // 在细网格上将指定z范围内的磁场分量清零
            NullifyMF(m_fields, "Bfield_fp", Direction{0}, lev, z_min, z_max);
            NullifyMF(m_fields, "Bfield_fp", Direction{1}, lev, z_min, z_max);
            NullifyMF(m_fields, "Bfield_fp", Direction{2}, lev, z_min, z_max);

            // 如果使用散度清理，将F/G场也清零
            NullifyMF(m_fields, "F_fp", lev, z_min, z_max);
            NullifyMF(m_fields, "G_fp", lev, z_min, z_max);

            // 对于粗网格层级（lev>0），同样处理粗网格上的场
            if (lev>0)
            {
                // 在粗网格上将指定z范围内的电场分量清零
                NullifyMF(m_fields, "Efield_cp", Direction{0}, lev, z_min, z_max);
                NullifyMF(m_fields, "Efield_cp", Direction{1}, lev, z_min, z_max);
                NullifyMF(m_fields, "Efield_cp", Direction{2}, lev, z_min, z_max);
                // 在粗网格上将指定z范围内的磁场分量清零
                NullifyMF(m_fields, "Bfield_cp", Direction{0}, lev, z_min, z_max);
                NullifyMF(m_fields, "Bfield_cp", Direction{1}, lev, z_min, z_max);
                NullifyMF(m_fields, "Bfield_cp", Direction{2}, lev, z_min, z_max);

                // 对粗网格上的散度清理场进行清零
                NullifyMF(m_fields, "F_cp", lev, z_min, z_max);
                NullifyMF(m_fields, "G_cp", lev, z_min, z_max);
            }
        }
    }
}


void
WarpX::HandleSignals()
{
    SignalHandling::WaitSignals();

    // SIGNAL_REQUESTS_BREAK is handled directly in WarpX::Evolve

    if (SignalHandling::TestAndResetActionRequestFlag(SignalHandling::SIGNAL_REQUESTS_CHECKPOINT)) {
        multi_diags->FilterComputePackFlushLastTimestep( istep[0] );
        ExecutePythonCallback("oncheckpointsignal");
    }
}
