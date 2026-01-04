/* Copyright 2019-2020 Andrew Myers, Ann Almgren, Axel Huebl
 * Burlen Loring, David Grote, Gunther H. Weber
 * Junmin Gu, Maxence Thevenet, Remi Lehe
 * Revathi Jambunathan, Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "WarpX.H"


#include "BoundaryConditions/PML.H"
#if (defined WARPX_DIM_RZ) && (defined WARPX_USE_FFT)
#    include "BoundaryConditions/PML_RZ.H"
#endif
#include "Diagnostics/Diagnostics.H"
#include "Diagnostics/MultiDiagnostics.H"
#include "Diagnostics/ReducedDiags/MultiReducedDiags.H"
#include "EmbeddedBoundary/Enabled.H"
#include "Fields.H"
#include "FieldIO.H"
#include "FieldSolver/ImplicitSolvers/ImplicitSolver.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXProfilerWrapper.H"

#include <ablastr/fields/MultiFabRegister.H>
#include <ablastr/utils/text/StreamUtils.H>

#ifdef AMREX_USE_SENSEI_INSITU
#   include <AMReX_AmrMeshInSituBridge.H>
#endif
#include <AMReX_BoxArray.H>
#include <AMReX_Config.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>
#include <AMReX_String.H>
#include <AMReX_Utility.H>
#include <AMReX_Vector.H>
#include <AMReX_VisMF.H>

#include <memory>
#include <string>
#include <sstream>

using namespace amrex;

namespace
{
    const std::string level_prefix {"Level_"};
}

amrex::DistributionMapping
WarpX::GetRestartDMap (const std::string& chkfile, const amrex::BoxArray& ba, int lev) const {
    std::string DMFileName = chkfile;
    if (!DMFileName.empty() && DMFileName[DMFileName.size()-1] != '/') {DMFileName += '/';}
    DMFileName = amrex::Concatenate(DMFileName + "Level_", lev, 1);
    DMFileName += "/DM";

    if (!amrex::FileExists(DMFileName)) {
        return amrex::DistributionMapping{ba, ParallelDescriptor::NProcs()};
    }

    Vector<char> fileCharPtr;
    ParallelDescriptor::ReadAndBcastFile(DMFileName, fileCharPtr);
    const std::string fileCharPtrString(fileCharPtr.dataPtr());
    std::istringstream DMFile(fileCharPtrString, std::istringstream::in);
    if ( ! DMFile.good()) { amrex::FileOpenFailed(DMFileName); }
    DMFile.exceptions(std::ios_base::failbit | std::ios_base::badbit);

    int nprocs_in_checkpoint;
    DMFile >> nprocs_in_checkpoint;
    if (nprocs_in_checkpoint != ParallelDescriptor::NProcs()) {
        return amrex::DistributionMapping{ba, ParallelDescriptor::NProcs()};
    }

    amrex::DistributionMapping dm;
    dm.readFrom(DMFile);
    if (dm.size() != ba.size()) {
        return amrex::DistributionMapping{ba, ParallelDescriptor::NProcs()};
    }

    return dm;
}

void
WarpX::InitFromCheckpoint ()
{
    // 引入字段方向枚举和字段类型枚举，用于后续的场数据恢复
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    // 性能分析标记：标记检查点重启函数的开始，用于性能监控和调试
    WARPX_PROFILE("WarpX::InitFromCheckpoint()");

    // 向用户输出重启信息，使用Utils::TextMsg::Info格式化消息
    // 输出格式：### INFO: restart from checkpoint [检查点路径]
    amrex::Print()<< Utils::TextMsg::Info(
        "restart from checkpoint " + restart_chkfile);

    // Header
    {
        // 构建检查点头文件路径：检查点目录 + WarpXHeader主文件
        const std::string File(restart_chkfile + "/WarpXHeader");

        // 获取AMReX可视化I/O的缓冲区大小，优化并行文件读取性能
        const VisMF::IO_Buffer io_buffer(VisMF::GetIOBufferSize());

        // 并行读取检查点头文件：IO处理器读取文件并广播给所有MPI进程
        Vector<char> fileCharPtr;
        ParallelDescriptor::ReadAndBcastFile(File, fileCharPtr);
        
        // 将读取的二进制数据转换为字符串，用于后续解析
        const std::string fileCharPtrString(fileCharPtr.dataPtr());
        std::istringstream is(fileCharPtrString, std::istringstream::in);
        
        // 设置严格的异常处理：任何读取错误都会抛出异常
        is.exceptions(std::ios_base::failbit | std::ios_base::badbit);

        // 临时变量用于解析文件内容
        std::string line, word;

        // 读取并跳过文件头标题行（版本信息或文件标识）
        std::getline(is, line);

        // 读取网格层级数量：检查点中保存的AMR网格层数
        int nlevs;
        is >> nlevs;
        ablastr::utils::text::goto_next_line(is);  // 跳过到下一行
        
        // 设置最细网格层级索引（AMReX使用0-based索引，nlevs-1为最细层）
        finest_level = nlevs-1;

    // 从检查点头文件中读取时间步进控制参数
    // 这些数据对于恢复模拟的精确状态至关重要，支持自适应网格细化(AMR)的多层级时间控制
    // 读取当前检查点运行的总步数
    std::getline(is, line);
    {
        std::istringstream lis(line);
        lis.exceptions(std::ios_base::failbit | std::ios_base::badbit);
        // 读取每个AMR层级的当前步数
        // istep数组保存每个细化层级的迭代步数，用于主循环控制
        // istep为层数 -- 1
        // istep[0] = 20000
        for (auto& istep_lev : istep) {
            lis >> word;
            // 转化为整数 istep_lev
            istep_lev = std::stoi(word);
        }
    }

    std::getline(is, line);
    {
        std::istringstream lis(line);
        lis.exceptions(std::ios_base::failbit | std::ios_base::badbit);
        // 读取每个层级的子步数配置
        // nsubsteps控制每个细化层级在一个主时间步内执行的子循环次数
        for (auto& nsub : nsubsteps) {
            lis >> word;
            nsub = std::stoi(word);
        }
    }

    // 当前模拟时间
    std::getline(is, line);
    {
        std::istringstream lis(line);
        lis.exceptions(std::ios_base::failbit | std::ios_base::badbit);
        // 读取每个层级的当前物理时间
        // t_new数组保存每个细化层级的最新物理时间，用于时间相关物理计算
        for (auto& t_new_lev : t_new) {
            lis >> word;
            t_new_lev = static_cast<Real>(std::stod(word));
        }
    }

    // 当前模拟上一步时间
    std::getline(is, line);
    {
        std::istringstream lis(line);
        lis.exceptions(std::ios_base::failbit | std::ios_base::badbit);
        // 读取每个层级的前一物理时间
        // t_old数组保存每个细化层级的前一时间步物理时间，用于时间导数计算
        for (auto& t_old_lev : t_old) {
            lis >> word;
            t_old_lev = static_cast<Real>(std::stod(word));
        }
    }

    // 时间步长
    std::getline(is, line);
    {
        std::istringstream lis(line);
        lis.exceptions(std::ios_base::failbit | std::ios_base::badbit);
        // 读取每个层级的时间步长
        // dt数组保存每个细化层级的本地时间步长，支持自适应时间步进和子循环
        for (auto& dt_lev : dt) {
            lis >> word;
            dt_lev = static_cast<Real>(std::stod(word));
        }
    }

        // 移动窗口的当前位置
        // moving_window_x_checkpoint保存了模拟中断时移动窗口的精确位置
        // 这对于长时程等离子体加速模拟至关重要，确保重启后窗口继续正确推进
        amrex::Real moving_window_x_checkpoint;
        is >> moving_window_x_checkpoint;
        ablastr::utils::text::goto_next_line(is);

        // 读取场同步状态标志
        // m_is_synchronized指示粒子速度与场是否处于同步状态
        // 在PIC算法中，这关系到粒子推进与场更新的时序一致性
        // 重启时必须恢复此状态以确保数值稳定性
        is >> m_is_synchronized;
        ablastr::utils::text::goto_next_line(is);

        // 从检查点读取计算域的物理空间下限
        // prob_lo数组保存了计算域在每个空间维度的物理下限坐标
        // 这对于设置网格和粒子初始化位置至关重要
        amrex::Vector<amrex::Real> prob_lo( AMREX_SPACEDIM );
        std::getline(is, line);
        {
            std::istringstream lis(line);
            lis.exceptions(std::ios_base::failbit | std::ios_base::badbit);
            for (auto& prob_lo_comp : prob_lo) {
                lis >> word;
                prob_lo_comp = static_cast<Real>(std::stod(word));
            }
        }

        // 从检查点读取计算域的物理空间上限
        // prob_hi数组保存了计算域在每个空间维度的物理上限坐标
        // 这对于设置网格和粒子初始化位置至关重要
        amrex::Vector<amrex::Real> prob_hi( AMREX_SPACEDIM );
        std::getline(is, line);
        {
            std::istringstream lis(line);
            lis.exceptions(std::ios_base::failbit | std::ios_base::badbit);
            for (auto& prob_hi_comp : prob_hi) {
                lis >> word;
                prob_hi_comp = static_cast<Real>(std::stod(word));
            }
        }

        // 从检查点恢复计算域的物理空间范围
        // 使用ResetProbDomain设置新的物理空间范围，确保与检查点一致
        ResetProbDomain(RealBox(prob_lo.data(),prob_hi.data()));

        // 重构AMR层级网格结构：从检查点恢复BoxArray和DistributionMapping
        // 遍历所有AMR层级，重建网格拓扑结构，为大规模并行计算做准备
        for (int lev = 0; lev < nlevs; ++lev) {
            BoxArray ba;                    // 存储当前层级的网格盒数组
            ba.readFrom(is);                // 从输入流读取BoxArray配置信息
            ablastr::utils::text::goto_next_line(is);  // 跳过当前行，准备读取下一项
            
            // 获取当前层级的分布映射：处理处理器重新分配，支持不同进程数重启
            const DistributionMapping dm = GetRestartDMap(restart_chkfile, ba, lev);
            
            // 设置当前层级的网格和分布信息
            SetBoxArray(lev, ba);           // 注册BoxArray到AMR核心
            SetDistributionMap(lev, dm);    // 注册DistributionMapping，定义数据分布
            
            // 分配层级数据存储：为场量、粒子等分配内存空间
            // 包括电磁场、电流密度、电荷密度等MultiFab的初始化
            AllocLevelData(lev, ba, dm);
        }

        // 恢复粒子容器状态：读取粒子种类信息和全局参数
        mypc->ReadHeader(is);               // 调用多粒子容器头信息读取
        const int n_species = mypc->nSpecies();  // 获取粒子种类总数

        // 读取粒子注入位置信息
        // 支持连续注入算法，保持粒子源的空间定位精度
        for (int i=0; i<n_species; i++)
        {
             is >> mypc->GetParticleContainer(i).m_current_injection_position;
             ablastr::utils::text::goto_next_line(is);
        }

        // 重启动前是否推动移动窗口
        int do_moving_window_before_restart;
        is >> do_moving_window_before_restart;
        ablastr::utils::text::goto_next_line(is);

        if (do_moving_window_before_restart) {
            moving_window_x = moving_window_x_checkpoint;
        }

        // 读取重启动前粒子移动时间点
        is >> time_of_last_gal_shift;
        ablastr::utils::text::goto_next_line(is);

        for (int idiag = 0; idiag < multi_diags->GetTotalDiags(); ++idiag)
        {
            if( multi_diags->diagstypes(idiag) == DiagTypes::BackTransformed )
            {
                auto& diag = multi_diags->GetDiag(idiag);
                if (diag.getnumbuffers() > 0) {
                    diag.InitDataBeforeRestart();
                    for (int i_buffer=0; i_buffer<diag.getnumbuffers(); ++i_buffer){
                        amrex::Real tlab;
                        is >> tlab;
                        diag.settlab(i_buffer, tlab);
                        int kindex_hi;
                        is >> kindex_hi;
                        diag.set_buffer_k_index_hi(i_buffer, kindex_hi);

                        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                            amrex::Real snapshot_lo;
                            is >> snapshot_lo;
                            diag.setSnapshotDomainLo(i_buffer, idim, snapshot_lo);
                        }
                        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                            amrex::Real snapshot_hi;
                            is >> snapshot_hi;
                            diag.setSnapshotDomainHi(i_buffer, idim, snapshot_hi);
                        }

                        int flush_counter;
                        is >> flush_counter;
                        diag.set_flush_counter(i_buffer, flush_counter);

                        int last_valid_Zslice;
                        is >> last_valid_Zslice;
                        diag.set_last_valid_Zslice(i_buffer, last_valid_Zslice);

                        int snapshot_full_flag;
                        is >> snapshot_full_flag;
                        diag.set_snapshot_full(i_buffer, snapshot_full_flag);

                    }
                    diag.InitDataAfterRestart(*mypc);
                } else {
                    diag.InitData(*mypc);
                }
            } else {
                multi_diags->GetDiag(idiag).InitData(*mypc);
            }
        }
    }

    const int nlevs = finestLevel()+1;

    // Initialize the field data
    for (int lev = 0; lev < nlevs; ++lev)
    {
        for (int i = 0; i < 3; ++i) {
            m_fields.get(FieldType::current_fp, Direction{i}, lev)->setVal(0.0);
            m_fields.get(FieldType::Efield_fp, Direction{i}, lev)->setVal(0.0);
            m_fields.get(FieldType::Bfield_fp, Direction{i}, lev)->setVal(0.0);
        }

        if (lev > 0) {
            for (int i = 0; i < 3; ++i) {
                m_fields.get(FieldType::Efield_aux, Direction{i}, lev)->setVal(0.0);
                m_fields.get(FieldType::Bfield_aux, Direction{i}, lev)->setVal(0.0);

                m_fields.get(FieldType::current_cp, Direction{i}, lev)->setVal(0.0);
                m_fields.get(FieldType::Efield_cp, Direction{i}, lev)->setVal(0.0);
                m_fields.get(FieldType::Bfield_cp, Direction{i}, lev)->setVal(0.0);
            }
        }

        VisMF::Read(*m_fields.get(FieldType::Efield_fp, Direction{0}, lev),
                    amrex::MultiFabFileFullPrefix(lev, restart_chkfile, level_prefix, "Ex_fp"));
        VisMF::Read(*m_fields.get(FieldType::Efield_fp, Direction{1}, lev),
                    amrex::MultiFabFileFullPrefix(lev, restart_chkfile, level_prefix, "Ey_fp"));
        VisMF::Read(*m_fields.get(FieldType::Efield_fp, Direction{2}, lev),
                    amrex::MultiFabFileFullPrefix(lev, restart_chkfile, level_prefix, "Ez_fp"));

        VisMF::Read(*m_fields.get(FieldType::Bfield_fp, Direction{0}, lev),
                    amrex::MultiFabFileFullPrefix(lev, restart_chkfile, level_prefix, "Bx_fp"));
        VisMF::Read(*m_fields.get(FieldType::Bfield_fp, Direction{1}, lev),
                    amrex::MultiFabFileFullPrefix(lev, restart_chkfile, level_prefix, "By_fp"));
        VisMF::Read(*m_fields.get(FieldType::Bfield_fp, Direction{2}, lev),
                    amrex::MultiFabFileFullPrefix(lev, restart_chkfile, level_prefix, "Bz_fp"));

        if (WarpX::fft_do_time_averaging)
        {
            VisMF::Read(*m_fields.get(FieldType::Efield_avg_fp, Direction{0}, lev),
                        amrex::MultiFabFileFullPrefix(lev, restart_chkfile, level_prefix, "Ex_avg_fp"));
            VisMF::Read(*m_fields.get(FieldType::Efield_avg_fp, Direction{1}, lev),
                        amrex::MultiFabFileFullPrefix(lev, restart_chkfile, level_prefix, "Ey_avg_fp"));
            VisMF::Read(*m_fields.get(FieldType::Efield_avg_fp, Direction{2}, lev),
                        amrex::MultiFabFileFullPrefix(lev, restart_chkfile, level_prefix, "Ez_avg_fp"));

            VisMF::Read(*m_fields.get(FieldType::Bfield_avg_fp, Direction{0}, lev),
                        amrex::MultiFabFileFullPrefix(lev, restart_chkfile, level_prefix, "Bx_avg_fp"));
            VisMF::Read(*m_fields.get(FieldType::Bfield_avg_fp, Direction{1}, lev),
                        amrex::MultiFabFileFullPrefix(lev, restart_chkfile, level_prefix, "By_avg_fp"));
            VisMF::Read(*m_fields.get(FieldType::Bfield_avg_fp, Direction{2}, lev),
                        amrex::MultiFabFileFullPrefix(lev, restart_chkfile, level_prefix, "Bz_avg_fp"));
        }

        if (m_is_synchronized) {
            VisMF::Read(*m_fields.get(FieldType::current_fp, Direction{0}, lev),
                        amrex::MultiFabFileFullPrefix(lev, restart_chkfile, level_prefix, "jx_fp"));
            VisMF::Read(*m_fields.get(FieldType::current_fp, Direction{1}, lev),
                        amrex::MultiFabFileFullPrefix(lev, restart_chkfile, level_prefix, "jy_fp"));
            VisMF::Read(*m_fields.get(FieldType::current_fp, Direction{2}, lev),
                        amrex::MultiFabFileFullPrefix(lev, restart_chkfile, level_prefix, "jz_fp"));
        }

        if (lev > 0)
        {
            VisMF::Read(*m_fields.get(FieldType::Efield_cp, Direction{0}, lev),
                        amrex::MultiFabFileFullPrefix(lev, restart_chkfile, level_prefix, "Ex_cp"));
            VisMF::Read(*m_fields.get(FieldType::Efield_cp, Direction{1}, lev),
                        amrex::MultiFabFileFullPrefix(lev, restart_chkfile, level_prefix, "Ey_cp"));
            VisMF::Read(*m_fields.get(FieldType::Efield_cp, Direction{2}, lev),
                        amrex::MultiFabFileFullPrefix(lev, restart_chkfile, level_prefix, "Ez_cp"));

            VisMF::Read(*m_fields.get(FieldType::Bfield_cp, Direction{0}, lev),
                        amrex::MultiFabFileFullPrefix(lev, restart_chkfile, level_prefix, "Bx_cp"));
            VisMF::Read(*m_fields.get(FieldType::Bfield_cp, Direction{1}, lev),
                        amrex::MultiFabFileFullPrefix(lev, restart_chkfile, level_prefix, "By_cp"));
            VisMF::Read(*m_fields.get(FieldType::Bfield_cp, Direction{2}, lev),
                        amrex::MultiFabFileFullPrefix(lev, restart_chkfile, level_prefix, "Bz_cp"));

            if (WarpX::fft_do_time_averaging)
            {
                VisMF::Read(*m_fields.get(FieldType::Efield_avg_cp, Direction{0}, lev),
                            amrex::MultiFabFileFullPrefix(lev, restart_chkfile, level_prefix, "Ex_avg_cp"));
                VisMF::Read(*m_fields.get(FieldType::Efield_avg_cp, Direction{1}, lev),
                            amrex::MultiFabFileFullPrefix(lev, restart_chkfile, level_prefix, "Ey_avg_cp"));
                VisMF::Read(*m_fields.get(FieldType::Efield_avg_cp, Direction{2}, lev),
                            amrex::MultiFabFileFullPrefix(lev, restart_chkfile, level_prefix, "Ez_avg_cp"));

                VisMF::Read(*m_fields.get(FieldType::Bfield_avg_cp, Direction{0}, lev),
                            amrex::MultiFabFileFullPrefix(lev, restart_chkfile, level_prefix, "Bx_avg_cp"));
                VisMF::Read(*m_fields.get(FieldType::Bfield_avg_cp, Direction{1}, lev),
                            amrex::MultiFabFileFullPrefix(lev, restart_chkfile, level_prefix, "By_avg_cp"));
                VisMF::Read(*m_fields.get(FieldType::Bfield_avg_cp, Direction{2}, lev),
                            amrex::MultiFabFileFullPrefix(lev, restart_chkfile, level_prefix, "Bz_avg_cp"));
            }

            if (m_is_synchronized) {
                VisMF::Read(*m_fields.get(FieldType::current_cp, Direction{0}, lev),
                            amrex::MultiFabFileFullPrefix(lev, restart_chkfile, level_prefix, "jx_cp"));
                VisMF::Read(*m_fields.get(FieldType::current_cp, Direction{1}, lev),
                            amrex::MultiFabFileFullPrefix(lev, restart_chkfile, level_prefix, "jy_cp"));
                VisMF::Read(*m_fields.get(FieldType::current_cp, Direction{2}, lev),
                            amrex::MultiFabFileFullPrefix(lev, restart_chkfile, level_prefix, "jz_cp"));
            }
        }
    }

    InitPML();
    if (do_pml)
    {
        for (int lev = 0; lev < nlevs; ++lev) {
            if (pml[lev]) {
                pml[lev]->Restart(m_fields, amrex::MultiFabFileFullPrefix(lev, restart_chkfile, level_prefix, "pml"));
            }
#if (defined WARPX_DIM_RZ) && (defined WARPX_USE_FFT)
            if (pml_rz[lev]) {
                pml_rz[lev]->Restart(m_fields, amrex::MultiFabFileFullPrefix(lev, restart_chkfile, level_prefix, "pml_rz"));
            }
#endif
        }
    }

    if (EB::enabled()) { InitializeEBGridData(maxLevel()); }

    reduced_diags->ReadCheckpointData(restart_chkfile);

    // Initialize particles
    mypc->AllocData();
    mypc->Restart(restart_chkfile);

    if (m_implicit_solver) {

        m_implicit_solver->Define(this);
        m_implicit_solver->CreateParticleAttributes();
    }

}
