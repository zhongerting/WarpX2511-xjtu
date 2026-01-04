/* Copyright 2025 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "FieldBoundaries.H"

#include "Utils/TextMsg.H"

#include <AMReX_ParmParse.H>
#include <AMReX_SPACE.H>

#include <algorithm>

namespace warpx::boundary_conditions
{

    namespace detail
    {
        void check_periodicity_consistency (
            const amrex::Array<FieldBoundaryType, AMREX_SPACEDIM>& field_boundary_lo,
            const amrex::Array<FieldBoundaryType, AMREX_SPACEDIM>& field_boundary_hi)
        {
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim){
                const bool is_lo_periodic =
                    (field_boundary_lo[idim]  == FieldBoundaryType::Periodic);
                const bool is_hi_periodic =
                    (field_boundary_hi[idim]  == FieldBoundaryType::Periodic);
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    (is_lo_periodic == is_hi_periodic),
                    "field boundary must be consistenly periodic in both lo and hi");
            }
        }
    }

    /**
     * @brief 解析并返回场边界条件配置
     * 
     * 该函数从输入文件中读取场边界条件设置，返回低边界和高边界的边界类型数组。
     * 支持每个维度的独立边界条件设置，并进行周期性一致性检查。
     * 
     * @return std::pair 包含两个Array的pair：
     *         - first: 低边界条件数组（每个维度一个）
     *         - second: 高边界条件数组（每个维度一个）
     * 
     * 支持的边界类型包括：
     * - Periodic: 周期性边界
     * - PEC: 完美电导体边界
     * - PMC: 完美磁导体边界
     * - Absorbing: 吸收边界
     * - Default: 默认边界条件
     */
    std::pair<
        amrex::Array<FieldBoundaryType, AMREX_SPACEDIM>,  // 低边界条件数组
        amrex::Array<FieldBoundaryType, AMREX_SPACEDIM>   // 高边界条件数组
    >
    parse_field_boundaries ()
    {
        // 初始化低边界条件数组，所有维度默认为Default类型
        // AMREX_D_DECL宏根据维度数量展开相应的参数
        auto field_boundary_lo =
            amrex::Array<FieldBoundaryType, AMREX_SPACEDIM>{
                AMREX_D_DECL(
                    FieldBoundaryType::Default,    // X维度低边界
                    FieldBoundaryType::Default,    // Y维度低边界（2D/3D）
                    FieldBoundaryType::Default)};  // Z维度低边界（3D）
        
        // 高边界条件初始化为与低边界相同
        auto field_boundary_hi = field_boundary_lo;

        // 创建boundary参数组解析器，用于读取输入文件中的边界设置
        const auto pp_boundary = amrex::ParmParse{"boundary"};

        // 遍历每个空间维度，读取对应的边界条件
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            // 读取低边界条件，支持带维度后缀的参数名（如field_lo-x, field_lo-y等）
            // "-_"参数允许使用下划线或连字符作为分隔符
            pp_boundary.query_enum_sloppy("field_lo",
                field_boundary_lo[idim], "-_", idim);
            
            // 读取高边界条件，同样支持维度后缀
            pp_boundary.query_enum_sloppy("field_hi",
                field_boundary_hi[idim], "-_", idim);
        }

        // 检查周期性边界条件的一致性
        // 确保如果一个维度是周期性的，其高低边界都设置为Periodic
        detail::check_periodicity_consistency(field_boundary_lo, field_boundary_hi);

        // 返回包含高低边界条件的pair
        return {field_boundary_lo, field_boundary_hi};
    }


    /**
     * @brief 根据场边界条件生成周期性标识数组
     * 
     * 该函数分析每个维度的场边界条件，确定哪些维度采用周期性边界条件。
     * 首先进行一致性检查确保边界条件设置合理，然后生成布尔数组标识周期性维度。
     * 
     * @param field_boundary_lo 低边界条件数组（每个维度一个）
     * @param field_boundary_hi 高边界条件数组（每个维度一个）
     * @return amrex::Array<bool, AMREX_SPACEDIM> 布尔数组，true表示对应维度为周期性边界
     * 
     * @note 周期性边界条件要求：如果一个维度是周期性的，其高低边界都必须设置为Periodic
     */
    amrex::Array<bool, AMREX_SPACEDIM>
    get_periodicity_array (
        const amrex::Array<FieldBoundaryType, AMREX_SPACEDIM>& field_boundary_lo,
        const amrex::Array<FieldBoundaryType, AMREX_SPACEDIM>& field_boundary_hi)
    {
        // 首先进行周期性一致性检查，确保边界条件设置合理
        // 检查包括：周期性维度的高低边界必须一致，非周期性边界不能混用等
        detail::check_periodicity_consistency(field_boundary_lo, field_boundary_hi);

        // 创建布尔数组用于存储每个维度的周期性标识
        auto is_field_boundary_periodic = amrex::Array<bool, AMREX_SPACEDIM>{};
        
        // 使用std::transform算法遍历低边界条件数组
        // 将每个维度的边界条件转换为布尔值（Periodic -> true，其他 -> false）
        std::transform (
            field_boundary_lo.begin(), field_boundary_lo.end(),           // 输入范围：低边界条件数组
            is_field_boundary_periodic.begin(),                           // 输出范围：周期性标识数组
            [](const auto& fb){return (fb == FieldBoundaryType::Periodic);}); // 转换函数：检查是否为Periodic
        
        // 返回周期性标识数组，用于后续的几何配置和算法选择
        // true表示对应维度采用周期性边界条件，false表示非周期性
        return is_field_boundary_periodic;
    }

}
