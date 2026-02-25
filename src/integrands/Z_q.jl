# ============================================================================
# src/integrands/Z_q.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module Z_q

export integrand_Z_q

include("Z_q/1_fundamental_sbar_block.jl")
include("Z_q/2_hmn_kernel_block.jl")
include("Z_q/3_smzk_khat_block.jl")
include("Z_q/4_smzk_scalar_block.jl")
include("Z_q/5_smzk_tensor_block.jl")
include("Z_q/6_smzk_prop_block.jl")
include("Z_q/7_pmu_block.jl")
include("Z_q/8_omunu_feyn_block.jl")
include("Z_q/9_omunu_land_block.jl")
include("Z_q/10_sum_sx_sy_omunu_feyn_block.jl")
include("Z_q/11_sum_sx_sy_omunu_land_block.jl")
include("Z_q/12_sum_sin_sum_sin_omunu_block.jl")

function integrand_Z_q(k)

    (
        sbar0, sbar1, sbar2, sbar3,
        cbar0, cbar1, cbar2, cbar3,
        sin1,  sin2,  sin3,  sin4,
        _s1sq, _s2sq, _s3sq, _s4sq,
        _s14t, _s24t, _s34t, _s44t,
        _s16t, _s26t, _s36t, _s46t,
        _s18t, _s28t, _s38t, _s48t,
        sum_ssq,
        sum_s4t,
        sum_s6t,
        sum_s8t,
        sum_ssq_sq,
        sum_ssq_3t,
        sum_ssq_4t,
        sum_ssq_5t,
        sum_s4t_sq,
        inv_sum_ssq,
        inv_sum_ssq_sq,
        inv_sum_ssq_s4t,
        inv_sum_ssq_s4t_sq
    ) = fundamental_sbar_block(
        k
    )

    # Smearing kernel h_{\mu\nu}

    (
        hmn_diag_1, hmn_diag_2, hmn_diag_3, hmn_diag_4,
        hmn_off_diag_1_2, hmn_off_diag_1_3, hmn_off_diag_1_4,
        hmn_off_diag_2_3, hmn_off_diag_2_4,
        hmn_off_diag_3_4,
        hmn_off_diag_2_1, hmn_off_diag_3_1, hmn_off_diag_4_1,
        hmn_off_diag_3_2, hmn_off_diag_4_2, 
        hmn_off_diag_4_3,
        hmn_diag_1_sq, hmn_diag_2_sq, hmn_diag_3_sq, hmn_diag_4_sq
    ) = hmn_kernel_block(
        _s1sq, _s2sq, _s3sq, _s4sq
    )

    # Symanzik improved gluon propagator

    (
        khat_1, khat_2, khat_3, khat_4,
        khat_1_sq, khat_2_sq, khat_3_sq, khat_4_sq,
        khat_1_4t, khat_2_4t, khat_3_4t, khat_4_4t,
        sum_khat_sq,
        inv_sum_khat_sq,
        sum_khat_sq_sq,
        sum_khat_sq_3t,
        sum_khat_sq_4t,
        sum_khat_sq_6t,
        khat_1_3t, khat_2_3t, khat_3_3t, khat_4_3t,
        sum_khat_4t,
        sum_khat_4t_sq,
        khat_1_5t, khat_2_5t, khat_3_5t, khat_4_5t,
        khat_1_6t, khat_2_6t, khat_3_6t, khat_4_6t,
        sum_khat_6t,
        khat_1_8t, khat_2_8t, khat_3_8t, khat_4_8t,
        sum_khat_8t
    ) = smzk_khat_block(
        sbar0, sbar1, sbar2, sbar3
    )
    
    (
        c_smzk_prop,
        f_smzk_prop,
        inv_f_smzk_prop,
        c_tilde_smzk_prop,
        c_tilde_smzk_prop_sq,
        c_tilde_smzk_prop_3t,
        x1_smzk_prop,
        x2_smzk_prop,
        x3_smzk_prop,
        smzk_block_01,
        smzk_block_02,
        deno_smzk_prop,
    ) = smzk_scalar_block(
        sum_khat_sq,
        inv_sum_khat_sq,
        sum_khat_sq_sq,
        sum_khat_sq_3t,
        sum_khat_sq_4t,
        sum_khat_sq_6t,
        sum_khat_4t,
        sum_khat_4t_sq,
        sum_khat_6t,
        sum_khat_8t
    )

    (
        p_mn_smzk_prop_11, p_mn_smzk_prop_22, p_mn_smzk_prop_33, p_mn_smzk_prop_44,
        p_mn_smzk_prop_12, p_mn_smzk_prop_13, p_mn_smzk_prop_14,
        p_mn_smzk_prop_23, p_mn_smzk_prop_24, 
        p_mn_smzk_prop_34,
        p_mn_smzk_prop_21, p_mn_smzk_prop_31, p_mn_smzk_prop_41,
        p_mn_smzk_prop_32, p_mn_smzk_prop_42, 
        p_mn_smzk_prop_43,

        coeff_delta_t,
        delta_t_smzk_prop_11, delta_t_smzk_prop_22, delta_t_smzk_prop_33, delta_t_smzk_prop_44,
        delta_t_smzk_prop_12, delta_t_smzk_prop_13, delta_t_smzk_prop_14, 
        delta_t_smzk_prop_23, delta_t_smzk_prop_24, 
        delta_t_smzk_prop_34,
        delta_t_smzk_prop_21, delta_t_smzk_prop_31, delta_t_smzk_prop_41, 
        delta_t_smzk_prop_32, delta_t_smzk_prop_42, 
        delta_t_smzk_prop_43,

        coeff_m_mn,
        m_mn_smzk_prop_11, m_mn_smzk_prop_22, m_mn_smzk_prop_33, m_mn_smzk_prop_44,
        m_mn_smzk_prop_12, m_mn_smzk_prop_13, m_mn_smzk_prop_14, 
        m_mn_smzk_prop_23, m_mn_smzk_prop_24, 
        m_mn_smzk_prop_34,
        m_mn_smzk_prop_21, m_mn_smzk_prop_31, m_mn_smzk_prop_41, 
        m_mn_smzk_prop_32, m_mn_smzk_prop_42, 
        m_mn_smzk_prop_43,

        coeff_m_sq_mn,
        m_sq_mn_smzk_prop_11, m_sq_mn_smzk_prop_22, m_sq_mn_smzk_prop_33, m_sq_mn_smzk_prop_44,
        m_sq_mn_smzk_prop_12, m_sq_mn_smzk_prop_13, m_sq_mn_smzk_prop_14, 
        m_sq_mn_smzk_prop_23, m_sq_mn_smzk_prop_24, 
        m_sq_mn_smzk_prop_34,
        m_sq_mn_smzk_prop_21, m_sq_mn_smzk_prop_31, m_sq_mn_smzk_prop_41, 
        m_sq_mn_smzk_prop_32, m_sq_mn_smzk_prop_42, 
        m_sq_mn_smzk_prop_43
    ) = smzk_tensor_block(
        inv_sum_khat_sq,
        smzk_block_01, 
        smzk_block_02,
        deno_smzk_prop, 
        c_tilde_smzk_prop,
        khat_1_sq, khat_2_sq, khat_3_sq, khat_4_sq,
        khat_1_4t, khat_2_4t, khat_3_4t, khat_4_4t,
        sum_khat_sq, 
        sum_khat_4t,
        c_tilde_smzk_prop_sq,
        khat_1_6t, khat_2_6t, khat_3_6t, khat_4_6t,
        sum_khat_sq_sq,
        sum_khat_4t_sq,
        sum_khat_6t
    )
    
    (
        imp_smzk_prop_feyn_11, imp_smzk_prop_feyn_22, imp_smzk_prop_feyn_33, imp_smzk_prop_feyn_44,
        imp_smzk_prop_feyn_12, imp_smzk_prop_feyn_13, imp_smzk_prop_feyn_14, 
        imp_smzk_prop_feyn_23, imp_smzk_prop_feyn_24,
        imp_smzk_prop_feyn_34, 
        imp_smzk_prop_feyn_21, imp_smzk_prop_feyn_31, imp_smzk_prop_feyn_41, 
        imp_smzk_prop_feyn_32, imp_smzk_prop_feyn_42, 
        imp_smzk_prop_feyn_43,

        imp_smzk_prop_land_11, imp_smzk_prop_land_22, imp_smzk_prop_land_33, imp_smzk_prop_land_44,
        imp_smzk_prop_land_12, imp_smzk_prop_land_13, imp_smzk_prop_land_14, 
        imp_smzk_prop_land_23, imp_smzk_prop_land_24,  
        imp_smzk_prop_land_34, 
        imp_smzk_prop_land_21, imp_smzk_prop_land_31, imp_smzk_prop_land_41, 
        imp_smzk_prop_land_32, imp_smzk_prop_land_42, 
        imp_smzk_prop_land_43
    ) = smzk_prop_block(
        coeff_delta_t,
        delta_t_smzk_prop_11, delta_t_smzk_prop_22, delta_t_smzk_prop_33, delta_t_smzk_prop_44,
        delta_t_smzk_prop_12, delta_t_smzk_prop_13, delta_t_smzk_prop_14, 
        delta_t_smzk_prop_23, delta_t_smzk_prop_24, 
        delta_t_smzk_prop_34,

        coeff_m_mn,
        m_mn_smzk_prop_11, m_mn_smzk_prop_22, m_mn_smzk_prop_33, m_mn_smzk_prop_44,
        m_mn_smzk_prop_12, m_mn_smzk_prop_13, m_mn_smzk_prop_14, 
        m_mn_smzk_prop_23, m_mn_smzk_prop_24, 
        m_mn_smzk_prop_34,

        coeff_m_sq_mn,
        m_sq_mn_smzk_prop_11, m_sq_mn_smzk_prop_22, m_sq_mn_smzk_prop_33, m_sq_mn_smzk_prop_44,
        m_sq_mn_smzk_prop_12, m_sq_mn_smzk_prop_13, m_sq_mn_smzk_prop_14, 
        m_sq_mn_smzk_prop_23, m_sq_mn_smzk_prop_24, 
        m_sq_mn_smzk_prop_34,

        inv_sum_khat_sq,
        p_mn_smzk_prop_11, p_mn_smzk_prop_22, p_mn_smzk_prop_33, p_mn_smzk_prop_44,
        p_mn_smzk_prop_12, p_mn_smzk_prop_13, p_mn_smzk_prop_14, 
        p_mn_smzk_prop_23, p_mn_smzk_prop_24, 
        p_mn_smzk_prop_34
    )

    (
        pmu_feyn_1, pmu_feyn_2, pmu_feyn_3, pmu_feyn_4,
        pmu_land_1, pmu_land_2, pmu_land_3, pmu_land_4,

        sum_pmu_feyn,
        sum_s2_pmu_feyn, sum_s4_pmu_feyn, sum_s6_pmu_feyn, sum_s8_pmu_feyn,

        sum_pmu_land,
        sum_s2_pmu_land, sum_s4_pmu_land, sum_s6_pmu_land, sum_s8_pmu_land
    ) = pmu_block(
        _s1sq, _s2sq, _s3sq, _s4sq,
        _s14t, _s24t, _s34t, _s44t,
        _s16t, _s26t, _s36t, _s46t,
        _s18t, _s28t, _s38t, _s48t,
        sum_ssq,

        imp_smzk_prop_feyn_11, imp_smzk_prop_feyn_22, imp_smzk_prop_feyn_33, imp_smzk_prop_feyn_44,
        imp_smzk_prop_feyn_12, imp_smzk_prop_feyn_13, imp_smzk_prop_feyn_14, 
        imp_smzk_prop_feyn_23, imp_smzk_prop_feyn_24,
        imp_smzk_prop_feyn_34, 
        imp_smzk_prop_feyn_21, imp_smzk_prop_feyn_31, imp_smzk_prop_feyn_41, 
        imp_smzk_prop_feyn_32, imp_smzk_prop_feyn_42, 
        imp_smzk_prop_feyn_43,

        imp_smzk_prop_land_11, imp_smzk_prop_land_22, imp_smzk_prop_land_33, imp_smzk_prop_land_44,
        imp_smzk_prop_land_12, imp_smzk_prop_land_13, imp_smzk_prop_land_14, 
        imp_smzk_prop_land_23, imp_smzk_prop_land_24,  
        imp_smzk_prop_land_34, 
        imp_smzk_prop_land_21, imp_smzk_prop_land_31, imp_smzk_prop_land_41, 
        imp_smzk_prop_land_32, imp_smzk_prop_land_42, 
        imp_smzk_prop_land_43,

        hmn_diag_1, hmn_diag_2, hmn_diag_3, hmn_diag_4,
        hmn_off_diag_1_2, hmn_off_diag_1_3, hmn_off_diag_1_4,
        hmn_off_diag_2_3, hmn_off_diag_2_4,
        hmn_off_diag_3_4,
        hmn_off_diag_2_1, hmn_off_diag_3_1, hmn_off_diag_4_1,
        hmn_off_diag_3_2, hmn_off_diag_4_2, 
        hmn_off_diag_4_3,
        hmn_diag_1_sq, hmn_diag_2_sq, hmn_diag_3_sq, hmn_diag_4_sq
    )

    (
        omunu_feyn_12, omunu_feyn_13, omunu_feyn_14,
        omunu_feyn_23, omunu_feyn_24,
        omunu_feyn_34,
        omunu_feyn_21, omunu_feyn_31, omunu_feyn_41, 
        omunu_feyn_32, omunu_feyn_42, 
        omunu_feyn_43
    ) = omunu_feyn_block(
        _s1sq, _s2sq, _s3sq, _s4sq,
        sum_ssq,

        imp_smzk_prop_feyn_11, imp_smzk_prop_feyn_22, imp_smzk_prop_feyn_33, imp_smzk_prop_feyn_44,
        imp_smzk_prop_feyn_12, imp_smzk_prop_feyn_13, imp_smzk_prop_feyn_14, 
        imp_smzk_prop_feyn_23, imp_smzk_prop_feyn_24,
        imp_smzk_prop_feyn_34, 
        imp_smzk_prop_feyn_21, imp_smzk_prop_feyn_31, imp_smzk_prop_feyn_41, 
        imp_smzk_prop_feyn_32, imp_smzk_prop_feyn_42, 
        imp_smzk_prop_feyn_43,

        hmn_diag_1, hmn_diag_2, hmn_diag_3, hmn_diag_4,
        hmn_off_diag_1_2, hmn_off_diag_1_3, hmn_off_diag_1_4,
        hmn_off_diag_2_3, hmn_off_diag_2_4,
        hmn_off_diag_3_4,
        hmn_off_diag_2_1, hmn_off_diag_3_1, hmn_off_diag_4_1,
        hmn_off_diag_3_2, hmn_off_diag_4_2, 
        hmn_off_diag_4_3
    )

    (
        omunu_land_12, omunu_land_13, omunu_land_14,
        omunu_land_23, omunu_land_24,
        omunu_land_34,
        omunu_land_21, omunu_land_31, omunu_land_41,
        omunu_land_32, omunu_land_42, 
        omunu_land_43
    ) = omunu_land_block(
        _s1sq, _s2sq, _s3sq, _s4sq, 
        sum_ssq,
        
        imp_smzk_prop_land_11, imp_smzk_prop_land_22, imp_smzk_prop_land_33, imp_smzk_prop_land_44,
        imp_smzk_prop_land_12, imp_smzk_prop_land_13, imp_smzk_prop_land_14, 
        imp_smzk_prop_land_23, imp_smzk_prop_land_24,  
        imp_smzk_prop_land_34, 
        imp_smzk_prop_land_21, imp_smzk_prop_land_31, imp_smzk_prop_land_41, 
        imp_smzk_prop_land_32, imp_smzk_prop_land_42, 
        imp_smzk_prop_land_43,
        
        hmn_diag_1, hmn_diag_2, hmn_diag_3, hmn_diag_4,
        hmn_off_diag_1_2, hmn_off_diag_1_3, hmn_off_diag_1_4,
        hmn_off_diag_2_3, hmn_off_diag_2_4,
        hmn_off_diag_3_4,
        hmn_off_diag_2_1, hmn_off_diag_3_1, hmn_off_diag_4_1,
        hmn_off_diag_3_2, hmn_off_diag_4_2, 
        hmn_off_diag_4_3
    )

    (
        sum_s2_sum_s2_omunu_feyn,
        sum_s2_sum_s4_omunu_feyn,
        sum_s4_sum_s2_omunu_feyn,
        sum_s4_sum_s4_omunu_feyn,
        sum_s6_sum_s2_omunu_feyn,
        sum_s6_sum_s4_omunu_feyn 
    ) = sum_sx_sy_omunu_feyn_block(
        _s1sq, _s2sq, _s3sq, _s4sq,
        _s14t, _s24t, _s34t, _s44t,
        _s16t, _s26t, _s36t, _s46t,

        omunu_feyn_12, omunu_feyn_13, omunu_feyn_14,
        omunu_feyn_23, omunu_feyn_24,
        omunu_feyn_34,
        omunu_feyn_21, omunu_feyn_31, omunu_feyn_41, 
        omunu_feyn_32, omunu_feyn_42, 
        omunu_feyn_43
    )

    (
        sum_s2_sum_s2_omunu_land,
        sum_s2_sum_s4_omunu_land,
        sum_s4_sum_s2_omunu_land,
        sum_s4_sum_s4_omunu_land,
        sum_s6_sum_s2_omunu_land,
        sum_s6_sum_s4_omunu_land 
    ) = sum_sx_sy_omunu_land_block(
        _s1sq, _s2sq, _s3sq, _s4sq,
        _s14t, _s24t, _s34t, _s44t,
        _s16t, _s26t, _s36t, _s46t,

        omunu_land_12, omunu_land_13, omunu_land_14,
        omunu_land_23, omunu_land_24,
        omunu_land_34,
        omunu_land_21, omunu_land_31, omunu_land_41,
        omunu_land_32, omunu_land_42, 
        omunu_land_43
    )

    (
        sum_sin_sum_sin_omunu_feyn,
        sum_sin_sum_sin_omunu_land 
    ) = sum_sin_sum_sin_omunu_block(
        sin1, sin2, sin3, sin4,

        omunu_feyn_12, omunu_feyn_13, omunu_feyn_14,
        omunu_feyn_23, omunu_feyn_24,
        omunu_feyn_34,
        omunu_feyn_21, omunu_feyn_31, omunu_feyn_41, 
        omunu_feyn_32, omunu_feyn_42, 
        omunu_feyn_43,

        omunu_land_12, omunu_land_13, omunu_land_14,
        omunu_land_23, omunu_land_24,
        omunu_land_34,
        omunu_land_21, omunu_land_31, omunu_land_41,
        omunu_land_32, omunu_land_42, 
        omunu_land_43
    )

    rtn = 0.0
    
    # ====================================================================
    # Z_F^{finite} + ZT_F
    # ====================================================================
    
    rtn = ( ( 0.5*sum_ssq*( sum_pmu_feyn - sum_s2_pmu_feyn ) -2.0*( sum_s2_pmu_feyn - sum_s4_pmu_feyn ) -4.0*( sum_s2_sum_s2_omunu_feyn - sum_s2_sum_s4_omunu_feyn ) + ( - sum_s2_pmu_feyn + 4.0*sum_s4_pmu_feyn - 5.0*sum_s6_pmu_feyn + 2.0*sum_s8_pmu_feyn - ( sum_s4t - sum_s6t )*( sum_pmu_feyn - sum_s2_pmu_feyn ) - 4.0*sum_s2_sum_s2_omunu_feyn + 4.0*sum_s2_sum_s4_omunu_feyn + 12.0*sum_s4_sum_s2_omunu_feyn - 12.0*sum_s4_sum_s4_omunu_feyn - 8.0*sum_s6_sum_s2_omunu_feyn + 8.0*sum_s6_sum_s4_omunu_feyn )*inv_sum_ssq_s4t ) * inv_sum_ssq*inv_sum_ssq_s4t + 1.0*inv_sum_ssq_sq + 0.5*sum_pmu_feyn*inv_sum_ssq ) * 1.0/16.0
    
    # ====================================================================
    # 2016年 10月 27日 木曜日 15:26:36 KST */
    # Generator Type: mt19937 */
    # Seed = 4357 */
    # First Value = 4293858116 */
    # VEGAS Warm-up ================== */
    # Result =  0.77390106 */
    # Sigma  =  0.00034883 */
    # VEGAS Converging... */
    # Result =  0.7737669383 Sigma =  1.490196097e-06 Chi^2/d.o.f. = 0.3 */
    # Result =  0.7737683288 Sigma =  1.17167743e-06 Chi^2/d.o.f. = 0.6 */
    # VEGAS Final ================== */
    # Result =  0.77376833 */
    # Sigma  =  0.00000117 */
    # 2016年 10月 27日 木曜日 15:52:07 KST */
    # ====================================================================
    
    # ====================================================================
    # ZM_F^{finite}
    # ====================================================================
        
    # rtn =  ( -1.0*(sum_pmu_feyn - sum_s2_pmu_feyn)*inv_sum_ssq*inv_sum_ssq_s4t + 4.0*inv_sum_ssq_sq ) * 1.0/16.0
    
    # ====================================================================
    # 2016年 10月 27日 木曜日 15:55:35 KST */
    # Generator Type: mt19937 */
    # Seed = 4357 */
    # First Value = 4293858116 */
    # VEGAS Warm-up ================== */
    # Result = -13.24175623 */
    # Sigma  =  0.00218409 */
    # VEGAS Converging... */
    # Result = -13.24243126 Sigma =  1.064123997e-05 Chi^2/d.o.f. = 1.4 */
    # VEGAS Final ================== */
    # Result = -13.24243126 */
    # Sigma  =  0.00001064 */
    # 2016年 10月 27日 木曜日 16:04:11 KST */
    # ====================================================================


    measure = 16*π*π / (2*π)^4

    # return rtn * measure
    return rtn * measure * 16
end

end  # module Z_q