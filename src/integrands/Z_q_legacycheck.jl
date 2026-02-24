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

function integrand_Z_q(k)

    # ------------------------------------------------------------
    # Fundamental \bar{s}_{\mu}
    # ------------------------------------------------------------

    sbar0 = sin(k[1]*0.5)
    sbar1 = sin(k[2]*0.5)
    sbar2 = sin(k[3]*0.5)
    sbar3 = sin(k[4]*0.5)

    cbar0 = cos(k[1]*0.5)
    cbar1 = cos(k[2]*0.5)
    cbar2 = cos(k[3]*0.5)
    cbar3 = cos(k[4]*0.5)

    sin1 = 2.0*sbar0*cbar0
    sin2 = 2.0*sbar1*cbar1
    sin3 = 2.0*sbar2*cbar2
    sin4 = 2.0*sbar3*cbar3

    _s1sq = sbar0*sbar0
    _s2sq = sbar1*sbar1
    _s3sq = sbar2*sbar2
    _s4sq = sbar3*sbar3

    _s14t = _s1sq*_s1sq
    _s24t = _s2sq*_s2sq
    _s34t = _s3sq*_s3sq
    _s44t = _s4sq*_s4sq

    _s16t = _s1sq*_s14t
    _s26t = _s2sq*_s24t
    _s36t = _s3sq*_s34t
    _s46t = _s4sq*_s44t

    _s18t = _s14t*_s14t
    _s28t = _s24t*_s24t
    _s38t = _s34t*_s34t
    _s48t = _s44t*_s44t

    # ------------------------------------------------------------
    # Various summation of \bar{s}_{\mu}
    # ------------------------------------------------------------

    sum_ssq = _s1sq + _s2sq + _s3sq + _s4sq
    sum_s4t = _s14t + _s24t + _s34t + _s44t
    sum_s6t = _s16t + _s26t + _s36t + _s46t
    sum_s8t = _s18t + _s28t + _s38t + _s48t

    sum_ssq_sq = sum_ssq * sum_ssq
    sum_ssq_3t = sum_ssq_sq * sum_ssq
    sum_ssq_4t = sum_ssq_sq * sum_ssq_sq
    sum_ssq_5t = sum_ssq_sq * sum_ssq_3t

    sum_s4t_sq = sum_s4t * sum_s4t

    inv_sum_ssq = 1.0/(sum_ssq)
    inv_sum_ssq_sq = inv_sum_ssq*inv_sum_ssq

    inv_sum_ssq_s4t = 0.0
    inv_sum_ssq_s4t = sum_ssq - sum_s4t
    inv_sum_ssq_s4t = 1.0/(inv_sum_ssq_s4t)
    inv_sum_ssq_s4t_sq = inv_sum_ssq_s4t * inv_sum_ssq_s4t

    # ------------------------------------------------------------
    # Smearing kernel h_{\mu\nu}
    # ------------------------------------------------------------

    hmn_diag_1 = 1
        - ( _s2sq + _s3sq + _s4sq )
        + ( _s2sq*_s3sq + _s2sq*_s4sq + _s3sq*_s4sq )
        - ( _s2sq*_s3sq*_s4sq )
    hmn_diag_2 = 1
        - ( _s1sq + _s3sq + _s4sq )
        + ( _s1sq*_s3sq + _s1sq*_s4sq + _s3sq*_s4sq )
        - ( _s1sq*_s3sq*_s4sq )
    hmn_diag_3 = 1
        - ( _s1sq + _s2sq + _s4sq )
        + ( _s1sq*_s2sq + _s2sq*_s4sq + _s1sq*_s4sq )
        - ( _s1sq*_s2sq*_s4sq )
    hmn_diag_4 = 1
        - ( _s1sq + _s2sq + _s3sq )
        + ( _s1sq*_s2sq + _s1sq*_s3sq + _s2sq*_s3sq )
        - ( _s1sq*_s2sq*_s3sq )

    hmn_off_diag_2_1 = 1 - 0.5*( _s3sq + _s4sq ) + (1.0/3.0)*_s3sq*_s4sq 
    hmn_off_diag_3_1 = 1 - 0.5*( _s2sq + _s4sq ) + (1.0/3.0)*_s2sq*_s4sq 
    hmn_off_diag_4_1 = 1 - 0.5*( _s2sq + _s3sq ) + (1.0/3.0)*_s2sq*_s3sq 
    hmn_off_diag_3_2 = 1 - 0.5*( _s1sq + _s4sq ) + (1.0/3.0)*_s1sq*_s4sq 
    hmn_off_diag_4_2 = 1 - 0.5*( _s1sq + _s3sq ) + (1.0/3.0)*_s1sq*_s3sq 
    hmn_off_diag_4_3 = 1 - 0.5*( _s1sq + _s2sq ) + (1.0/3.0)*_s1sq*_s2sq 

    hmn_off_diag_1_2 = hmn_off_diag_2_1
    hmn_off_diag_1_3 = hmn_off_diag_3_1
    hmn_off_diag_1_4 = hmn_off_diag_4_1
    hmn_off_diag_2_3 = hmn_off_diag_3_2
    hmn_off_diag_2_4 = hmn_off_diag_4_2
    hmn_off_diag_3_4 = hmn_off_diag_4_3
    
    hmn_diag_1_sq = hmn_diag_1 * hmn_diag_1
    hmn_diag_2_sq = hmn_diag_2 * hmn_diag_2
    hmn_diag_3_sq = hmn_diag_3 * hmn_diag_3
    hmn_diag_4_sq = hmn_diag_4 * hmn_diag_4

    # ------------------------------------------------------------
    # Symanzik improved gluon propagator
    # ------------------------------------------------------------

    khat_1 = 2.0*sbar0
    khat_2 = 2.0*sbar1
    khat_3 = 2.0*sbar2
    khat_4 = 2.0*sbar3

    khat_1_sq = khat_1*khat_1
    khat_2_sq = khat_2*khat_2
    khat_3_sq = khat_3*khat_3
    khat_4_sq = khat_4*khat_4

    khat_1_4t = khat_1_sq*khat_1_sq
    khat_2_4t = khat_2_sq*khat_2_sq
    khat_3_4t = khat_3_sq*khat_3_sq
    khat_4_4t = khat_4_sq*khat_4_sq

    sum_khat_sq = khat_1_sq + khat_2_sq + khat_3_sq + khat_4_sq 
    inv_sum_khat_sq = 1.0/(sum_khat_sq)
    sum_khat_sq_sq = sum_khat_sq*sum_khat_sq
    sum_khat_sq_3t = sum_khat_sq_sq*sum_khat_sq
    sum_khat_sq_4t = sum_khat_sq_sq*sum_khat_sq_sq
    sum_khat_sq_6t = sum_khat_sq_4t*sum_khat_sq_sq
    
    khat_1_3t = khat_1_sq*khat_1
    khat_2_3t = khat_2_sq*khat_2
    khat_3_3t = khat_3_sq*khat_3
    khat_4_3t = khat_4_sq*khat_4

    sum_khat_4t = khat_1_4t + khat_2_4t + khat_3_4t + khat_4_4t 
    sum_khat_4t_sq = sum_khat_4t*sum_khat_4t

    khat_1_5t = khat_1_3t*khat_1_sq
    khat_2_5t = khat_2_3t*khat_2_sq
    khat_3_5t = khat_3_3t*khat_3_sq
    khat_4_5t = khat_4_3t*khat_4_sq

    khat_1_6t = khat_1_3t*khat_1_3t
    khat_2_6t = khat_2_3t*khat_2_3t
    khat_3_6t = khat_3_3t*khat_3_3t
    khat_4_6t = khat_4_3t*khat_4_3t

    sum_khat_6t = khat_1_6t + khat_2_6t + khat_3_6t + khat_4_6t 

    khat_1_8t = khat_1_4t*khat_1_4t
    khat_2_8t = khat_2_4t*khat_2_4t
    khat_3_8t = khat_3_4t*khat_3_4t
    khat_4_8t = khat_4_4t*khat_4_4t

    sum_khat_8t = khat_1_8t + khat_2_8t + khat_3_8t + khat_4_8t 
    
    c_smzk_prop = -1.0/12.0
    
    f_smzk_prop = 1.0 - c_smzk_prop * sum_khat_4t * inv_sum_khat_sq 
    inv_f_smzk_prop = 1.0/(f_smzk_prop) 

    c_tilde_smzk_prop = c_smzk_prop * inv_f_smzk_prop 
    c_tilde_smzk_prop_sq = c_tilde_smzk_prop*c_tilde_smzk_prop
    c_tilde_smzk_prop_3t = c_tilde_smzk_prop*c_tilde_smzk_prop_sq

    x1_smzk_prop = sum_khat_sq_sq - sum_khat_4t
    x2_smzk_prop = sum_khat_sq*sum_khat_6t - 1.5*sum_khat_sq_sq*sum_khat_4t + 0.5*sum_khat_sq_4t
    x3_smzk_prop = (1.0/6.0)*sum_khat_sq_6t + 0.5*sum_khat_sq_sq*sum_khat_4t_sq
        - sum_khat_sq_4t*sum_khat_4t + (4.0/3.0)*sum_khat_sq_3t*sum_khat_6t
        - sum_khat_sq_sq*sum_khat_8t

    smzk_block_01 = sum_khat_sq - c_tilde_smzk_prop*x1_smzk_prop
    smzk_block_02 = sum_khat_sq*smzk_block_01 + c_tilde_smzk_prop_sq*x2_smzk_prop
    
    deno_smzk_prop = 0.0
    deno_smzk_prop = sum_khat_sq*smzk_block_02 - c_tilde_smzk_prop_3t*x3_smzk_prop
    deno_smzk_prop *= f_smzk_prop
    deno_smzk_prop = 1.0/(deno_smzk_prop)

    p_mn_smzk_prop_11 = khat_1_sq*inv_sum_khat_sq 
    p_mn_smzk_prop_22 = khat_2_sq*inv_sum_khat_sq 
    p_mn_smzk_prop_33 = khat_3_sq*inv_sum_khat_sq 
    p_mn_smzk_prop_44 = khat_4_sq*inv_sum_khat_sq 

    # ------------------------------------------------------------
    # off-diagonal part : factor out \hat{k}_\mu \hat{k}_\nu
    # ------------------------------------------------------------

    p_mn_smzk_prop_12 = inv_sum_khat_sq 
    p_mn_smzk_prop_13 = inv_sum_khat_sq 
    p_mn_smzk_prop_14 = inv_sum_khat_sq 
    p_mn_smzk_prop_23 = inv_sum_khat_sq 
    p_mn_smzk_prop_24 = inv_sum_khat_sq 
    p_mn_smzk_prop_34 = inv_sum_khat_sq 

    p_mn_smzk_prop_21 = p_mn_smzk_prop_12 
    p_mn_smzk_prop_31 = p_mn_smzk_prop_13 
    p_mn_smzk_prop_41 = p_mn_smzk_prop_14 
    p_mn_smzk_prop_32 = p_mn_smzk_prop_23 
    p_mn_smzk_prop_42 = p_mn_smzk_prop_24 
    p_mn_smzk_prop_43 = p_mn_smzk_prop_34 

    coeff_delta_t = smzk_block_02*deno_smzk_prop
    
    delta_t_smzk_prop_11 = 1.0 - p_mn_smzk_prop_11 
    delta_t_smzk_prop_22 = 1.0 - p_mn_smzk_prop_22 
    delta_t_smzk_prop_33 = 1.0 - p_mn_smzk_prop_33 
    delta_t_smzk_prop_44 = 1.0 - p_mn_smzk_prop_44 

    delta_t_smzk_prop_12 = - p_mn_smzk_prop_12 
    delta_t_smzk_prop_13 = - p_mn_smzk_prop_13 
    delta_t_smzk_prop_14 = - p_mn_smzk_prop_14 
    delta_t_smzk_prop_23 = - p_mn_smzk_prop_23 
    delta_t_smzk_prop_24 = - p_mn_smzk_prop_24 
    delta_t_smzk_prop_34 = - p_mn_smzk_prop_34 
    
    delta_t_smzk_prop_21 = delta_t_smzk_prop_12 
    delta_t_smzk_prop_31 = delta_t_smzk_prop_13 
    delta_t_smzk_prop_41 = delta_t_smzk_prop_14 
    delta_t_smzk_prop_32 = delta_t_smzk_prop_23 
    delta_t_smzk_prop_42 = delta_t_smzk_prop_24 
    delta_t_smzk_prop_43 = delta_t_smzk_prop_34 

    coeff_m_mn = c_tilde_smzk_prop*smzk_block_01*deno_smzk_prop
    
    m_mn_smzk_prop_11 =
        khat_1_sq*sum_khat_sq - 2.0*khat_1_4t + khat_1_sq*sum_khat_4t*inv_sum_khat_sq
    m_mn_smzk_prop_22 =
        khat_2_sq*sum_khat_sq - 2.0*khat_2_4t + khat_2_sq*sum_khat_4t*inv_sum_khat_sq
    m_mn_smzk_prop_33 =
        khat_3_sq*sum_khat_sq - 2.0*khat_3_4t + khat_3_sq*sum_khat_4t*inv_sum_khat_sq
    m_mn_smzk_prop_44 =
        khat_4_sq*sum_khat_sq - 2.0*khat_4_4t + khat_4_sq*sum_khat_4t*inv_sum_khat_sq

    m_mn_smzk_prop_12 = - khat_1_sq - khat_2_sq + sum_khat_4t*inv_sum_khat_sq
    m_mn_smzk_prop_13 = - khat_1_sq - khat_3_sq + sum_khat_4t*inv_sum_khat_sq
    m_mn_smzk_prop_14 = - khat_1_sq - khat_4_sq + sum_khat_4t*inv_sum_khat_sq
    m_mn_smzk_prop_23 = - khat_2_sq - khat_3_sq + sum_khat_4t*inv_sum_khat_sq
    m_mn_smzk_prop_24 = - khat_2_sq - khat_4_sq + sum_khat_4t*inv_sum_khat_sq
    m_mn_smzk_prop_34 = - khat_3_sq - khat_4_sq + sum_khat_4t*inv_sum_khat_sq
    
    m_mn_smzk_prop_21 = m_mn_smzk_prop_12
    m_mn_smzk_prop_31 = m_mn_smzk_prop_13
    m_mn_smzk_prop_41 = m_mn_smzk_prop_14
    m_mn_smzk_prop_32 = m_mn_smzk_prop_23
    m_mn_smzk_prop_42 = m_mn_smzk_prop_24
    m_mn_smzk_prop_43 = m_mn_smzk_prop_34
        
    coeff_m_sq_mn = c_tilde_smzk_prop_sq*deno_smzk_prop
    
    m_sq_mn_smzk_prop_11 =
        khat_1_4t*sum_khat_sq_sq - 3.0*khat_1_6t*sum_khat_sq + 2.0*khat_1_4t*sum_khat_4t
        + khat_1_sq*sum_khat_6t - khat_1_sq*sum_khat_4t_sq*inv_sum_khat_sq
    m_sq_mn_smzk_prop_22 =
        khat_2_4t*sum_khat_sq_sq - 3.0*khat_2_6t*sum_khat_sq + 2.0*khat_2_4t*sum_khat_4t
        + khat_2_sq*sum_khat_6t - khat_2_sq*sum_khat_4t_sq*inv_sum_khat_sq
    m_sq_mn_smzk_prop_33 =
        khat_3_4t*sum_khat_sq_sq - 3.0*khat_3_6t*sum_khat_sq + 2.0*khat_3_4t*sum_khat_4t
        + khat_3_sq*sum_khat_6t - khat_3_sq*sum_khat_4t_sq*inv_sum_khat_sq
    m_sq_mn_smzk_prop_44 =
        khat_4_4t*sum_khat_sq_sq - 3.0*khat_4_6t*sum_khat_sq + 2.0*khat_4_4t*sum_khat_4t
        + khat_4_sq*sum_khat_6t - khat_4_sq*sum_khat_4t_sq*inv_sum_khat_sq

    m_sq_mn_smzk_prop_12 =
        - ( khat_1_4t + khat_2_4t + khat_1_sq*khat_2_sq )*sum_khat_sq
        + ( khat_1_sq + khat_2_sq )*sum_khat_4t
        + sum_khat_6t - sum_khat_4t_sq*inv_sum_khat_sq
    m_sq_mn_smzk_prop_13 =
        - ( khat_1_4t + khat_3_4t + khat_1_sq*khat_3_sq )*sum_khat_sq
        + ( khat_1_sq + khat_3_sq )*sum_khat_4t
        + sum_khat_6t - sum_khat_4t_sq*inv_sum_khat_sq
    m_sq_mn_smzk_prop_14 =
        - ( khat_1_4t + khat_4_4t + khat_1_sq*khat_4_sq )*sum_khat_sq
        + ( khat_1_sq + khat_4_sq )*sum_khat_4t
        + sum_khat_6t - sum_khat_4t_sq*inv_sum_khat_sq
    m_sq_mn_smzk_prop_23 =
        - ( khat_2_4t + khat_3_4t + khat_2_sq*khat_3_sq )*sum_khat_sq
        + ( khat_2_sq + khat_3_sq )*sum_khat_4t
        + sum_khat_6t - sum_khat_4t_sq*inv_sum_khat_sq
    m_sq_mn_smzk_prop_24 =
        - ( khat_2_4t + khat_4_4t + khat_2_sq*khat_4_sq )*sum_khat_sq
        + ( khat_2_sq + khat_4_sq )*sum_khat_4t
        + sum_khat_6t - sum_khat_4t_sq*inv_sum_khat_sq
    m_sq_mn_smzk_prop_34 =
        - ( khat_3_4t + khat_4_4t + khat_3_sq*khat_4_sq )*sum_khat_sq
        + ( khat_3_sq + khat_4_sq )*sum_khat_4t
        + sum_khat_6t - sum_khat_4t_sq*inv_sum_khat_sq

    m_sq_mn_smzk_prop_21 = m_sq_mn_smzk_prop_12 
    m_sq_mn_smzk_prop_31 = m_sq_mn_smzk_prop_13 
    m_sq_mn_smzk_prop_41 = m_sq_mn_smzk_prop_14 
    m_sq_mn_smzk_prop_32 = m_sq_mn_smzk_prop_23 
    m_sq_mn_smzk_prop_42 = m_sq_mn_smzk_prop_24 
    m_sq_mn_smzk_prop_43 = m_sq_mn_smzk_prop_34 
    
    imp_smzk_prop_feyn_11 =
        coeff_delta_t*delta_t_smzk_prop_11
        + coeff_m_mn*m_mn_smzk_prop_11
        + coeff_m_sq_mn*m_sq_mn_smzk_prop_11
        + inv_sum_khat_sq*p_mn_smzk_prop_11 
    imp_smzk_prop_feyn_22 =
        coeff_delta_t*delta_t_smzk_prop_22
        + coeff_m_mn*m_mn_smzk_prop_22
        + coeff_m_sq_mn*m_sq_mn_smzk_prop_22
        + inv_sum_khat_sq*p_mn_smzk_prop_22 
    imp_smzk_prop_feyn_33 =
        coeff_delta_t*delta_t_smzk_prop_33
        + coeff_m_mn*m_mn_smzk_prop_33
        + coeff_m_sq_mn*m_sq_mn_smzk_prop_33
        + inv_sum_khat_sq*p_mn_smzk_prop_33 
    imp_smzk_prop_feyn_44 =
        coeff_delta_t*delta_t_smzk_prop_44
        + coeff_m_mn*m_mn_smzk_prop_44
        + coeff_m_sq_mn*m_sq_mn_smzk_prop_44
        + inv_sum_khat_sq*p_mn_smzk_prop_44 
    imp_smzk_prop_feyn_12 =
        coeff_delta_t*delta_t_smzk_prop_12
        + coeff_m_mn*m_mn_smzk_prop_12
        + coeff_m_sq_mn*m_sq_mn_smzk_prop_12 
        + inv_sum_khat_sq*p_mn_smzk_prop_12 
    imp_smzk_prop_feyn_13 =
        coeff_delta_t*delta_t_smzk_prop_13
        + coeff_m_mn*m_mn_smzk_prop_13
        + coeff_m_sq_mn*m_sq_mn_smzk_prop_13
        + inv_sum_khat_sq*p_mn_smzk_prop_13 
    imp_smzk_prop_feyn_14 =
        coeff_delta_t*delta_t_smzk_prop_14
        + coeff_m_mn*m_mn_smzk_prop_14
        + coeff_m_sq_mn*m_sq_mn_smzk_prop_14
        + inv_sum_khat_sq*p_mn_smzk_prop_14 
    imp_smzk_prop_feyn_23 =
        coeff_delta_t*delta_t_smzk_prop_23
        + coeff_m_mn*m_mn_smzk_prop_23
        + coeff_m_sq_mn*m_sq_mn_smzk_prop_23
        + inv_sum_khat_sq*p_mn_smzk_prop_23 
    imp_smzk_prop_feyn_24 =
        coeff_delta_t*delta_t_smzk_prop_24
        + coeff_m_mn*m_mn_smzk_prop_24
        + coeff_m_sq_mn*m_sq_mn_smzk_prop_24
        + inv_sum_khat_sq*p_mn_smzk_prop_24 
    imp_smzk_prop_feyn_34 =
        coeff_delta_t*delta_t_smzk_prop_34
        + coeff_m_mn*m_mn_smzk_prop_34
        + coeff_m_sq_mn*m_sq_mn_smzk_prop_34
        + inv_sum_khat_sq*p_mn_smzk_prop_34 
    imp_smzk_prop_feyn_21 = imp_smzk_prop_feyn_12 
    imp_smzk_prop_feyn_31 = imp_smzk_prop_feyn_13 
    imp_smzk_prop_feyn_41 = imp_smzk_prop_feyn_14   
    imp_smzk_prop_feyn_32 = imp_smzk_prop_feyn_23 
    imp_smzk_prop_feyn_42 = imp_smzk_prop_feyn_24 
    imp_smzk_prop_feyn_43 = imp_smzk_prop_feyn_34 
    
    imp_smzk_prop_land_11 = - p_mn_smzk_prop_11*inv_sum_khat_sq 
    imp_smzk_prop_land_22 = - p_mn_smzk_prop_22*inv_sum_khat_sq 
    imp_smzk_prop_land_33 = - p_mn_smzk_prop_33*inv_sum_khat_sq 
    imp_smzk_prop_land_44 = - p_mn_smzk_prop_44*inv_sum_khat_sq 
    imp_smzk_prop_land_12 = - p_mn_smzk_prop_12*inv_sum_khat_sq 
    imp_smzk_prop_land_13 = - p_mn_smzk_prop_13*inv_sum_khat_sq 
    imp_smzk_prop_land_14 = - p_mn_smzk_prop_14*inv_sum_khat_sq 
    imp_smzk_prop_land_23 = - p_mn_smzk_prop_23*inv_sum_khat_sq 
    imp_smzk_prop_land_24 = - p_mn_smzk_prop_24*inv_sum_khat_sq 
    imp_smzk_prop_land_34 = - p_mn_smzk_prop_34*inv_sum_khat_sq 
    imp_smzk_prop_land_21 = imp_smzk_prop_land_12 
    imp_smzk_prop_land_31 = imp_smzk_prop_land_13 
    imp_smzk_prop_land_41 = imp_smzk_prop_land_14 
    imp_smzk_prop_land_32 = imp_smzk_prop_land_23 
    imp_smzk_prop_land_42 = imp_smzk_prop_land_24 
    imp_smzk_prop_land_43 = imp_smzk_prop_land_34 	    

    # ------------------------------------------------------------
    # Diagonal part of smeared-smeared propagator
    # ------------------------------------------------------------

    pmu_feyn_1 =
        imp_smzk_prop_feyn_11 * hmn_diag_1_sq
        + _s1sq*_s2sq*imp_smzk_prop_feyn_22*hmn_off_diag_2_1*hmn_off_diag_2_1
        + _s1sq*_s3sq*imp_smzk_prop_feyn_33*hmn_off_diag_3_1*hmn_off_diag_3_1
        + _s1sq*_s4sq*imp_smzk_prop_feyn_44*hmn_off_diag_4_1*hmn_off_diag_4_1
        + 8.0*_s1sq*_s2sq*imp_smzk_prop_feyn_12*hmn_off_diag_2_1*hmn_diag_1
        + 8.0*_s1sq*_s3sq*imp_smzk_prop_feyn_13*hmn_off_diag_3_1*hmn_diag_1
        + 8.0*_s1sq*_s4sq*imp_smzk_prop_feyn_14*hmn_off_diag_4_1*hmn_diag_1
        + 8.0*_s1sq*_s2sq*_s3sq*imp_smzk_prop_feyn_23*hmn_off_diag_2_1*hmn_off_diag_3_1
        + 8.0*_s1sq*_s2sq*_s4sq*imp_smzk_prop_feyn_24*hmn_off_diag_2_1*hmn_off_diag_4_1
        + 8.0*_s1sq*_s3sq*_s4sq*imp_smzk_prop_feyn_34*hmn_off_diag_3_1*hmn_off_diag_4_1
    pmu_feyn_1 *= 4.0*sum_ssq
    pmu_feyn_2 =
        imp_smzk_prop_feyn_22 * hmn_diag_2_sq
        + _s2sq*_s1sq*imp_smzk_prop_feyn_11*hmn_off_diag_1_2*hmn_off_diag_1_2
        + _s2sq*_s3sq*imp_smzk_prop_feyn_33*hmn_off_diag_3_2*hmn_off_diag_3_2
        + _s2sq*_s4sq*imp_smzk_prop_feyn_44*hmn_off_diag_4_2*hmn_off_diag_4_2
        + 8.0*_s2sq*_s1sq*imp_smzk_prop_feyn_21*hmn_off_diag_1_2*hmn_diag_2
        + 8.0*_s2sq*_s3sq*imp_smzk_prop_feyn_23*hmn_off_diag_3_2*hmn_diag_2
        + 8.0*_s2sq*_s4sq*imp_smzk_prop_feyn_24*hmn_off_diag_4_2*hmn_diag_2
        + 8.0*_s2sq*_s1sq*_s3sq*imp_smzk_prop_feyn_13*hmn_off_diag_1_2*hmn_off_diag_3_2
        + 8.0*_s2sq*_s1sq*_s4sq*imp_smzk_prop_feyn_14*hmn_off_diag_1_2*hmn_off_diag_4_2
        + 8.0*_s2sq*_s3sq*_s4sq*imp_smzk_prop_feyn_34*hmn_off_diag_3_2*hmn_off_diag_4_2
    pmu_feyn_2 *= 4.0*sum_ssq
    pmu_feyn_3 =
        imp_smzk_prop_feyn_33 * hmn_diag_3_sq
        + _s3sq*_s1sq*imp_smzk_prop_feyn_11*hmn_off_diag_1_3*hmn_off_diag_1_3
        + _s3sq*_s2sq*imp_smzk_prop_feyn_22*hmn_off_diag_2_3*hmn_off_diag_2_3
        + _s3sq*_s4sq*imp_smzk_prop_feyn_44*hmn_off_diag_4_3*hmn_off_diag_4_3
        + 8.0*_s3sq*_s1sq*imp_smzk_prop_feyn_31*hmn_off_diag_1_3*hmn_diag_3
        + 8.0*_s3sq*_s2sq*imp_smzk_prop_feyn_32*hmn_off_diag_2_3*hmn_diag_3
        + 8.0*_s3sq*_s4sq*imp_smzk_prop_feyn_34*hmn_off_diag_4_3*hmn_diag_3
        + 8.0*_s3sq*_s1sq*_s2sq*imp_smzk_prop_feyn_12*hmn_off_diag_1_3*hmn_off_diag_2_3
        + 8.0*_s3sq*_s1sq*_s4sq*imp_smzk_prop_feyn_14*hmn_off_diag_1_3*hmn_off_diag_4_3
        + 8.0*_s3sq*_s2sq*_s4sq*imp_smzk_prop_feyn_24*hmn_off_diag_2_3*hmn_off_diag_4_3
    pmu_feyn_3 *= 4.0*sum_ssq
    pmu_feyn_4 =
        imp_smzk_prop_feyn_44 * hmn_diag_4_sq
        + _s4sq*_s1sq*imp_smzk_prop_feyn_11*hmn_off_diag_1_4*hmn_off_diag_1_4
        + _s4sq*_s2sq*imp_smzk_prop_feyn_22*hmn_off_diag_2_4*hmn_off_diag_2_4
        + _s4sq*_s3sq*imp_smzk_prop_feyn_33*hmn_off_diag_3_4*hmn_off_diag_3_4
        + 8.0*_s4sq*_s1sq*imp_smzk_prop_feyn_41*hmn_off_diag_1_4*hmn_diag_4
        + 8.0*_s4sq*_s2sq*imp_smzk_prop_feyn_42*hmn_off_diag_2_4*hmn_diag_4
        + 8.0*_s4sq*_s3sq*imp_smzk_prop_feyn_43*hmn_off_diag_3_4*hmn_diag_4
        + 8.0*_s4sq*_s1sq*_s2sq*imp_smzk_prop_feyn_12*hmn_off_diag_1_4*hmn_off_diag_2_4
        + 8.0*_s4sq*_s1sq*_s3sq*imp_smzk_prop_feyn_13*hmn_off_diag_1_4*hmn_off_diag_3_4
        + 8.0*_s4sq*_s2sq*_s3sq*imp_smzk_prop_feyn_23*hmn_off_diag_2_4*hmn_off_diag_3_4
    pmu_feyn_4 *= 4.0*sum_ssq

    pmu_land_1 =
        imp_smzk_prop_land_11 * hmn_diag_1_sq
        + _s1sq*_s2sq*imp_smzk_prop_land_22*hmn_off_diag_2_1*hmn_off_diag_2_1
        + _s1sq*_s3sq*imp_smzk_prop_land_33*hmn_off_diag_3_1*hmn_off_diag_3_1
        + _s1sq*_s4sq*imp_smzk_prop_land_44*hmn_off_diag_4_1*hmn_off_diag_4_1
        + 8.0*_s1sq*_s2sq*imp_smzk_prop_land_12*hmn_off_diag_2_1*hmn_diag_1
        + 8.0*_s1sq*_s3sq*imp_smzk_prop_land_13*hmn_off_diag_3_1*hmn_diag_1
        + 8.0*_s1sq*_s4sq*imp_smzk_prop_land_14*hmn_off_diag_4_1*hmn_diag_1
        + 8.0*_s1sq*_s2sq*_s3sq*imp_smzk_prop_land_23*hmn_off_diag_2_1*hmn_off_diag_3_1
        + 8.0*_s1sq*_s2sq*_s4sq*imp_smzk_prop_land_24*hmn_off_diag_2_1*hmn_off_diag_4_1
        + 8.0*_s1sq*_s3sq*_s4sq*imp_smzk_prop_land_34*hmn_off_diag_3_1*hmn_off_diag_4_1
    pmu_land_1 *= 4.0*sum_ssq
    pmu_land_2 =
        imp_smzk_prop_land_22 * hmn_diag_2_sq
        + _s2sq*_s1sq*imp_smzk_prop_land_11*hmn_off_diag_1_2*hmn_off_diag_1_2
        + _s2sq*_s3sq*imp_smzk_prop_land_33*hmn_off_diag_3_2*hmn_off_diag_3_2
        + _s2sq*_s4sq*imp_smzk_prop_land_44*hmn_off_diag_4_2*hmn_off_diag_4_2
        + 8.0*_s2sq*_s1sq*imp_smzk_prop_land_21*hmn_off_diag_1_2*hmn_diag_2
        + 8.0*_s2sq*_s3sq*imp_smzk_prop_land_23*hmn_off_diag_3_2*hmn_diag_2
        + 8.0*_s2sq*_s4sq*imp_smzk_prop_land_24*hmn_off_diag_4_2*hmn_diag_2
        + 8.0*_s2sq*_s1sq*_s3sq*imp_smzk_prop_land_13*hmn_off_diag_1_2*hmn_off_diag_3_2
        + 8.0*_s2sq*_s1sq*_s4sq*imp_smzk_prop_land_14*hmn_off_diag_1_2*hmn_off_diag_4_2
        + 8.0*_s2sq*_s3sq*_s4sq*imp_smzk_prop_land_34*hmn_off_diag_3_2*hmn_off_diag_4_2
    pmu_land_2 *= 4.0*sum_ssq
    pmu_land_3 =
        imp_smzk_prop_land_33 * hmn_diag_3_sq
        + _s3sq*_s1sq*imp_smzk_prop_land_11*hmn_off_diag_1_3*hmn_off_diag_1_3
        + _s3sq*_s2sq*imp_smzk_prop_land_22*hmn_off_diag_2_3*hmn_off_diag_2_3
        + _s3sq*_s4sq*imp_smzk_prop_land_44*hmn_off_diag_4_3*hmn_off_diag_4_3
        + 8.0*_s3sq*_s1sq*imp_smzk_prop_land_31*hmn_off_diag_1_3*hmn_diag_3
        + 8.0*_s3sq*_s2sq*imp_smzk_prop_land_32*hmn_off_diag_2_3*hmn_diag_3
        + 8.0*_s3sq*_s4sq*imp_smzk_prop_land_34*hmn_off_diag_4_3*hmn_diag_3
        + 8.0*_s3sq*_s1sq*_s2sq*imp_smzk_prop_land_12*hmn_off_diag_1_3*hmn_off_diag_2_3
        + 8.0*_s3sq*_s1sq*_s4sq*imp_smzk_prop_land_14*hmn_off_diag_1_3*hmn_off_diag_4_3
        + 8.0*_s3sq*_s2sq*_s4sq*imp_smzk_prop_land_24*hmn_off_diag_2_3*hmn_off_diag_4_3
    pmu_land_3 *= 4.0*sum_ssq
    pmu_land_4 =
        imp_smzk_prop_land_44 * hmn_diag_4_sq
        + _s4sq*_s1sq*imp_smzk_prop_land_11*hmn_off_diag_1_4*hmn_off_diag_1_4
        + _s4sq*_s2sq*imp_smzk_prop_land_22*hmn_off_diag_2_4*hmn_off_diag_2_4
        + _s4sq*_s3sq*imp_smzk_prop_land_33*hmn_off_diag_3_4*hmn_off_diag_3_4
        + 8.0*_s4sq*_s1sq*imp_smzk_prop_land_41*hmn_off_diag_1_4*hmn_diag_4
        + 8.0*_s4sq*_s2sq*imp_smzk_prop_land_42*hmn_off_diag_2_4*hmn_diag_4
        + 8.0*_s4sq*_s3sq*imp_smzk_prop_land_43*hmn_off_diag_3_4*hmn_diag_4
        + 8.0*_s4sq*_s1sq*_s2sq*imp_smzk_prop_land_12*hmn_off_diag_1_4*hmn_off_diag_2_4
        + 8.0*_s4sq*_s1sq*_s3sq*imp_smzk_prop_land_13*hmn_off_diag_1_4*hmn_off_diag_3_4
        + 8.0*_s4sq*_s2sq*_s3sq*imp_smzk_prop_land_23*hmn_off_diag_2_4*hmn_off_diag_3_4
    pmu_land_4 *= 4.0*sum_ssq

    sum_pmu_feyn = pmu_feyn_1 + pmu_feyn_2 + pmu_feyn_3 + pmu_feyn_4 
    sum_s2_pmu_feyn =
        _s1sq*pmu_feyn_1 + _s2sq*pmu_feyn_2 + _s3sq*pmu_feyn_3 + _s4sq*pmu_feyn_4 
    sum_s4_pmu_feyn =
        _s14t*pmu_feyn_1 + _s24t*pmu_feyn_2 + _s34t*pmu_feyn_3 + _s44t*pmu_feyn_4 
    sum_s6_pmu_feyn =
        _s16t*pmu_feyn_1 + _s26t*pmu_feyn_2 + _s36t*pmu_feyn_3 + _s46t*pmu_feyn_4 
    sum_s8_pmu_feyn =
        _s18t*pmu_feyn_1 + _s28t*pmu_feyn_2 + _s38t*pmu_feyn_3 + _s48t*pmu_feyn_4 
    sum_pmu_land = pmu_land_1 + pmu_land_2 + pmu_land_3 + pmu_land_4 
    sum_s2_pmu_land =
        _s1sq*pmu_land_1 + _s2sq*pmu_land_2 + _s3sq*pmu_land_3 + _s4sq*pmu_land_4 
    sum_s4_pmu_land =
        _s14t*pmu_land_1 + _s24t*pmu_land_2 + _s34t*pmu_land_3 + _s44t*pmu_land_4 
    sum_s6_pmu_land =
        _s16t*pmu_land_1 + _s26t*pmu_land_2 + _s36t*pmu_land_3 + _s46t*pmu_land_4 
    sum_s8_pmu_land =
        _s18t*pmu_land_1 + _s28t*pmu_land_2 + _s38t*pmu_land_3 + _s48t*pmu_land_4   

    # ------------------------------------------------------------
    # Off-diagonal part of smeared-smeared propagator
    # ------------------------------------------------------------
    
    omunu_feyn_12 = 0.0
    omunu_feyn_12 =
        imp_smzk_prop_feyn_11*hmn_off_diag_1_2*hmn_diag_1
        + imp_smzk_prop_feyn_22*hmn_off_diag_2_1*hmn_diag_2
        + 4.0*imp_smzk_prop_feyn_12*hmn_diag_1*hmn_diag_2
        + _s3sq*imp_smzk_prop_feyn_33*hmn_off_diag_3_1*hmn_off_diag_3_2
        + _s4sq*imp_smzk_prop_feyn_44*hmn_off_diag_4_1*hmn_off_diag_4_2
        + 4.0*_s3sq*imp_smzk_prop_feyn_13*hmn_off_diag_3_2*hmn_diag_1
        + 4.0*_s3sq*imp_smzk_prop_feyn_32*hmn_off_diag_3_1*hmn_diag_2
        + 4.0*_s4sq*imp_smzk_prop_feyn_14*hmn_off_diag_4_2*hmn_diag_1
        + 4.0*_s4sq*imp_smzk_prop_feyn_42*hmn_off_diag_4_1*hmn_diag_2
        + 4.0*_s2sq*_s1sq*imp_smzk_prop_feyn_21*hmn_off_diag_2_1*hmn_off_diag_1_2
        + 4.0*_s2sq*_s3sq*imp_smzk_prop_feyn_23*hmn_off_diag_2_1*hmn_off_diag_3_2
        + 4.0*_s2sq*_s4sq*imp_smzk_prop_feyn_24*hmn_off_diag_2_1*hmn_off_diag_4_2
        + 4.0*_s3sq*_s1sq*imp_smzk_prop_feyn_31*hmn_off_diag_3_1*hmn_off_diag_1_2
        + 4.0*_s3sq*_s4sq*imp_smzk_prop_feyn_34*hmn_off_diag_3_1*hmn_off_diag_4_2
        + 4.0*_s4sq*_s1sq*imp_smzk_prop_feyn_41*hmn_off_diag_4_1*hmn_off_diag_1_2
        + 4.0*_s4sq*_s3sq*imp_smzk_prop_feyn_43*hmn_off_diag_4_1*hmn_off_diag_3_2
    omunu_feyn_12 *= sum_ssq

    # ------------------------------------------------------------
    # Caution! I have to check below later with my hand
    # ------------------------------------------------------------
    
    omunu_feyn_13 = 0.0
    omunu_feyn_13 =
        imp_smzk_prop_feyn_11*hmn_off_diag_1_3*hmn_diag_1
        + imp_smzk_prop_feyn_33*hmn_off_diag_3_1*hmn_diag_3
        + 4.0*imp_smzk_prop_feyn_13*hmn_diag_1*hmn_diag_3
        + _s2sq*imp_smzk_prop_feyn_22*hmn_off_diag_2_1*hmn_off_diag_2_3
        + _s4sq*imp_smzk_prop_feyn_44*hmn_off_diag_4_1*hmn_off_diag_4_3
        + 4.0*_s2sq*imp_smzk_prop_feyn_12*hmn_off_diag_2_3*hmn_diag_1
        + 4.0*_s2sq*imp_smzk_prop_feyn_23*hmn_off_diag_2_1*hmn_diag_3
        + 4.0*_s4sq*imp_smzk_prop_feyn_14*hmn_off_diag_4_3*hmn_diag_1
        + 4.0*_s4sq*imp_smzk_prop_feyn_43*hmn_off_diag_4_1*hmn_diag_3
        + 4.0*_s3sq*_s1sq*imp_smzk_prop_feyn_31*hmn_off_diag_3_1*hmn_off_diag_1_3
        + 4.0*_s3sq*_s2sq*imp_smzk_prop_feyn_32*hmn_off_diag_3_1*hmn_off_diag_2_3
        + 4.0*_s3sq*_s4sq*imp_smzk_prop_feyn_34*hmn_off_diag_3_1*hmn_off_diag_4_3
        + 4.0*_s2sq*_s1sq*imp_smzk_prop_feyn_21*hmn_off_diag_2_1*hmn_off_diag_1_3
        + 4.0*_s2sq*_s4sq*imp_smzk_prop_feyn_24*hmn_off_diag_2_1*hmn_off_diag_4_3
        + 4.0*_s4sq*_s1sq*imp_smzk_prop_feyn_41*hmn_off_diag_4_1*hmn_off_diag_1_3
        + 4.0*_s4sq*_s2sq*imp_smzk_prop_feyn_42*hmn_off_diag_4_1*hmn_off_diag_2_3
    omunu_feyn_13 *= sum_ssq

    omunu_feyn_14 = 0.0
    omunu_feyn_14 =
        imp_smzk_prop_feyn_11*hmn_off_diag_1_4*hmn_diag_1
        + imp_smzk_prop_feyn_44*hmn_off_diag_4_1*hmn_diag_4
        + 4.0*imp_smzk_prop_feyn_14*hmn_diag_1*hmn_diag_4
        + _s2sq*imp_smzk_prop_feyn_22*hmn_off_diag_2_1*hmn_off_diag_2_4
        + _s3sq*imp_smzk_prop_feyn_33*hmn_off_diag_3_1*hmn_off_diag_3_4
        + 4.0*_s2sq*imp_smzk_prop_feyn_12*hmn_off_diag_2_4*hmn_diag_1
        + 4.0*_s2sq*imp_smzk_prop_feyn_24*hmn_off_diag_2_1*hmn_diag_4
        + 4.0*_s3sq*imp_smzk_prop_feyn_13*hmn_off_diag_3_4*hmn_diag_1
        + 4.0*_s3sq*imp_smzk_prop_feyn_34*hmn_off_diag_3_1*hmn_diag_4    
        + 4.0*_s4sq*_s1sq*imp_smzk_prop_feyn_41*hmn_off_diag_4_1*hmn_off_diag_1_4
        + 4.0*_s4sq*_s2sq*imp_smzk_prop_feyn_42*hmn_off_diag_4_1*hmn_off_diag_2_4
        + 4.0*_s4sq*_s3sq*imp_smzk_prop_feyn_43*hmn_off_diag_4_1*hmn_off_diag_3_4
        + 4.0*_s2sq*_s1sq*imp_smzk_prop_feyn_21*hmn_off_diag_2_1*hmn_off_diag_1_4
        + 4.0*_s2sq*_s3sq*imp_smzk_prop_feyn_23*hmn_off_diag_2_1*hmn_off_diag_3_4
        + 4.0*_s3sq*_s1sq*imp_smzk_prop_feyn_31*hmn_off_diag_3_1*hmn_off_diag_1_4
        + 4.0*_s3sq*_s2sq*imp_smzk_prop_feyn_32*hmn_off_diag_3_1*hmn_off_diag_2_4
    omunu_feyn_14 *= sum_ssq

    omunu_feyn_23 = 0.0
    omunu_feyn_23 =
        imp_smzk_prop_feyn_22*hmn_off_diag_2_3*hmn_diag_2
        + imp_smzk_prop_feyn_33*hmn_off_diag_3_2*hmn_diag_3
        + 4.0*imp_smzk_prop_feyn_23*hmn_diag_2*hmn_diag_3
        + _s1sq*imp_smzk_prop_feyn_11*hmn_off_diag_1_2*hmn_off_diag_1_3
        + _s4sq*imp_smzk_prop_feyn_44*hmn_off_diag_4_2*hmn_off_diag_4_3
        + 4.0*_s1sq*imp_smzk_prop_feyn_21*hmn_off_diag_1_3*hmn_diag_2
        + 4.0*_s1sq*imp_smzk_prop_feyn_13*hmn_off_diag_1_2*hmn_diag_3
        + 4.0*_s4sq*imp_smzk_prop_feyn_24*hmn_off_diag_4_3*hmn_diag_2
        + 4.0*_s4sq*imp_smzk_prop_feyn_43*hmn_off_diag_4_2*hmn_diag_3
        + 4.0*_s3sq*_s2sq*imp_smzk_prop_feyn_32*hmn_off_diag_3_2*hmn_off_diag_2_3
        + 4.0*_s3sq*_s1sq*imp_smzk_prop_feyn_31*hmn_off_diag_3_2*hmn_off_diag_1_3
        + 4.0*_s3sq*_s4sq*imp_smzk_prop_feyn_34*hmn_off_diag_3_2*hmn_off_diag_4_3
        + 4.0*_s1sq*_s2sq*imp_smzk_prop_feyn_12*hmn_off_diag_1_2*hmn_off_diag_2_3
        + 4.0*_s1sq*_s4sq*imp_smzk_prop_feyn_14*hmn_off_diag_1_2*hmn_off_diag_4_3
        + 4.0*_s4sq*_s2sq*imp_smzk_prop_feyn_42*hmn_off_diag_4_2*hmn_off_diag_2_3
        + 4.0*_s4sq*_s1sq*imp_smzk_prop_feyn_41*hmn_off_diag_4_2*hmn_off_diag_1_3
    omunu_feyn_23 *= sum_ssq

    omunu_feyn_24 = 0.0
    omunu_feyn_24 =
        imp_smzk_prop_feyn_22*hmn_off_diag_2_4*hmn_diag_2
        + imp_smzk_prop_feyn_44*hmn_off_diag_4_2*hmn_diag_4
        + 4.0*imp_smzk_prop_feyn_24*hmn_diag_2*hmn_diag_4
        + _s1sq*imp_smzk_prop_feyn_11*hmn_off_diag_1_2*hmn_off_diag_1_4
        + _s3sq*imp_smzk_prop_feyn_33*hmn_off_diag_3_2*hmn_off_diag_3_4
        + 4.0*_s1sq*imp_smzk_prop_feyn_21*hmn_off_diag_1_4*hmn_diag_2
        + 4.0*_s1sq*imp_smzk_prop_feyn_14*hmn_off_diag_1_2*hmn_diag_4
        + 4.0*_s3sq*imp_smzk_prop_feyn_23*hmn_off_diag_3_4*hmn_diag_2
        + 4.0*_s3sq*imp_smzk_prop_feyn_34*hmn_off_diag_3_2*hmn_diag_4
        + 4.0*_s4sq*_s2sq*imp_smzk_prop_feyn_42*hmn_off_diag_4_2*hmn_off_diag_2_4
        + 4.0*_s4sq*_s1sq*imp_smzk_prop_feyn_41*hmn_off_diag_4_2*hmn_off_diag_1_4
        + 4.0*_s4sq*_s3sq*imp_smzk_prop_feyn_43*hmn_off_diag_4_2*hmn_off_diag_3_4
        + 4.0*_s1sq*_s2sq*imp_smzk_prop_feyn_12*hmn_off_diag_1_2*hmn_off_diag_2_4
        + 4.0*_s1sq*_s3sq*imp_smzk_prop_feyn_13*hmn_off_diag_1_2*hmn_off_diag_3_4
        + 4.0*_s3sq*_s2sq*imp_smzk_prop_feyn_32*hmn_off_diag_3_2*hmn_off_diag_2_4
        + 4.0*_s3sq*_s1sq*imp_smzk_prop_feyn_31*hmn_off_diag_3_2*hmn_off_diag_1_4
    omunu_feyn_24 *= sum_ssq

    omunu_feyn_34 = 0.0
    omunu_feyn_34 =
        imp_smzk_prop_feyn_33*hmn_off_diag_3_4*hmn_diag_3
        + imp_smzk_prop_feyn_44*hmn_off_diag_4_3*hmn_diag_4
        + 4.0*imp_smzk_prop_feyn_34*hmn_diag_3*hmn_diag_4
        + _s1sq*imp_smzk_prop_feyn_11*hmn_off_diag_1_3*hmn_off_diag_1_4
        + _s2sq*imp_smzk_prop_feyn_22*hmn_off_diag_2_3*hmn_off_diag_2_4
        + 4.0*_s1sq*imp_smzk_prop_feyn_31*hmn_off_diag_1_4*hmn_diag_3
        + 4.0*_s1sq*imp_smzk_prop_feyn_14*hmn_off_diag_1_3*hmn_diag_4
        + 4.0*_s2sq*imp_smzk_prop_feyn_32*hmn_off_diag_2_4*hmn_diag_3
        + 4.0*_s2sq*imp_smzk_prop_feyn_24*hmn_off_diag_2_3*hmn_diag_4
        + 4.0*_s4sq*_s3sq*imp_smzk_prop_feyn_43*hmn_off_diag_4_3*hmn_off_diag_3_4
        + 4.0*_s4sq*_s1sq*imp_smzk_prop_feyn_41*hmn_off_diag_4_3*hmn_off_diag_1_4
        + 4.0*_s4sq*_s2sq*imp_smzk_prop_feyn_42*hmn_off_diag_4_3*hmn_off_diag_2_4
        + 4.0*_s1sq*_s3sq*imp_smzk_prop_feyn_13*hmn_off_diag_1_3*hmn_off_diag_3_4
        + 4.0*_s1sq*_s2sq*imp_smzk_prop_feyn_12*hmn_off_diag_1_3*hmn_off_diag_2_4
        + 4.0*_s2sq*_s3sq*imp_smzk_prop_feyn_23*hmn_off_diag_2_3*hmn_off_diag_3_4
        + 4.0*_s2sq*_s1sq*imp_smzk_prop_feyn_21*hmn_off_diag_2_3*hmn_off_diag_1_4
    omunu_feyn_34 *= sum_ssq

    omunu_feyn_21 = omunu_feyn_12
    omunu_feyn_31 = omunu_feyn_13
    omunu_feyn_41 = omunu_feyn_14
    omunu_feyn_32 = omunu_feyn_23
    omunu_feyn_42 = omunu_feyn_24
    omunu_feyn_43 = omunu_feyn_34


    omunu_land_12 = 0.0
    omunu_land_12 =
        imp_smzk_prop_land_11*hmn_off_diag_1_2*hmn_diag_1
        + imp_smzk_prop_land_22*hmn_off_diag_2_1*hmn_diag_2
        + 4.0*imp_smzk_prop_land_12*hmn_diag_1*hmn_diag_2
        + _s3sq*imp_smzk_prop_land_33*hmn_off_diag_3_1*hmn_off_diag_3_2
        + _s4sq*imp_smzk_prop_land_44*hmn_off_diag_4_1*hmn_off_diag_4_2
        + 4.0*_s3sq*imp_smzk_prop_land_13*hmn_off_diag_3_2*hmn_diag_1
        + 4.0*_s3sq*imp_smzk_prop_land_32*hmn_off_diag_3_1*hmn_diag_2
        + 4.0*_s4sq*imp_smzk_prop_land_14*hmn_off_diag_4_2*hmn_diag_1
        + 4.0*_s4sq*imp_smzk_prop_land_42*hmn_off_diag_4_1*hmn_diag_2
        + 4.0*_s2sq*_s1sq*imp_smzk_prop_land_21*hmn_off_diag_2_1*hmn_off_diag_1_2
        + 4.0*_s2sq*_s3sq*imp_smzk_prop_land_23*hmn_off_diag_2_1*hmn_off_diag_3_2
        + 4.0*_s2sq*_s4sq*imp_smzk_prop_land_24*hmn_off_diag_2_1*hmn_off_diag_4_2
        + 4.0*_s3sq*_s1sq*imp_smzk_prop_land_31*hmn_off_diag_3_1*hmn_off_diag_1_2
        + 4.0*_s3sq*_s4sq*imp_smzk_prop_land_34*hmn_off_diag_3_1*hmn_off_diag_4_2
        + 4.0*_s4sq*_s1sq*imp_smzk_prop_land_41*hmn_off_diag_4_1*hmn_off_diag_1_2
        + 4.0*_s4sq*_s3sq*imp_smzk_prop_land_43*hmn_off_diag_4_1*hmn_off_diag_3_2
    omunu_land_12 *= sum_ssq

    # ------------------------------------------------------------
    # Caution! I have to check below later with my hand
    # ------------------------------------------------------------
    
    omunu_land_13 = 0.0
    omunu_land_13 =
        imp_smzk_prop_land_11*hmn_off_diag_1_3*hmn_diag_1
        + imp_smzk_prop_land_33*hmn_off_diag_3_1*hmn_diag_3
        + 4.0*imp_smzk_prop_land_13*hmn_diag_1*hmn_diag_3
        + _s2sq*imp_smzk_prop_land_22*hmn_off_diag_2_1*hmn_off_diag_2_3
        + _s4sq*imp_smzk_prop_land_44*hmn_off_diag_4_1*hmn_off_diag_4_3
        + 4.0*_s2sq*imp_smzk_prop_land_12*hmn_off_diag_2_3*hmn_diag_1
        + 4.0*_s2sq*imp_smzk_prop_land_23*hmn_off_diag_2_1*hmn_diag_3
        + 4.0*_s4sq*imp_smzk_prop_land_14*hmn_off_diag_4_3*hmn_diag_1
        + 4.0*_s4sq*imp_smzk_prop_land_43*hmn_off_diag_4_1*hmn_diag_3
        + 4.0*_s3sq*_s1sq*imp_smzk_prop_land_31*hmn_off_diag_3_1*hmn_off_diag_1_3
        + 4.0*_s3sq*_s2sq*imp_smzk_prop_land_32*hmn_off_diag_3_1*hmn_off_diag_2_3
        + 4.0*_s3sq*_s4sq*imp_smzk_prop_land_34*hmn_off_diag_3_1*hmn_off_diag_4_3
        + 4.0*_s2sq*_s1sq*imp_smzk_prop_land_21*hmn_off_diag_2_1*hmn_off_diag_1_3
        + 4.0*_s2sq*_s4sq*imp_smzk_prop_land_24*hmn_off_diag_2_1*hmn_off_diag_4_3
        + 4.0*_s4sq*_s1sq*imp_smzk_prop_land_41*hmn_off_diag_4_1*hmn_off_diag_1_3
        + 4.0*_s4sq*_s2sq*imp_smzk_prop_land_42*hmn_off_diag_4_1*hmn_off_diag_2_3
    omunu_land_13 *= sum_ssq

    omunu_land_14 = 0.0
    omunu_land_14 =
        imp_smzk_prop_land_11*hmn_off_diag_1_4*hmn_diag_1
        + imp_smzk_prop_land_44*hmn_off_diag_4_1*hmn_diag_4
        + 4.0*imp_smzk_prop_land_14*hmn_diag_1*hmn_diag_4
        + _s2sq*imp_smzk_prop_land_22*hmn_off_diag_2_1*hmn_off_diag_2_4
        + _s3sq*imp_smzk_prop_land_33*hmn_off_diag_3_1*hmn_off_diag_3_4
        + 4.0*_s2sq*imp_smzk_prop_land_12*hmn_off_diag_2_4*hmn_diag_1
        + 4.0*_s2sq*imp_smzk_prop_land_24*hmn_off_diag_2_1*hmn_diag_4
        + 4.0*_s3sq*imp_smzk_prop_land_13*hmn_off_diag_3_4*hmn_diag_1
        + 4.0*_s3sq*imp_smzk_prop_land_34*hmn_off_diag_3_1*hmn_diag_4
        + 4.0*_s4sq*_s1sq*imp_smzk_prop_land_41*hmn_off_diag_4_1*hmn_off_diag_1_4
        + 4.0*_s4sq*_s2sq*imp_smzk_prop_land_42*hmn_off_diag_4_1*hmn_off_diag_2_4
        + 4.0*_s4sq*_s3sq*imp_smzk_prop_land_43*hmn_off_diag_4_1*hmn_off_diag_3_4
        + 4.0*_s2sq*_s1sq*imp_smzk_prop_land_21*hmn_off_diag_2_1*hmn_off_diag_1_4
        + 4.0*_s2sq*_s3sq*imp_smzk_prop_land_23*hmn_off_diag_2_1*hmn_off_diag_3_4
        + 4.0*_s3sq*_s1sq*imp_smzk_prop_land_31*hmn_off_diag_3_1*hmn_off_diag_1_4
        + 4.0*_s3sq*_s2sq*imp_smzk_prop_land_32*hmn_off_diag_3_1*hmn_off_diag_2_4
    omunu_land_14 *= sum_ssq

    omunu_land_23 = 0.0
    omunu_land_23 =
        imp_smzk_prop_land_22*hmn_off_diag_2_3*hmn_diag_2
        + imp_smzk_prop_land_33*hmn_off_diag_3_2*hmn_diag_3
        + 4.0*imp_smzk_prop_land_23*hmn_diag_2*hmn_diag_3
        + _s1sq*imp_smzk_prop_land_11*hmn_off_diag_1_2*hmn_off_diag_1_3
        + _s4sq*imp_smzk_prop_land_44*hmn_off_diag_4_2*hmn_off_diag_4_3
        + 4.0*_s1sq*imp_smzk_prop_land_21*hmn_off_diag_1_3*hmn_diag_2
        + 4.0*_s1sq*imp_smzk_prop_land_13*hmn_off_diag_1_2*hmn_diag_3
        + 4.0*_s4sq*imp_smzk_prop_land_24*hmn_off_diag_4_3*hmn_diag_2
        + 4.0*_s4sq*imp_smzk_prop_land_43*hmn_off_diag_4_2*hmn_diag_3
        + 4.0*_s3sq*_s2sq*imp_smzk_prop_land_32*hmn_off_diag_3_2*hmn_off_diag_2_3
        + 4.0*_s3sq*_s1sq*imp_smzk_prop_land_31*hmn_off_diag_3_2*hmn_off_diag_1_3
        + 4.0*_s3sq*_s4sq*imp_smzk_prop_land_34*hmn_off_diag_3_2*hmn_off_diag_4_3
        + 4.0*_s1sq*_s2sq*imp_smzk_prop_land_12*hmn_off_diag_1_2*hmn_off_diag_2_3
        + 4.0*_s1sq*_s4sq*imp_smzk_prop_land_14*hmn_off_diag_1_2*hmn_off_diag_4_3
        + 4.0*_s4sq*_s2sq*imp_smzk_prop_land_42*hmn_off_diag_4_2*hmn_off_diag_2_3
        + 4.0*_s4sq*_s1sq*imp_smzk_prop_land_41*hmn_off_diag_4_2*hmn_off_diag_1_3
    omunu_land_23 *= sum_ssq

    omunu_land_24 = 0.0
    omunu_land_24 =
        imp_smzk_prop_land_22*hmn_off_diag_2_4*hmn_diag_2
        + imp_smzk_prop_land_44*hmn_off_diag_4_2*hmn_diag_4
        + 4.0*imp_smzk_prop_land_24*hmn_diag_2*hmn_diag_4
        + _s1sq*imp_smzk_prop_land_11*hmn_off_diag_1_2*hmn_off_diag_1_4
        + _s3sq*imp_smzk_prop_land_33*hmn_off_diag_3_2*hmn_off_diag_3_4
        + 4.0*_s1sq*imp_smzk_prop_land_21*hmn_off_diag_1_4*hmn_diag_2
        + 4.0*_s1sq*imp_smzk_prop_land_14*hmn_off_diag_1_2*hmn_diag_4
        + 4.0*_s3sq*imp_smzk_prop_land_23*hmn_off_diag_3_4*hmn_diag_2
        + 4.0*_s3sq*imp_smzk_prop_land_34*hmn_off_diag_3_2*hmn_diag_4
        + 4.0*_s4sq*_s2sq*imp_smzk_prop_land_42*hmn_off_diag_4_2*hmn_off_diag_2_4
        + 4.0*_s4sq*_s1sq*imp_smzk_prop_land_41*hmn_off_diag_4_2*hmn_off_diag_1_4
        + 4.0*_s4sq*_s3sq*imp_smzk_prop_land_43*hmn_off_diag_4_2*hmn_off_diag_3_4
        + 4.0*_s1sq*_s2sq*imp_smzk_prop_land_12*hmn_off_diag_1_2*hmn_off_diag_2_4
        + 4.0*_s1sq*_s3sq*imp_smzk_prop_land_13*hmn_off_diag_1_2*hmn_off_diag_3_4
        + 4.0*_s3sq*_s2sq*imp_smzk_prop_land_32*hmn_off_diag_3_2*hmn_off_diag_2_4
        + 4.0*_s3sq*_s1sq*imp_smzk_prop_land_31*hmn_off_diag_3_2*hmn_off_diag_1_4
    omunu_land_24 *= sum_ssq

    omunu_land_34 = 0.0
    omunu_land_34 =
        imp_smzk_prop_land_33*hmn_off_diag_3_4*hmn_diag_3
        + imp_smzk_prop_land_44*hmn_off_diag_4_3*hmn_diag_4
        + 4.0*imp_smzk_prop_land_34*hmn_diag_3*hmn_diag_4
        + _s1sq*imp_smzk_prop_land_11*hmn_off_diag_1_3*hmn_off_diag_1_4
        + _s2sq*imp_smzk_prop_land_22*hmn_off_diag_2_3*hmn_off_diag_2_4
        + 4.0*_s1sq*imp_smzk_prop_land_31*hmn_off_diag_1_4*hmn_diag_3
        + 4.0*_s1sq*imp_smzk_prop_land_14*hmn_off_diag_1_3*hmn_diag_4
        + 4.0*_s2sq*imp_smzk_prop_land_32*hmn_off_diag_2_4*hmn_diag_3
        + 4.0*_s2sq*imp_smzk_prop_land_24*hmn_off_diag_2_3*hmn_diag_4
        + 4.0*_s4sq*_s3sq*imp_smzk_prop_land_43*hmn_off_diag_4_3*hmn_off_diag_3_4
        + 4.0*_s4sq*_s1sq*imp_smzk_prop_land_41*hmn_off_diag_4_3*hmn_off_diag_1_4
        + 4.0*_s4sq*_s2sq*imp_smzk_prop_land_42*hmn_off_diag_4_3*hmn_off_diag_2_4
        + 4.0*_s1sq*_s3sq*imp_smzk_prop_land_13*hmn_off_diag_1_3*hmn_off_diag_3_4
        + 4.0*_s1sq*_s2sq*imp_smzk_prop_land_12*hmn_off_diag_1_3*hmn_off_diag_2_4
        + 4.0*_s2sq*_s3sq*imp_smzk_prop_land_23*hmn_off_diag_2_3*hmn_off_diag_3_4
        + 4.0*_s2sq*_s1sq*imp_smzk_prop_land_21*hmn_off_diag_2_3*hmn_off_diag_1_4
    omunu_land_34 *= sum_ssq

    omunu_land_21 = omunu_land_12
    omunu_land_31 = omunu_land_13
    omunu_land_41 = omunu_land_14
    omunu_land_32 = omunu_land_23
    omunu_land_42 = omunu_land_24
    omunu_land_43 = omunu_land_34


    sum_s2_sum_s2_omunu_feyn =
        _s1sq*omunu_feyn_12*_s2sq + _s1sq*omunu_feyn_13*_s3sq + _s1sq*omunu_feyn_14*_s4sq
        + _s2sq*omunu_feyn_21*_s1sq + _s2sq*omunu_feyn_23*_s3sq + _s2sq*omunu_feyn_24*_s4sq
        + _s3sq*omunu_feyn_31*_s1sq + _s3sq*omunu_feyn_32*_s2sq + _s3sq*omunu_feyn_34*_s4sq
        + _s4sq*omunu_feyn_41*_s1sq + _s4sq*omunu_feyn_42*_s2sq + _s4sq*omunu_feyn_43*_s3sq

    sum_s2_sum_s4_omunu_feyn =
        _s1sq*omunu_feyn_12*_s24t + _s1sq*omunu_feyn_13*_s34t + _s1sq*omunu_feyn_14*_s44t
        + _s2sq*omunu_feyn_21*_s14t + _s2sq*omunu_feyn_23*_s34t + _s2sq*omunu_feyn_24*_s44t
        + _s3sq*omunu_feyn_31*_s14t + _s3sq*omunu_feyn_32*_s24t + _s3sq*omunu_feyn_34*_s44t
        + _s4sq*omunu_feyn_41*_s14t + _s4sq*omunu_feyn_42*_s24t + _s4sq*omunu_feyn_43*_s34t
    
    sum_s4_sum_s2_omunu_feyn =
        _s14t*omunu_feyn_12*_s2sq + _s14t*omunu_feyn_13*_s3sq + _s14t*omunu_feyn_14*_s4sq
        + _s24t*omunu_feyn_21*_s1sq + _s24t*omunu_feyn_23*_s3sq + _s24t*omunu_feyn_24*_s4sq
        + _s34t*omunu_feyn_31*_s1sq + _s34t*omunu_feyn_32*_s2sq + _s34t*omunu_feyn_34*_s4sq
        + _s44t*omunu_feyn_41*_s1sq + _s44t*omunu_feyn_42*_s2sq + _s44t*omunu_feyn_43*_s3sq

    sum_s4_sum_s4_omunu_feyn =
        _s14t*omunu_feyn_12*_s24t + _s14t*omunu_feyn_13*_s34t + _s14t*omunu_feyn_14*_s44t
        + _s24t*omunu_feyn_21*_s14t + _s24t*omunu_feyn_23*_s34t + _s24t*omunu_feyn_24*_s44t
        + _s34t*omunu_feyn_31*_s14t + _s34t*omunu_feyn_32*_s24t + _s34t*omunu_feyn_34*_s44t
        + _s44t*omunu_feyn_41*_s14t + _s44t*omunu_feyn_42*_s24t + _s44t*omunu_feyn_43*_s34t

    sum_s6_sum_s2_omunu_feyn =
        _s16t*omunu_feyn_12*_s2sq + _s16t*omunu_feyn_13*_s3sq + _s16t*omunu_feyn_14*_s4sq
        + _s26t*omunu_feyn_21*_s1sq + _s26t*omunu_feyn_23*_s3sq + _s26t*omunu_feyn_24*_s4sq
        + _s36t*omunu_feyn_31*_s1sq + _s36t*omunu_feyn_32*_s2sq + _s36t*omunu_feyn_34*_s4sq
        + _s46t*omunu_feyn_41*_s1sq + _s46t*omunu_feyn_42*_s2sq + _s46t*omunu_feyn_43*_s3sq

    sum_s6_sum_s4_omunu_feyn =
        _s16t*omunu_feyn_12*_s24t + _s16t*omunu_feyn_13*_s34t + _s16t*omunu_feyn_14*_s44t
        + _s26t*omunu_feyn_21*_s14t + _s26t*omunu_feyn_23*_s34t + _s26t*omunu_feyn_24*_s44t
        + _s36t*omunu_feyn_31*_s14t + _s36t*omunu_feyn_32*_s24t + _s36t*omunu_feyn_34*_s44t
        + _s46t*omunu_feyn_41*_s14t + _s46t*omunu_feyn_42*_s24t + _s46t*omunu_feyn_43*_s34t  
    
    sum_s2_sum_s2_omunu_land =
        _s1sq*omunu_land_12*_s2sq + _s1sq*omunu_land_13*_s3sq + _s1sq*omunu_land_14*_s4sq
        + _s2sq*omunu_land_21*_s1sq + _s2sq*omunu_land_23*_s3sq + _s2sq*omunu_land_24*_s4sq
        + _s3sq*omunu_land_31*_s1sq + _s3sq*omunu_land_32*_s2sq + _s3sq*omunu_land_34*_s4sq
        + _s4sq*omunu_land_41*_s1sq + _s4sq*omunu_land_42*_s2sq + _s4sq*omunu_land_43*_s3sq

    sum_s2_sum_s4_omunu_land =
        _s1sq*omunu_land_12*_s24t + _s1sq*omunu_land_13*_s34t + _s1sq*omunu_land_14*_s44t
        + _s2sq*omunu_land_21*_s14t + _s2sq*omunu_land_23*_s34t + _s2sq*omunu_land_24*_s44t
        + _s3sq*omunu_land_31*_s14t + _s3sq*omunu_land_32*_s24t + _s3sq*omunu_land_34*_s44t
        + _s4sq*omunu_land_41*_s14t + _s4sq*omunu_land_42*_s24t + _s4sq*omunu_land_43*_s34t
    
    sum_s4_sum_s2_omunu_land =
        _s14t*omunu_land_12*_s2sq + _s14t*omunu_land_13*_s3sq + _s14t*omunu_land_14*_s4sq
        + _s24t*omunu_land_21*_s1sq + _s24t*omunu_land_23*_s3sq + _s24t*omunu_land_24*_s4sq
        + _s34t*omunu_land_31*_s1sq + _s34t*omunu_land_32*_s2sq + _s34t*omunu_land_34*_s4sq
        + _s44t*omunu_land_41*_s1sq + _s44t*omunu_land_42*_s2sq + _s44t*omunu_land_43*_s3sq

    sum_s4_sum_s4_omunu_land =
        _s14t*omunu_land_12*_s24t + _s14t*omunu_land_13*_s34t + _s14t*omunu_land_14*_s44t
        + _s24t*omunu_land_21*_s14t + _s24t*omunu_land_23*_s34t + _s24t*omunu_land_24*_s44t
        + _s34t*omunu_land_31*_s14t + _s34t*omunu_land_32*_s24t + _s34t*omunu_land_34*_s44t
        + _s44t*omunu_land_41*_s14t + _s44t*omunu_land_42*_s24t + _s44t*omunu_land_43*_s34t

    sum_s6_sum_s2_omunu_land =
        _s16t*omunu_land_12*_s2sq + _s16t*omunu_land_13*_s3sq + _s16t*omunu_land_14*_s4sq
        + _s26t*omunu_land_21*_s1sq + _s26t*omunu_land_23*_s3sq + _s26t*omunu_land_24*_s4sq
        + _s36t*omunu_land_31*_s1sq + _s36t*omunu_land_32*_s2sq + _s36t*omunu_land_34*_s4sq
        + _s46t*omunu_land_41*_s1sq + _s46t*omunu_land_42*_s2sq + _s46t*omunu_land_43*_s3sq

    sum_s6_sum_s4_omunu_land =
        _s16t*omunu_land_12*_s24t + _s16t*omunu_land_13*_s34t + _s16t*omunu_land_14*_s44t
        + _s26t*omunu_land_21*_s14t + _s26t*omunu_land_23*_s34t + _s26t*omunu_land_24*_s44t
        + _s36t*omunu_land_31*_s14t + _s36t*omunu_land_32*_s24t + _s36t*omunu_land_34*_s44t
        + _s46t*omunu_land_41*_s14t + _s46t*omunu_land_42*_s24t + _s46t*omunu_land_43*_s34t

    # ========

    sum_sin_sum_sin_omunu_feyn =
        sin1*omunu_feyn_12*sin2 + sin1*omunu_feyn_13*sin3 + sin1*omunu_feyn_14*sin4
        + sin2*omunu_feyn_21*sin1 + sin2*omunu_feyn_23*sin3 + sin2*omunu_feyn_24*sin4
        + sin3*omunu_feyn_31*sin1 + sin3*omunu_feyn_32*sin2 + sin3*omunu_feyn_34*sin4
        + sin4*omunu_feyn_41*sin1 + sin4*omunu_feyn_42*sin2 + sin4*omunu_feyn_43*sin3

    sum_sin_sum_sin_omunu_land =
        sin1*omunu_land_12*sin2 + sin1*omunu_land_13*sin3 + sin1*omunu_land_14*sin4
        + sin2*omunu_land_21*sin1 + sin2*omunu_land_23*sin3 + sin2*omunu_land_24*sin4
        + sin3*omunu_land_31*sin1 + sin3*omunu_land_32*sin2 + sin3*omunu_land_34*sin4
        + sin4*omunu_land_41*sin1 + sin4*omunu_land_42*sin2 + sin4*omunu_land_43*sin3


    # ------------------------------------------------------------
    # Final Benji Z_F^{finite} + ZT_F
    # ------------------------------------------------------------

    rtn = 0.0
    
    rtn = 0.5*sum_ssq*( sum_pmu_feyn - sum_s2_pmu_feyn ) 
    rtn += -2.0*( sum_s2_pmu_feyn - sum_s4_pmu_feyn )
        -4.0*( sum_s2_sum_s2_omunu_feyn - sum_s2_sum_s4_omunu_feyn )
    rtn +=
        ( - sum_s2_pmu_feyn + 4.0*sum_s4_pmu_feyn
        - 5.0*sum_s6_pmu_feyn + 2.0*sum_s8_pmu_feyn
        - ( sum_s4t - sum_s6t )*( sum_pmu_feyn - sum_s2_pmu_feyn )
        - 4.0*sum_s2_sum_s2_omunu_feyn + 4.0*sum_s2_sum_s4_omunu_feyn
        + 12.0*sum_s4_sum_s2_omunu_feyn - 12.0*sum_s4_sum_s4_omunu_feyn
        - 8.0*sum_s6_sum_s2_omunu_feyn + 8.0*sum_s6_sum_s4_omunu_feyn )*inv_sum_ssq_s4t
    rtn *= inv_sum_ssq*inv_sum_ssq_s4t
    rtn += 1.0*inv_sum_ssq_sq
    
    rtn += 0.5*sum_pmu_feyn*inv_sum_ssq
    rtn *= 1.0/16.0

    measure = 16*π*π / (2*π)^4

    return rtn * measure
end

end  # module Z_q