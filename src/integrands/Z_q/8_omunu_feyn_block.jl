@inline function omunu_feyn_block(
    sum_ssq,
    imp_smzk_prop_feyn_11,imp_smzk_prop_feyn_22,
    imp_smzk_prop_feyn_33,imp_smzk_prop_feyn_44,
    imp_smzk_prop_feyn_12,imp_smzk_prop_feyn_13,imp_smzk_prop_feyn_14,
    imp_smzk_prop_feyn_23,imp_smzk_prop_feyn_24,imp_smzk_prop_feyn_34,
    imp_smzk_prop_feyn_21,imp_smzk_prop_feyn_31,imp_smzk_prop_feyn_41,
    imp_smzk_prop_feyn_32,imp_smzk_prop_feyn_42,imp_smzk_prop_feyn_43,
    hmn_diag_1,hmn_diag_2,hmn_diag_3,hmn_diag_4,
    hmn_off_diag_1_2,hmn_off_diag_1_3,hmn_off_diag_1_4,
    hmn_off_diag_2_1,hmn_off_diag_2_3,hmn_off_diag_2_4,
    hmn_off_diag_3_1,hmn_off_diag_3_2,hmn_off_diag_3_4,
    hmn_off_diag_4_1,hmn_off_diag_4_2,hmn_off_diag_4_3,
    _s1sq,_s2sq,_s3sq,_s4sq
)

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

    return (
    omunu_feyn_12,omunu_feyn_13,omunu_feyn_14,
    omunu_feyn_23,omunu_feyn_24,omunu_feyn_34,
    omunu_feyn_21,omunu_feyn_31,omunu_feyn_41,
    omunu_feyn_32,omunu_feyn_42,omunu_feyn_43
    )
end