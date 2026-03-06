@inline function omunu_land_block(
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

    omunu_land_12 = 0.0

    omunu_land_12 = imp_smzk_prop_land_11*hmn_off_diag_1_2*hmn_diag_1 + imp_smzk_prop_land_22*hmn_off_diag_2_1*hmn_diag_2 + 4.0*imp_smzk_prop_land_12*hmn_diag_1*hmn_diag_2 + _s3sq*imp_smzk_prop_land_33*hmn_off_diag_3_1*hmn_off_diag_3_2 + _s4sq*imp_smzk_prop_land_44*hmn_off_diag_4_1*hmn_off_diag_4_2 + 4.0*_s3sq*imp_smzk_prop_land_13*hmn_off_diag_3_2*hmn_diag_1 + 4.0*_s3sq*imp_smzk_prop_land_32*hmn_off_diag_3_1*hmn_diag_2 + 4.0*_s4sq*imp_smzk_prop_land_14*hmn_off_diag_4_2*hmn_diag_1 + 4.0*_s4sq*imp_smzk_prop_land_42*hmn_off_diag_4_1*hmn_diag_2 + 4.0*_s2sq*_s1sq*imp_smzk_prop_land_21*hmn_off_diag_2_1*hmn_off_diag_1_2 + 4.0*_s2sq*_s3sq*imp_smzk_prop_land_23*hmn_off_diag_2_1*hmn_off_diag_3_2 + 4.0*_s2sq*_s4sq*imp_smzk_prop_land_24*hmn_off_diag_2_1*hmn_off_diag_4_2 + 4.0*_s3sq*_s1sq*imp_smzk_prop_land_31*hmn_off_diag_3_1*hmn_off_diag_1_2 + 4.0*_s3sq*_s4sq*imp_smzk_prop_land_34*hmn_off_diag_3_1*hmn_off_diag_4_2 + 4.0*_s4sq*_s1sq*imp_smzk_prop_land_41*hmn_off_diag_4_1*hmn_off_diag_1_2 + 4.0*_s4sq*_s3sq*imp_smzk_prop_land_43*hmn_off_diag_4_1*hmn_off_diag_3_2

    omunu_land_12 *= sum_ssq

    # Caution! I have to check below later with my hand
    
    omunu_land_13 = 0.0

    omunu_land_13 = imp_smzk_prop_land_11*hmn_off_diag_1_3*hmn_diag_1 + imp_smzk_prop_land_33*hmn_off_diag_3_1*hmn_diag_3 + 4.0*imp_smzk_prop_land_13*hmn_diag_1*hmn_diag_3 + _s2sq*imp_smzk_prop_land_22*hmn_off_diag_2_1*hmn_off_diag_2_3 + _s4sq*imp_smzk_prop_land_44*hmn_off_diag_4_1*hmn_off_diag_4_3 + 4.0*_s2sq*imp_smzk_prop_land_12*hmn_off_diag_2_3*hmn_diag_1 + 4.0*_s2sq*imp_smzk_prop_land_23*hmn_off_diag_2_1*hmn_diag_3 + 4.0*_s4sq*imp_smzk_prop_land_14*hmn_off_diag_4_3*hmn_diag_1 + 4.0*_s4sq*imp_smzk_prop_land_43*hmn_off_diag_4_1*hmn_diag_3 + 4.0*_s3sq*_s1sq*imp_smzk_prop_land_31*hmn_off_diag_3_1*hmn_off_diag_1_3 + 4.0*_s3sq*_s2sq*imp_smzk_prop_land_32*hmn_off_diag_3_1*hmn_off_diag_2_3 + 4.0*_s3sq*_s4sq*imp_smzk_prop_land_34*hmn_off_diag_3_1*hmn_off_diag_4_3 + 4.0*_s2sq*_s1sq*imp_smzk_prop_land_21*hmn_off_diag_2_1*hmn_off_diag_1_3 + 4.0*_s2sq*_s4sq*imp_smzk_prop_land_24*hmn_off_diag_2_1*hmn_off_diag_4_3 + 4.0*_s4sq*_s1sq*imp_smzk_prop_land_41*hmn_off_diag_4_1*hmn_off_diag_1_3 + 4.0*_s4sq*_s2sq*imp_smzk_prop_land_42*hmn_off_diag_4_1*hmn_off_diag_2_3

    omunu_land_13 *= sum_ssq

    omunu_land_14 = 0.0

    omunu_land_14 = imp_smzk_prop_land_11*hmn_off_diag_1_4*hmn_diag_1 + imp_smzk_prop_land_44*hmn_off_diag_4_1*hmn_diag_4 + 4.0*imp_smzk_prop_land_14*hmn_diag_1*hmn_diag_4 + _s2sq*imp_smzk_prop_land_22*hmn_off_diag_2_1*hmn_off_diag_2_4 + _s3sq*imp_smzk_prop_land_33*hmn_off_diag_3_1*hmn_off_diag_3_4 + 4.0*_s2sq*imp_smzk_prop_land_12*hmn_off_diag_2_4*hmn_diag_1 + 4.0*_s2sq*imp_smzk_prop_land_24*hmn_off_diag_2_1*hmn_diag_4 + 4.0*_s3sq*imp_smzk_prop_land_13*hmn_off_diag_3_4*hmn_diag_1 + 4.0*_s3sq*imp_smzk_prop_land_34*hmn_off_diag_3_1*hmn_diag_4 + 4.0*_s4sq*_s1sq*imp_smzk_prop_land_41*hmn_off_diag_4_1*hmn_off_diag_1_4 + 4.0*_s4sq*_s2sq*imp_smzk_prop_land_42*hmn_off_diag_4_1*hmn_off_diag_2_4 + 4.0*_s4sq*_s3sq*imp_smzk_prop_land_43*hmn_off_diag_4_1*hmn_off_diag_3_4 + 4.0*_s2sq*_s1sq*imp_smzk_prop_land_21*hmn_off_diag_2_1*hmn_off_diag_1_4 + 4.0*_s2sq*_s3sq*imp_smzk_prop_land_23*hmn_off_diag_2_1*hmn_off_diag_3_4 + 4.0*_s3sq*_s1sq*imp_smzk_prop_land_31*hmn_off_diag_3_1*hmn_off_diag_1_4 + 4.0*_s3sq*_s2sq*imp_smzk_prop_land_32*hmn_off_diag_3_1*hmn_off_diag_2_4

    omunu_land_14 *= sum_ssq

    omunu_land_23 = 0.0

    omunu_land_23 = imp_smzk_prop_land_22*hmn_off_diag_2_3*hmn_diag_2 + imp_smzk_prop_land_33*hmn_off_diag_3_2*hmn_diag_3 + 4.0*imp_smzk_prop_land_23*hmn_diag_2*hmn_diag_3 + _s1sq*imp_smzk_prop_land_11*hmn_off_diag_1_2*hmn_off_diag_1_3 + _s4sq*imp_smzk_prop_land_44*hmn_off_diag_4_2*hmn_off_diag_4_3 + 4.0*_s1sq*imp_smzk_prop_land_21*hmn_off_diag_1_3*hmn_diag_2 + 4.0*_s1sq*imp_smzk_prop_land_13*hmn_off_diag_1_2*hmn_diag_3 + 4.0*_s4sq*imp_smzk_prop_land_24*hmn_off_diag_4_3*hmn_diag_2 + 4.0*_s4sq*imp_smzk_prop_land_43*hmn_off_diag_4_2*hmn_diag_3 + 4.0*_s3sq*_s2sq*imp_smzk_prop_land_32*hmn_off_diag_3_2*hmn_off_diag_2_3 + 4.0*_s3sq*_s1sq*imp_smzk_prop_land_31*hmn_off_diag_3_2*hmn_off_diag_1_3 + 4.0*_s3sq*_s4sq*imp_smzk_prop_land_34*hmn_off_diag_3_2*hmn_off_diag_4_3 + 4.0*_s1sq*_s2sq*imp_smzk_prop_land_12*hmn_off_diag_1_2*hmn_off_diag_2_3 + 4.0*_s1sq*_s4sq*imp_smzk_prop_land_14*hmn_off_diag_1_2*hmn_off_diag_4_3 + 4.0*_s4sq*_s2sq*imp_smzk_prop_land_42*hmn_off_diag_4_2*hmn_off_diag_2_3 + 4.0*_s4sq*_s1sq*imp_smzk_prop_land_41*hmn_off_diag_4_2*hmn_off_diag_1_3

    omunu_land_23 *= sum_ssq

    omunu_land_24 = 0.0

    omunu_land_24 = imp_smzk_prop_land_22*hmn_off_diag_2_4*hmn_diag_2 + imp_smzk_prop_land_44*hmn_off_diag_4_2*hmn_diag_4 + 4.0*imp_smzk_prop_land_24*hmn_diag_2*hmn_diag_4 + _s1sq*imp_smzk_prop_land_11*hmn_off_diag_1_2*hmn_off_diag_1_4 + _s3sq*imp_smzk_prop_land_33*hmn_off_diag_3_2*hmn_off_diag_3_4 + 4.0*_s1sq*imp_smzk_prop_land_21*hmn_off_diag_1_4*hmn_diag_2 + 4.0*_s1sq*imp_smzk_prop_land_14*hmn_off_diag_1_2*hmn_diag_4 + 4.0*_s3sq*imp_smzk_prop_land_23*hmn_off_diag_3_4*hmn_diag_2 + 4.0*_s3sq*imp_smzk_prop_land_34*hmn_off_diag_3_2*hmn_diag_4 + 4.0*_s4sq*_s2sq*imp_smzk_prop_land_42*hmn_off_diag_4_2*hmn_off_diag_2_4 + 4.0*_s4sq*_s1sq*imp_smzk_prop_land_41*hmn_off_diag_4_2*hmn_off_diag_1_4 + 4.0*_s4sq*_s3sq*imp_smzk_prop_land_43*hmn_off_diag_4_2*hmn_off_diag_3_4 + 4.0*_s1sq*_s2sq*imp_smzk_prop_land_12*hmn_off_diag_1_2*hmn_off_diag_2_4 + 4.0*_s1sq*_s3sq*imp_smzk_prop_land_13*hmn_off_diag_1_2*hmn_off_diag_3_4 + 4.0*_s3sq*_s2sq*imp_smzk_prop_land_32*hmn_off_diag_3_2*hmn_off_diag_2_4 + 4.0*_s3sq*_s1sq*imp_smzk_prop_land_31*hmn_off_diag_3_2*hmn_off_diag_1_4

    omunu_land_24 *= sum_ssq

    omunu_land_34 = 0.0

    omunu_land_34 = imp_smzk_prop_land_33*hmn_off_diag_3_4*hmn_diag_3 + imp_smzk_prop_land_44*hmn_off_diag_4_3*hmn_diag_4 + 4.0*imp_smzk_prop_land_34*hmn_diag_3*hmn_diag_4 + _s1sq*imp_smzk_prop_land_11*hmn_off_diag_1_3*hmn_off_diag_1_4 + _s2sq*imp_smzk_prop_land_22*hmn_off_diag_2_3*hmn_off_diag_2_4 + 4.0*_s1sq*imp_smzk_prop_land_31*hmn_off_diag_1_4*hmn_diag_3 + 4.0*_s1sq*imp_smzk_prop_land_14*hmn_off_diag_1_3*hmn_diag_4 + 4.0*_s2sq*imp_smzk_prop_land_32*hmn_off_diag_2_4*hmn_diag_3 + 4.0*_s2sq*imp_smzk_prop_land_24*hmn_off_diag_2_3*hmn_diag_4 + 4.0*_s4sq*_s3sq*imp_smzk_prop_land_43*hmn_off_diag_4_3*hmn_off_diag_3_4 + 4.0*_s4sq*_s1sq*imp_smzk_prop_land_41*hmn_off_diag_4_3*hmn_off_diag_1_4 + 4.0*_s4sq*_s2sq*imp_smzk_prop_land_42*hmn_off_diag_4_3*hmn_off_diag_2_4 + 4.0*_s1sq*_s3sq*imp_smzk_prop_land_13*hmn_off_diag_1_3*hmn_off_diag_3_4 + 4.0*_s1sq*_s2sq*imp_smzk_prop_land_12*hmn_off_diag_1_3*hmn_off_diag_2_4 + 4.0*_s2sq*_s3sq*imp_smzk_prop_land_23*hmn_off_diag_2_3*hmn_off_diag_3_4 + 4.0*_s2sq*_s1sq*imp_smzk_prop_land_21*hmn_off_diag_2_3*hmn_off_diag_1_4

    omunu_land_34 *= sum_ssq

    omunu_land_21 = omunu_land_12
    omunu_land_31 = omunu_land_13
    omunu_land_41 = omunu_land_14
    omunu_land_32 = omunu_land_23
    omunu_land_42 = omunu_land_24
    omunu_land_43 = omunu_land_34

    return (
        omunu_land_12, omunu_land_13, omunu_land_14,
        omunu_land_21, omunu_land_23, omunu_land_24,
        omunu_land_31, omunu_land_32, omunu_land_34, 
        omunu_land_41, omunu_land_42, omunu_land_43
    )
end