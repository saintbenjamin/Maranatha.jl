@inline function pmu_block(
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

    pmu_feyn_1 = imp_smzk_prop_feyn_11*hmn_diag_1_sq + _s1sq*_s2sq*imp_smzk_prop_feyn_22*hmn_off_diag_2_1*hmn_off_diag_2_1 + _s1sq*_s3sq*imp_smzk_prop_feyn_33*hmn_off_diag_3_1*hmn_off_diag_3_1 + _s1sq*_s4sq*imp_smzk_prop_feyn_44*hmn_off_diag_4_1*hmn_off_diag_4_1 + 8.0*_s1sq*_s2sq*imp_smzk_prop_feyn_12*hmn_off_diag_2_1*hmn_diag_1 + 8.0*_s1sq*_s3sq*imp_smzk_prop_feyn_13*hmn_off_diag_3_1*hmn_diag_1 + 8.0*_s1sq*_s4sq*imp_smzk_prop_feyn_14*hmn_off_diag_4_1*hmn_diag_1 + 8.0*_s1sq*_s2sq*_s3sq*imp_smzk_prop_feyn_23*hmn_off_diag_2_1*hmn_off_diag_3_1 + 8.0*_s1sq*_s2sq*_s4sq*imp_smzk_prop_feyn_24*hmn_off_diag_2_1*hmn_off_diag_4_1 + 8.0*_s1sq*_s3sq*_s4sq*imp_smzk_prop_feyn_34*hmn_off_diag_3_1*hmn_off_diag_4_1

    pmu_feyn_1 *= 4.0*sum_ssq

    pmu_feyn_2 = imp_smzk_prop_feyn_22*hmn_diag_2_sq + _s2sq*_s1sq*imp_smzk_prop_feyn_11*hmn_off_diag_1_2*hmn_off_diag_1_2 + _s2sq*_s3sq*imp_smzk_prop_feyn_33*hmn_off_diag_3_2*hmn_off_diag_3_2 + _s2sq*_s4sq*imp_smzk_prop_feyn_44*hmn_off_diag_4_2*hmn_off_diag_4_2 + 8.0*_s2sq*_s1sq*imp_smzk_prop_feyn_21*hmn_off_diag_1_2*hmn_diag_2 + 8.0*_s2sq*_s3sq*imp_smzk_prop_feyn_23*hmn_off_diag_3_2*hmn_diag_2 + 8.0*_s2sq*_s4sq*imp_smzk_prop_feyn_24*hmn_off_diag_4_2*hmn_diag_2 + 8.0*_s2sq*_s1sq*_s3sq*imp_smzk_prop_feyn_13*hmn_off_diag_1_2*hmn_off_diag_3_2 + 8.0*_s2sq*_s1sq*_s4sq*imp_smzk_prop_feyn_14*hmn_off_diag_1_2*hmn_off_diag_4_2 + 8.0*_s2sq*_s3sq*_s4sq*imp_smzk_prop_feyn_34*hmn_off_diag_3_2*hmn_off_diag_4_2

    pmu_feyn_2 *= 4.0*sum_ssq
    
    pmu_feyn_3 = imp_smzk_prop_feyn_33*hmn_diag_3_sq + _s3sq*_s1sq*imp_smzk_prop_feyn_11*hmn_off_diag_1_3*hmn_off_diag_1_3 + _s3sq*_s2sq*imp_smzk_prop_feyn_22*hmn_off_diag_2_3*hmn_off_diag_2_3 + _s3sq*_s4sq*imp_smzk_prop_feyn_44*hmn_off_diag_4_3*hmn_off_diag_4_3 + 8.0*_s3sq*_s1sq*imp_smzk_prop_feyn_31*hmn_off_diag_1_3*hmn_diag_3 + 8.0*_s3sq*_s2sq*imp_smzk_prop_feyn_32*hmn_off_diag_2_3*hmn_diag_3 + 8.0*_s3sq*_s4sq*imp_smzk_prop_feyn_34*hmn_off_diag_4_3*hmn_diag_3 + 8.0*_s3sq*_s1sq*_s2sq*imp_smzk_prop_feyn_12*hmn_off_diag_1_3*hmn_off_diag_2_3 + 8.0*_s3sq*_s1sq*_s4sq*imp_smzk_prop_feyn_14*hmn_off_diag_1_3*hmn_off_diag_4_3 + 8.0*_s3sq*_s2sq*_s4sq*imp_smzk_prop_feyn_24*hmn_off_diag_2_3*hmn_off_diag_4_3

    pmu_feyn_3 *= 4.0*sum_ssq

    pmu_feyn_4 = imp_smzk_prop_feyn_44*hmn_diag_4_sq + _s4sq*_s1sq*imp_smzk_prop_feyn_11*hmn_off_diag_1_4*hmn_off_diag_1_4 + _s4sq*_s2sq*imp_smzk_prop_feyn_22*hmn_off_diag_2_4*hmn_off_diag_2_4 + _s4sq*_s3sq*imp_smzk_prop_feyn_33*hmn_off_diag_3_4*hmn_off_diag_3_4 + 8.0*_s4sq*_s1sq*imp_smzk_prop_feyn_41*hmn_off_diag_1_4*hmn_diag_4 + 8.0*_s4sq*_s2sq*imp_smzk_prop_feyn_42*hmn_off_diag_2_4*hmn_diag_4 + 8.0*_s4sq*_s3sq*imp_smzk_prop_feyn_43*hmn_off_diag_3_4*hmn_diag_4 + 8.0*_s4sq*_s1sq*_s2sq*imp_smzk_prop_feyn_12*hmn_off_diag_1_4*hmn_off_diag_2_4 + 8.0*_s4sq*_s1sq*_s3sq*imp_smzk_prop_feyn_13*hmn_off_diag_1_4*hmn_off_diag_3_4 + 8.0*_s4sq*_s2sq*_s3sq*imp_smzk_prop_feyn_23*hmn_off_diag_2_4*hmn_off_diag_3_4

    pmu_feyn_4 *= 4.0*sum_ssq

    pmu_land_1 = imp_smzk_prop_land_11*hmn_diag_1_sq + _s1sq*_s2sq*imp_smzk_prop_land_22*hmn_off_diag_2_1*hmn_off_diag_2_1 + _s1sq*_s3sq*imp_smzk_prop_land_33*hmn_off_diag_3_1*hmn_off_diag_3_1 + _s1sq*_s4sq*imp_smzk_prop_land_44*hmn_off_diag_4_1*hmn_off_diag_4_1 + 8.0*_s1sq*_s2sq*imp_smzk_prop_land_12*hmn_off_diag_2_1*hmn_diag_1 + 8.0*_s1sq*_s3sq*imp_smzk_prop_land_13*hmn_off_diag_3_1*hmn_diag_1 + 8.0*_s1sq*_s4sq*imp_smzk_prop_land_14*hmn_off_diag_4_1*hmn_diag_1 + 8.0*_s1sq*_s2sq*_s3sq*imp_smzk_prop_land_23*hmn_off_diag_2_1*hmn_off_diag_3_1 + 8.0*_s1sq*_s2sq*_s4sq*imp_smzk_prop_land_24*hmn_off_diag_2_1*hmn_off_diag_4_1 + 8.0*_s1sq*_s3sq*_s4sq*imp_smzk_prop_land_34*hmn_off_diag_3_1*hmn_off_diag_4_1

    pmu_land_1 *= 4.0*sum_ssq

    pmu_land_2 = imp_smzk_prop_land_22*hmn_diag_2_sq + _s2sq*_s1sq*imp_smzk_prop_land_11*hmn_off_diag_1_2*hmn_off_diag_1_2 + _s2sq*_s3sq*imp_smzk_prop_land_33*hmn_off_diag_3_2*hmn_off_diag_3_2 + _s2sq*_s4sq*imp_smzk_prop_land_44*hmn_off_diag_4_2*hmn_off_diag_4_2 + 8.0*_s2sq*_s1sq*imp_smzk_prop_land_21*hmn_off_diag_1_2*hmn_diag_2 + 8.0*_s2sq*_s3sq*imp_smzk_prop_land_23*hmn_off_diag_3_2*hmn_diag_2 + 8.0*_s2sq*_s4sq*imp_smzk_prop_land_24*hmn_off_diag_4_2*hmn_diag_2 + 8.0*_s2sq*_s1sq*_s3sq*imp_smzk_prop_land_13*hmn_off_diag_1_2*hmn_off_diag_3_2 + 8.0*_s2sq*_s1sq*_s4sq*imp_smzk_prop_land_14*hmn_off_diag_1_2*hmn_off_diag_4_2 + 8.0*_s2sq*_s3sq*_s4sq*imp_smzk_prop_land_34*hmn_off_diag_3_2*hmn_off_diag_4_2

    pmu_land_2 *= 4.0*sum_ssq

    pmu_land_3 = imp_smzk_prop_land_33*hmn_diag_3_sq + _s3sq*_s1sq*imp_smzk_prop_land_11*hmn_off_diag_1_3*hmn_off_diag_1_3 + _s3sq*_s2sq*imp_smzk_prop_land_22*hmn_off_diag_2_3*hmn_off_diag_2_3 + _s3sq*_s4sq*imp_smzk_prop_land_44*hmn_off_diag_4_3*hmn_off_diag_4_3 + 8.0*_s3sq*_s1sq*imp_smzk_prop_land_31*hmn_off_diag_1_3*hmn_diag_3 + 8.0*_s3sq*_s2sq*imp_smzk_prop_land_32*hmn_off_diag_2_3*hmn_diag_3 + 8.0*_s3sq*_s4sq*imp_smzk_prop_land_34*hmn_off_diag_4_3*hmn_diag_3 + 8.0*_s3sq*_s1sq*_s2sq*imp_smzk_prop_land_12*hmn_off_diag_1_3*hmn_off_diag_2_3 + 8.0*_s3sq*_s1sq*_s4sq*imp_smzk_prop_land_14*hmn_off_diag_1_3*hmn_off_diag_4_3 + 8.0*_s3sq*_s2sq*_s4sq*imp_smzk_prop_land_24*hmn_off_diag_2_3*hmn_off_diag_4_3

    pmu_land_3 *= 4.0*sum_ssq

    pmu_land_4 = imp_smzk_prop_land_44*hmn_diag_4_sq + _s4sq*_s1sq*imp_smzk_prop_land_11*hmn_off_diag_1_4*hmn_off_diag_1_4 + _s4sq*_s2sq*imp_smzk_prop_land_22*hmn_off_diag_2_4*hmn_off_diag_2_4 + _s4sq*_s3sq*imp_smzk_prop_land_33*hmn_off_diag_3_4*hmn_off_diag_3_4 + 8.0*_s4sq*_s1sq*imp_smzk_prop_land_41*hmn_off_diag_1_4*hmn_diag_4 + 8.0*_s4sq*_s2sq*imp_smzk_prop_land_42*hmn_off_diag_2_4*hmn_diag_4 + 8.0*_s4sq*_s3sq*imp_smzk_prop_land_43*hmn_off_diag_3_4*hmn_diag_4 + 8.0*_s4sq*_s1sq*_s2sq*imp_smzk_prop_land_12*hmn_off_diag_1_4*hmn_off_diag_2_4 + 8.0*_s4sq*_s1sq*_s3sq*imp_smzk_prop_land_13*hmn_off_diag_1_4*hmn_off_diag_3_4 + 8.0*_s4sq*_s2sq*_s3sq*imp_smzk_prop_land_23*hmn_off_diag_2_4*hmn_off_diag_3_4

    pmu_land_4 *= 4.0*sum_ssq

    sum_pmu_feyn = pmu_feyn_1 + pmu_feyn_2 + pmu_feyn_3 + pmu_feyn_4

    sum_s2_pmu_feyn = _s1sq*pmu_feyn_1 + _s2sq*pmu_feyn_2 + _s3sq*pmu_feyn_3 + _s4sq*pmu_feyn_4
    sum_s4_pmu_feyn = _s14t*pmu_feyn_1 + _s24t*pmu_feyn_2 + _s34t*pmu_feyn_3 + _s44t*pmu_feyn_4
    sum_s6_pmu_feyn = _s16t*pmu_feyn_1 + _s26t*pmu_feyn_2 + _s36t*pmu_feyn_3 + _s46t*pmu_feyn_4
    sum_s8_pmu_feyn = _s18t*pmu_feyn_1 + _s28t*pmu_feyn_2 + _s38t*pmu_feyn_3 + _s48t*pmu_feyn_4

    sum_pmu_land = pmu_land_1 + pmu_land_2 + pmu_land_3 + pmu_land_4

    sum_s2_pmu_land = _s1sq*pmu_land_1 + _s2sq*pmu_land_2 + _s3sq*pmu_land_3 + _s4sq*pmu_land_4
    sum_s4_pmu_land = _s14t*pmu_land_1 + _s24t*pmu_land_2 + _s34t*pmu_land_3 + _s44t*pmu_land_4
    sum_s6_pmu_land = _s16t*pmu_land_1 + _s26t*pmu_land_2 + _s36t*pmu_land_3 + _s46t*pmu_land_4
    sum_s8_pmu_land = _s18t*pmu_land_1 + _s28t*pmu_land_2 + _s38t*pmu_land_3 + _s48t*pmu_land_4

    return (
        pmu_feyn_1, pmu_feyn_2, pmu_feyn_3, pmu_feyn_4,
        pmu_land_1, pmu_land_2, pmu_land_3, pmu_land_4,

        sum_pmu_feyn,
        sum_s2_pmu_feyn, sum_s4_pmu_feyn, sum_s6_pmu_feyn, sum_s8_pmu_feyn,

        sum_pmu_land,
        sum_s2_pmu_land, sum_s4_pmu_land, sum_s6_pmu_land, sum_s8_pmu_land
    )
end