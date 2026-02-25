@inline function smzk_khat_block(
    sbar0,sbar1,sbar2,sbar3
)

    # ------------------------------------------------------------
    # Symanzik improved gluon propagator
    # k-hat construction and power sums
    # Algebra preserved exactly from original integrand.
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

    return (
        khat_1,khat_2,khat_3,khat_4,
        khat_1_sq,khat_2_sq,khat_3_sq,khat_4_sq,
        khat_1_4t,khat_2_4t,khat_3_4t,khat_4_4t,
        sum_khat_sq,inv_sum_khat_sq,
        sum_khat_sq_sq,sum_khat_sq_3t,sum_khat_sq_4t,sum_khat_sq_6t,
        khat_1_3t,khat_2_3t,khat_3_3t,khat_4_3t,
        sum_khat_4t,sum_khat_4t_sq,
        khat_1_5t,khat_2_5t,khat_3_5t,khat_4_5t,
        khat_1_6t,khat_2_6t,khat_3_6t,khat_4_6t,
        sum_khat_6t,
        khat_1_8t,khat_2_8t,khat_3_8t,khat_4_8t,
        sum_khat_8t
    )
end