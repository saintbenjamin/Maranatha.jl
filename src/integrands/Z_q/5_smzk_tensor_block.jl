@inline function smzk_tensor_block(
    # ---- common / stage1 inputs ----
    inv_sum_khat_sq,
    smzk_block_01, smzk_block_02,
    deno_smzk_prop, c_tilde_smzk_prop,
    khat_1_sq, khat_2_sq, khat_3_sq, khat_4_sq,
    khat_1_4t, khat_2_4t, khat_3_4t, khat_4_4t,
    sum_khat_sq, sum_khat_4t,

    # ---- stage2 extra inputs ----
    c_tilde_smzk_prop_sq,
    khat_1_6t, khat_2_6t, khat_3_6t, khat_4_6t,
    sum_khat_sq_sq,
    sum_khat_4t_sq,
    sum_khat_6t
)

    # ============================================================
    # stage 1
    # ============================================================

    p_mn_smzk_prop_11 = khat_1_sq * inv_sum_khat_sq
    p_mn_smzk_prop_22 = khat_2_sq * inv_sum_khat_sq
    p_mn_smzk_prop_33 = khat_3_sq * inv_sum_khat_sq
    p_mn_smzk_prop_44 = khat_4_sq * inv_sum_khat_sq

    # off-diagonal p_mn
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

    coeff_delta_t = smzk_block_02 * deno_smzk_prop

    delta_t_smzk_prop_11 = 1.0 - p_mn_smzk_prop_11
    delta_t_smzk_prop_22 = 1.0 - p_mn_smzk_prop_22
    delta_t_smzk_prop_33 = 1.0 - p_mn_smzk_prop_33
    delta_t_smzk_prop_44 = 1.0 - p_mn_smzk_prop_44

    delta_t_smzk_prop_12 = -p_mn_smzk_prop_12
    delta_t_smzk_prop_13 = -p_mn_smzk_prop_13
    delta_t_smzk_prop_14 = -p_mn_smzk_prop_14
    delta_t_smzk_prop_23 = -p_mn_smzk_prop_23
    delta_t_smzk_prop_24 = -p_mn_smzk_prop_24
    delta_t_smzk_prop_34 = -p_mn_smzk_prop_34

    delta_t_smzk_prop_21 = delta_t_smzk_prop_12
    delta_t_smzk_prop_31 = delta_t_smzk_prop_13
    delta_t_smzk_prop_41 = delta_t_smzk_prop_14
    delta_t_smzk_prop_32 = delta_t_smzk_prop_23
    delta_t_smzk_prop_42 = delta_t_smzk_prop_24
    delta_t_smzk_prop_43 = delta_t_smzk_prop_34

    coeff_m_mn = c_tilde_smzk_prop * smzk_block_01 * deno_smzk_prop

    m_mn_smzk_prop_11 =
        khat_1_sq * sum_khat_sq - 2.0 * khat_1_4t + khat_1_sq * sum_khat_4t * inv_sum_khat_sq
    m_mn_smzk_prop_22 =
        khat_2_sq * sum_khat_sq - 2.0 * khat_2_4t + khat_2_sq * sum_khat_4t * inv_sum_khat_sq
    m_mn_smzk_prop_33 =
        khat_3_sq * sum_khat_sq - 2.0 * khat_3_4t + khat_3_sq * sum_khat_4t * inv_sum_khat_sq
    m_mn_smzk_prop_44 =
        khat_4_sq * sum_khat_sq - 2.0 * khat_4_4t + khat_4_sq * sum_khat_4t * inv_sum_khat_sq

    m_mn_smzk_prop_12 = -khat_1_sq - khat_2_sq + sum_khat_4t * inv_sum_khat_sq
    m_mn_smzk_prop_13 = -khat_1_sq - khat_3_sq + sum_khat_4t * inv_sum_khat_sq
    m_mn_smzk_prop_14 = -khat_1_sq - khat_4_sq + sum_khat_4t * inv_sum_khat_sq
    m_mn_smzk_prop_23 = -khat_2_sq - khat_3_sq + sum_khat_4t * inv_sum_khat_sq
    m_mn_smzk_prop_24 = -khat_2_sq - khat_4_sq + sum_khat_4t * inv_sum_khat_sq
    m_mn_smzk_prop_34 = -khat_3_sq - khat_4_sq + sum_khat_4t * inv_sum_khat_sq

    m_mn_smzk_prop_21 = m_mn_smzk_prop_12
    m_mn_smzk_prop_31 = m_mn_smzk_prop_13
    m_mn_smzk_prop_41 = m_mn_smzk_prop_14
    m_mn_smzk_prop_32 = m_mn_smzk_prop_23
    m_mn_smzk_prop_42 = m_mn_smzk_prop_24
    m_mn_smzk_prop_43 = m_mn_smzk_prop_34

    # ============================================================
    # stage 2
    # ============================================================

    coeff_m_sq_mn = c_tilde_smzk_prop_sq * deno_smzk_prop

    m_sq_mn_smzk_prop_11 =
        khat_1_4t * sum_khat_sq_sq - 3.0 * khat_1_6t * sum_khat_sq + 2.0 * khat_1_4t * sum_khat_4t +
        khat_1_sq * sum_khat_6t - khat_1_sq * sum_khat_4t_sq * inv_sum_khat_sq
    m_sq_mn_smzk_prop_22 =
        khat_2_4t * sum_khat_sq_sq - 3.0 * khat_2_6t * sum_khat_sq + 2.0 * khat_2_4t * sum_khat_4t +
        khat_2_sq * sum_khat_6t - khat_2_sq * sum_khat_4t_sq * inv_sum_khat_sq
    m_sq_mn_smzk_prop_33 =
        khat_3_4t * sum_khat_sq_sq - 3.0 * khat_3_6t * sum_khat_sq + 2.0 * khat_3_4t * sum_khat_4t +
        khat_3_sq * sum_khat_6t - khat_3_sq * sum_khat_4t_sq * inv_sum_khat_sq
    m_sq_mn_smzk_prop_44 =
        khat_4_4t * sum_khat_sq_sq - 3.0 * khat_4_6t * sum_khat_sq + 2.0 * khat_4_4t * sum_khat_4t +
        khat_4_sq * sum_khat_6t - khat_4_sq * sum_khat_4t_sq * inv_sum_khat_sq

    m_sq_mn_smzk_prop_12 =
        - (khat_1_4t + khat_2_4t + khat_1_sq * khat_2_sq) * sum_khat_sq +
        (khat_1_sq + khat_2_sq) * sum_khat_4t +
        sum_khat_6t - sum_khat_4t_sq * inv_sum_khat_sq
    m_sq_mn_smzk_prop_13 =
        - (khat_1_4t + khat_3_4t + khat_1_sq * khat_3_sq) * sum_khat_sq +
        (khat_1_sq + khat_3_sq) * sum_khat_4t +
        sum_khat_6t - sum_khat_4t_sq * inv_sum_khat_sq
    m_sq_mn_smzk_prop_14 =
        - (khat_1_4t + khat_4_4t + khat_1_sq * khat_4_sq) * sum_khat_sq +
        (khat_1_sq + khat_4_sq) * sum_khat_4t +
        sum_khat_6t - sum_khat_4t_sq * inv_sum_khat_sq
    m_sq_mn_smzk_prop_23 =
        - (khat_2_4t + khat_3_4t + khat_2_sq * khat_3_sq) * sum_khat_sq +
        (khat_2_sq + khat_3_sq) * sum_khat_4t +
        sum_khat_6t - sum_khat_4t_sq * inv_sum_khat_sq
    m_sq_mn_smzk_prop_24 =
        - (khat_2_4t + khat_4_4t + khat_2_sq * khat_4_sq) * sum_khat_sq +
        (khat_2_sq + khat_4_sq) * sum_khat_4t +
        sum_khat_6t - sum_khat_4t_sq * inv_sum_khat_sq
    m_sq_mn_smzk_prop_34 =
        - (khat_3_4t + khat_4_4t + khat_3_sq * khat_4_sq) * sum_khat_sq +
        (khat_3_sq + khat_4_sq) * sum_khat_4t +
        sum_khat_6t - sum_khat_4t_sq * inv_sum_khat_sq

    m_sq_mn_smzk_prop_21 = m_sq_mn_smzk_prop_12
    m_sq_mn_smzk_prop_31 = m_sq_mn_smzk_prop_13
    m_sq_mn_smzk_prop_41 = m_sq_mn_smzk_prop_14
    m_sq_mn_smzk_prop_32 = m_sq_mn_smzk_prop_23
    m_sq_mn_smzk_prop_42 = m_sq_mn_smzk_prop_24
    m_sq_mn_smzk_prop_43 = m_sq_mn_smzk_prop_34

    return (
        # ---- p_mn ----
        p_mn_smzk_prop_11, p_mn_smzk_prop_22,
        p_mn_smzk_prop_33, p_mn_smzk_prop_44,
        p_mn_smzk_prop_12, p_mn_smzk_prop_13, p_mn_smzk_prop_14,
        p_mn_smzk_prop_23, p_mn_smzk_prop_24, p_mn_smzk_prop_34,
        p_mn_smzk_prop_21, p_mn_smzk_prop_31, p_mn_smzk_prop_41,
        p_mn_smzk_prop_32, p_mn_smzk_prop_42, p_mn_smzk_prop_43,

        # ---- delta_t ----
        coeff_delta_t,
        delta_t_smzk_prop_11, delta_t_smzk_prop_22,
        delta_t_smzk_prop_33, delta_t_smzk_prop_44,
        delta_t_smzk_prop_12, delta_t_smzk_prop_13,
        delta_t_smzk_prop_14, delta_t_smzk_prop_23,
        delta_t_smzk_prop_24, delta_t_smzk_prop_34,
        delta_t_smzk_prop_21, delta_t_smzk_prop_31,
        delta_t_smzk_prop_41, delta_t_smzk_prop_32,
        delta_t_smzk_prop_42, delta_t_smzk_prop_43,

        # ---- m_mn ----
        coeff_m_mn,
        m_mn_smzk_prop_11, m_mn_smzk_prop_22,
        m_mn_smzk_prop_33, m_mn_smzk_prop_44,
        m_mn_smzk_prop_12, m_mn_smzk_prop_13,
        m_mn_smzk_prop_14, m_mn_smzk_prop_23,
        m_mn_smzk_prop_24, m_mn_smzk_prop_34,
        m_mn_smzk_prop_21, m_mn_smzk_prop_31,
        m_mn_smzk_prop_41, m_mn_smzk_prop_32,
        m_mn_smzk_prop_42, m_mn_smzk_prop_43,

        # ---- m_sq_mn ----
        coeff_m_sq_mn,
        m_sq_mn_smzk_prop_11, m_sq_mn_smzk_prop_22,
        m_sq_mn_smzk_prop_33, m_sq_mn_smzk_prop_44,
        m_sq_mn_smzk_prop_12, m_sq_mn_smzk_prop_13,
        m_sq_mn_smzk_prop_14, m_sq_mn_smzk_prop_23,
        m_sq_mn_smzk_prop_24, m_sq_mn_smzk_prop_34,
        m_sq_mn_smzk_prop_21, m_sq_mn_smzk_prop_31,
        m_sq_mn_smzk_prop_41, m_sq_mn_smzk_prop_32,
        m_sq_mn_smzk_prop_42, m_sq_mn_smzk_prop_43
    )
end