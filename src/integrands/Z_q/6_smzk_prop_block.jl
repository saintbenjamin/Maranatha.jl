@inline function smzk_prop_block(
    # ---- feynman inputs ----
    coeff_delta_t,
    delta_t_smzk_prop_11, delta_t_smzk_prop_22,
    delta_t_smzk_prop_33, delta_t_smzk_prop_44,
    delta_t_smzk_prop_12, delta_t_smzk_prop_13,
    delta_t_smzk_prop_14, delta_t_smzk_prop_23,
    delta_t_smzk_prop_24, delta_t_smzk_prop_34,

    coeff_m_mn,
    m_mn_smzk_prop_11, m_mn_smzk_prop_22,
    m_mn_smzk_prop_33, m_mn_smzk_prop_44,
    m_mn_smzk_prop_12, m_mn_smzk_prop_13,
    m_mn_smzk_prop_14, m_mn_smzk_prop_23,
    m_mn_smzk_prop_24, m_mn_smzk_prop_34,

    coeff_m_sq_mn,
    m_sq_mn_smzk_prop_11, m_sq_mn_smzk_prop_22,
    m_sq_mn_smzk_prop_33, m_sq_mn_smzk_prop_44,
    m_sq_mn_smzk_prop_12, m_sq_mn_smzk_prop_13,
    m_sq_mn_smzk_prop_14, m_sq_mn_smzk_prop_23,
    m_sq_mn_smzk_prop_24, m_sq_mn_smzk_prop_34,

    # ---- shared / landau inputs ----
    inv_sum_khat_sq,
    p_mn_smzk_prop_11, p_mn_smzk_prop_22,
    p_mn_smzk_prop_33, p_mn_smzk_prop_44,
    p_mn_smzk_prop_12, p_mn_smzk_prop_13,
    p_mn_smzk_prop_14, p_mn_smzk_prop_23,
    p_mn_smzk_prop_24, p_mn_smzk_prop_34
)

    # ============================================================
    # Feynman part
    # ============================================================

    imp_smzk_prop_feyn_11 =
        coeff_delta_t * delta_t_smzk_prop_11 +
        coeff_m_mn * m_mn_smzk_prop_11 +
        coeff_m_sq_mn * m_sq_mn_smzk_prop_11 +
        inv_sum_khat_sq * p_mn_smzk_prop_11

    imp_smzk_prop_feyn_22 =
        coeff_delta_t * delta_t_smzk_prop_22 +
        coeff_m_mn * m_mn_smzk_prop_22 +
        coeff_m_sq_mn * m_sq_mn_smzk_prop_22 +
        inv_sum_khat_sq * p_mn_smzk_prop_22

    imp_smzk_prop_feyn_33 =
        coeff_delta_t * delta_t_smzk_prop_33 +
        coeff_m_mn * m_mn_smzk_prop_33 +
        coeff_m_sq_mn * m_sq_mn_smzk_prop_33 +
        inv_sum_khat_sq * p_mn_smzk_prop_33

    imp_smzk_prop_feyn_44 =
        coeff_delta_t * delta_t_smzk_prop_44 +
        coeff_m_mn * m_mn_smzk_prop_44 +
        coeff_m_sq_mn * m_sq_mn_smzk_prop_44 +
        inv_sum_khat_sq * p_mn_smzk_prop_44

    imp_smzk_prop_feyn_12 =
        coeff_delta_t * delta_t_smzk_prop_12 +
        coeff_m_mn * m_mn_smzk_prop_12 +
        coeff_m_sq_mn * m_sq_mn_smzk_prop_12 +
        inv_sum_khat_sq * p_mn_smzk_prop_12

    imp_smzk_prop_feyn_13 =
        coeff_delta_t * delta_t_smzk_prop_13 +
        coeff_m_mn * m_mn_smzk_prop_13 +
        coeff_m_sq_mn * m_sq_mn_smzk_prop_13 +
        inv_sum_khat_sq * p_mn_smzk_prop_13

    imp_smzk_prop_feyn_14 =
        coeff_delta_t * delta_t_smzk_prop_14 +
        coeff_m_mn * m_mn_smzk_prop_14 +
        coeff_m_sq_mn * m_sq_mn_smzk_prop_14 +
        inv_sum_khat_sq * p_mn_smzk_prop_14

    imp_smzk_prop_feyn_23 =
        coeff_delta_t * delta_t_smzk_prop_23 +
        coeff_m_mn * m_mn_smzk_prop_23 +
        coeff_m_sq_mn * m_sq_mn_smzk_prop_23 +
        inv_sum_khat_sq * p_mn_smzk_prop_23

    imp_smzk_prop_feyn_24 =
        coeff_delta_t * delta_t_smzk_prop_24 +
        coeff_m_mn * m_mn_smzk_prop_24 +
        coeff_m_sq_mn * m_sq_mn_smzk_prop_24 +
        inv_sum_khat_sq * p_mn_smzk_prop_24

    imp_smzk_prop_feyn_34 =
        coeff_delta_t * delta_t_smzk_prop_34 +
        coeff_m_mn * m_mn_smzk_prop_34 +
        coeff_m_sq_mn * m_sq_mn_smzk_prop_34 +
        inv_sum_khat_sq * p_mn_smzk_prop_34

    imp_smzk_prop_feyn_21 = imp_smzk_prop_feyn_12
    imp_smzk_prop_feyn_31 = imp_smzk_prop_feyn_13
    imp_smzk_prop_feyn_41 = imp_smzk_prop_feyn_14
    imp_smzk_prop_feyn_32 = imp_smzk_prop_feyn_23
    imp_smzk_prop_feyn_42 = imp_smzk_prop_feyn_24
    imp_smzk_prop_feyn_43 = imp_smzk_prop_feyn_34

    # ============================================================
    # Landau part
    # ============================================================

    imp_smzk_prop_land_11 = -p_mn_smzk_prop_11 * inv_sum_khat_sq
    imp_smzk_prop_land_22 = -p_mn_smzk_prop_22 * inv_sum_khat_sq
    imp_smzk_prop_land_33 = -p_mn_smzk_prop_33 * inv_sum_khat_sq
    imp_smzk_prop_land_44 = -p_mn_smzk_prop_44 * inv_sum_khat_sq

    imp_smzk_prop_land_12 = -p_mn_smzk_prop_12 * inv_sum_khat_sq
    imp_smzk_prop_land_13 = -p_mn_smzk_prop_13 * inv_sum_khat_sq
    imp_smzk_prop_land_14 = -p_mn_smzk_prop_14 * inv_sum_khat_sq
    imp_smzk_prop_land_23 = -p_mn_smzk_prop_23 * inv_sum_khat_sq
    imp_smzk_prop_land_24 = -p_mn_smzk_prop_24 * inv_sum_khat_sq
    imp_smzk_prop_land_34 = -p_mn_smzk_prop_34 * inv_sum_khat_sq

    imp_smzk_prop_land_21 = imp_smzk_prop_land_12
    imp_smzk_prop_land_31 = imp_smzk_prop_land_13
    imp_smzk_prop_land_41 = imp_smzk_prop_land_14
    imp_smzk_prop_land_32 = imp_smzk_prop_land_23
    imp_smzk_prop_land_42 = imp_smzk_prop_land_24
    imp_smzk_prop_land_43 = imp_smzk_prop_land_34

    return (
        # ---- feynman (16 entries) ----
        imp_smzk_prop_feyn_11, imp_smzk_prop_feyn_22,
        imp_smzk_prop_feyn_33, imp_smzk_prop_feyn_44,
        imp_smzk_prop_feyn_12, imp_smzk_prop_feyn_13,
        imp_smzk_prop_feyn_14, imp_smzk_prop_feyn_23,
        imp_smzk_prop_feyn_24, imp_smzk_prop_feyn_34,
        imp_smzk_prop_feyn_21, imp_smzk_prop_feyn_31,
        imp_smzk_prop_feyn_41, imp_smzk_prop_feyn_32,
        imp_smzk_prop_feyn_42, imp_smzk_prop_feyn_43,

        # ---- landau (16 entries) ----
        imp_smzk_prop_land_11, imp_smzk_prop_land_22,
        imp_smzk_prop_land_33, imp_smzk_prop_land_44,
        imp_smzk_prop_land_12, imp_smzk_prop_land_13,
        imp_smzk_prop_land_14, imp_smzk_prop_land_23,
        imp_smzk_prop_land_24, imp_smzk_prop_land_34,
        imp_smzk_prop_land_21, imp_smzk_prop_land_31,
        imp_smzk_prop_land_41, imp_smzk_prop_land_32,
        imp_smzk_prop_land_42, imp_smzk_prop_land_43
    )
end