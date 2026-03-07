@inline function smzk_scalar_block(
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

    c_smzk_prop = -1.0/12.0

    f_smzk_prop = 1.0 - c_smzk_prop * sum_khat_4t * inv_sum_khat_sq
    inv_f_smzk_prop = 1.0/(f_smzk_prop)

    c_tilde_smzk_prop    = c_smzk_prop*inv_f_smzk_prop
    c_tilde_smzk_prop_sq = c_tilde_smzk_prop*c_tilde_smzk_prop
    c_tilde_smzk_prop_3t = c_tilde_smzk_prop*c_tilde_smzk_prop_sq

    x1_smzk_prop = sum_khat_sq_sq - sum_khat_4t
    x2_smzk_prop = sum_khat_sq*sum_khat_6t - 1.5*sum_khat_sq_sq*sum_khat_4t + 0.5*sum_khat_sq_4t
    x3_smzk_prop = (1.0/6.0)*sum_khat_sq_6t + 0.5*sum_khat_sq_sq*sum_khat_4t_sq - sum_khat_sq_4t*sum_khat_4t + (4.0/3.0)*sum_khat_sq_3t*sum_khat_6t - sum_khat_sq_sq*sum_khat_8t

    smzk_block_01 = sum_khat_sq - c_tilde_smzk_prop*x1_smzk_prop
    smzk_block_02 = sum_khat_sq*smzk_block_01 + c_tilde_smzk_prop_sq*x2_smzk_prop

    deno_smzk_prop = 0.0
    deno_smzk_prop = sum_khat_sq*smzk_block_02 - c_tilde_smzk_prop_3t*x3_smzk_prop
    deno_smzk_prop *= f_smzk_prop
    deno_smzk_prop = 1.0/(deno_smzk_prop)

    return (
        c_tilde_smzk_prop,
        c_tilde_smzk_prop_sq,
        smzk_block_01,
        smzk_block_02,
        deno_smzk_prop
    )
end
