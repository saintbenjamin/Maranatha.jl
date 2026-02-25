@inline function fundamental_sbar_block(k)

    # --- Fundamental sbar / cbar ---
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

    # --- powers ---
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

    # --- sums ---
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

    inv_sum_ssq_s4t = sum_ssq - sum_s4t
    inv_sum_ssq_s4t = 1.0/(inv_sum_ssq_s4t)
    inv_sum_ssq_s4t_sq = inv_sum_ssq_s4t * inv_sum_ssq_s4t

    return (
        sbar0,sbar1,sbar2,sbar3,
        cbar0,cbar1,cbar2,cbar3,
        sin1,sin2,sin3,sin4,
        _s1sq,_s2sq,_s3sq,_s4sq,
        _s14t,_s24t,_s34t,_s44t,
        _s16t,_s26t,_s36t,_s46t,
        _s18t,_s28t,_s38t,_s48t,
        sum_ssq,sum_s4t,sum_s6t,sum_s8t,
        sum_ssq_sq,sum_ssq_3t,sum_ssq_4t,sum_ssq_5t,
        sum_s4t_sq,
        inv_sum_ssq,inv_sum_ssq_sq,
        inv_sum_ssq_s4t,inv_sum_ssq_s4t_sq
    )
end
