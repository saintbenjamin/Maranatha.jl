@inline function sum_sx_sy_omunu_feyn_block(
    omunu_feyn_12, omunu_feyn_13, omunu_feyn_14,
    omunu_feyn_21, omunu_feyn_23, omunu_feyn_24,
    omunu_feyn_31, omunu_feyn_32, omunu_feyn_34,
    omunu_feyn_41, omunu_feyn_42, omunu_feyn_43,
    _s1sq, _s2sq, _s3sq, _s4sq,
    _s14t, _s24t, _s34t, _s44t,
    _s16t, _s26t, _s36t, _s46t
)

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

    return (
        sum_s2_sum_s2_omunu_feyn,
        sum_s2_sum_s4_omunu_feyn,
        sum_s4_sum_s2_omunu_feyn,
        sum_s4_sum_s4_omunu_feyn,
        sum_s6_sum_s2_omunu_feyn,
        sum_s6_sum_s4_omunu_feyn
    )
end