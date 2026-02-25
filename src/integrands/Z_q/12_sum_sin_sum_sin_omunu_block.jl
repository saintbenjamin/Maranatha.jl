@inline function sum_sin_sum_sin_omunu_block(
    omunu_feyn_12, omunu_feyn_13, omunu_feyn_14,
    omunu_feyn_21, omunu_feyn_23, omunu_feyn_24,
    omunu_feyn_31, omunu_feyn_32, omunu_feyn_34,
    omunu_feyn_41, omunu_feyn_42, omunu_feyn_43,
    omunu_land_12, omunu_land_13, omunu_land_14,
    omunu_land_21, omunu_land_23, omunu_land_24,
    omunu_land_31, omunu_land_32, omunu_land_34,
    omunu_land_41, omunu_land_42, omunu_land_43,
    sin1, sin2, sin3, sin4
)

    sum_sin_sum_sin_omunu_feyn =
        sin1*omunu_feyn_12*sin2 + sin1*omunu_feyn_13*sin3 + sin1*omunu_feyn_14*sin4 +
        sin2*omunu_feyn_21*sin1 + sin2*omunu_feyn_23*sin3 + sin2*omunu_feyn_24*sin4 +
        sin3*omunu_feyn_31*sin1 + sin3*omunu_feyn_32*sin2 + sin3*omunu_feyn_34*sin4 +
        sin4*omunu_feyn_41*sin1 + sin4*omunu_feyn_42*sin2 + sin4*omunu_feyn_43*sin3

    sum_sin_sum_sin_omunu_land =
        sin1*omunu_land_12*sin2 + sin1*omunu_land_13*sin3 + sin1*omunu_land_14*sin4 +
        sin2*omunu_land_21*sin1 + sin2*omunu_land_23*sin3 + sin2*omunu_land_24*sin4 +
        sin3*omunu_land_31*sin1 + sin3*omunu_land_32*sin2 + sin3*omunu_land_34*sin4 +
        sin4*omunu_land_41*sin1 + sin4*omunu_land_42*sin2 + sin4*omunu_land_43*sin3

    return sum_sin_sum_sin_omunu_feyn, sum_sin_sum_sin_omunu_land
end