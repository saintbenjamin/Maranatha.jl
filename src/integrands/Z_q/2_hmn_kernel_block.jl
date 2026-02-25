@inline function hmn_kernel_block(
    _s1sq, _s2sq, _s3sq, _s4sq
)

    hmn_diag_1 = 1 - ( _s2sq + _s3sq + _s4sq ) + ( _s2sq*_s3sq + _s2sq*_s4sq + _s3sq*_s4sq ) - ( _s2sq*_s3sq*_s4sq )
    hmn_diag_2 = 1 - ( _s1sq + _s3sq + _s4sq ) + ( _s1sq*_s3sq + _s1sq*_s4sq + _s3sq*_s4sq ) - ( _s1sq*_s3sq*_s4sq )
    hmn_diag_3 = 1 - ( _s1sq + _s2sq + _s4sq ) + ( _s1sq*_s2sq + _s2sq*_s4sq + _s1sq*_s4sq ) - ( _s1sq*_s2sq*_s4sq )
    hmn_diag_4 = 1 - ( _s1sq + _s2sq + _s3sq ) + ( _s1sq*_s2sq + _s1sq*_s3sq + _s2sq*_s3sq ) - ( _s1sq*_s2sq*_s3sq )

    hmn_off_diag_2_1 = 1 - 0.5*( _s3sq + _s4sq ) + (1.0/3.0)*_s3sq*_s4sq 
    hmn_off_diag_3_1 = 1 - 0.5*( _s2sq + _s4sq ) + (1.0/3.0)*_s2sq*_s4sq 
    hmn_off_diag_4_1 = 1 - 0.5*( _s2sq + _s3sq ) + (1.0/3.0)*_s2sq*_s3sq 
    hmn_off_diag_3_2 = 1 - 0.5*( _s1sq + _s4sq ) + (1.0/3.0)*_s1sq*_s4sq 
    hmn_off_diag_4_2 = 1 - 0.5*( _s1sq + _s3sq ) + (1.0/3.0)*_s1sq*_s3sq 
    hmn_off_diag_4_3 = 1 - 0.5*( _s1sq + _s2sq ) + (1.0/3.0)*_s1sq*_s2sq 

    hmn_off_diag_1_2 = hmn_off_diag_2_1
    hmn_off_diag_1_3 = hmn_off_diag_3_1
    hmn_off_diag_1_4 = hmn_off_diag_4_1
    hmn_off_diag_2_3 = hmn_off_diag_3_2
    hmn_off_diag_2_4 = hmn_off_diag_4_2
    hmn_off_diag_3_4 = hmn_off_diag_4_3

    hmn_diag_1_sq = hmn_diag_1 * hmn_diag_1
    hmn_diag_2_sq = hmn_diag_2 * hmn_diag_2
    hmn_diag_3_sq = hmn_diag_3 * hmn_diag_3
    hmn_diag_4_sq = hmn_diag_4 * hmn_diag_4

    return (
        hmn_diag_1, hmn_diag_2, hmn_diag_3, hmn_diag_4,
        hmn_off_diag_1_2, hmn_off_diag_1_3, hmn_off_diag_1_4,
        hmn_off_diag_2_3, hmn_off_diag_2_4,
        hmn_off_diag_3_4,
        hmn_off_diag_2_1, hmn_off_diag_3_1, hmn_off_diag_4_1,
        hmn_off_diag_3_2, hmn_off_diag_4_2, 
        hmn_off_diag_4_3,
        hmn_diag_1_sq, hmn_diag_2_sq, hmn_diag_3_sq, hmn_diag_4_sq
    )
end