/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use std::arch::x86_64::*;

// Vector of four floating point values.
pub type SimdFloat4 = __m128;

// Vector of four integer values.
pub type SimdInt4 = __m128i;

pub mod simd_float4 {
    use std::arch::x86_64::*;
    use crate::simd_math::{SimdInt4, SimdFloat4};

    macro_rules! _mm_shuffle {
        ($z:expr, $y:expr, $x:expr, $w:expr) => {
            (($z << 6) | ($y << 4) | ($x << 2) | $w)
        };
    }

    macro_rules! ozz_shuffle_ps1 {
        ($_v:expr, $_m:expr) => {
            _mm_shuffle_ps($_v, $_v, $_m)
        };
    }

    macro_rules! ozz_sse_splat_f {
        ($_v:expr, $_i:expr) => {
            ozz_shuffle_ps1($_v, _mm_shuffle($_i,$_i,$_i,$_i))
        };
    }

    macro_rules! ozz_sse_splat_i {
        ($_v:expr, $_i:expr) => {
            _mm_shuffle_epi32($_v, _mm_shuffle($_i, $_i, $_i, $_i))
        };
    }

    macro_rules! ozz_sse_hadd2_f {
        ($_v:expr) => {
            _mm_add_ss($_v, ozz_sse_splat_i($_v, 1))
        };
    }

    macro_rules! ozz_sse_hadd3_f {
        ($_v:expr) => {
            _mm_add_ss(_mm_add_ss($_v, ozz_sse_splat_f($_v, 2)), ozz_sse_splat_f($_v, 1))
        };
    }

    macro_rules! ozz_sse_hadd4_f {
        ($_v:expr, $_r:expr) => {
            {
                let haddxyzw = _mm_add_ps($_v, _mm_movehl_ps($_v, $_v));
                $_r = _mm_add_ss(haddxyzw, ozz_sse_splat_f(haddxyzw, 1));
            }
        };
    }

    macro_rules! ozz_sse_dot2_f {
        ($_a:expr, $_b:expr, $_r:expr) => {
            {
                let ab = _mm_mul_ps($_a, $_b);
                $_r = _mm_add_ss(ab, ozz_sse_splat_f(ab, 1));
            }
        };
    }

    macro_rules! ozz_sse_dot3_f {
        ($_a:expr, $_b:expr, $_r:expr) => {
            $_r = _mm_dp_ps($_a, $_b, 0x7f);
        };
    }

    macro_rules! ozz_sse_dot4_f {
        ($_a:expr, $_b:expr, $_r:expr) => {
            $_r = _mm_dp_ps($_a, $_b, 0xff);
        };
    }

    macro_rules! ozz_madd {
        ($_a:expr, $_b:expr, $_c:expr) => {
            _mm_add_ps(_mm_mul_ps($_a, $_b), $_c)
        };
    }

    macro_rules! ozz_msub {
        ($_a:expr, $_b:expr, $_c:expr) => {
            _mm_sub_ps(_mm_mul_ps($_a, $_b), $_c)
        };
    }

    macro_rules! ozz_nmadd {
        ($_a:expr, $_b:expr, $_c:expr) => {
            _mm_sub_ps($_c, _mm_mul_ps($_a, $_b))
        };
    }

    macro_rules! ozz_nmsub {
        ($_a:expr, $_b:expr, $_c:expr) => {
            (-_mm_add_ps(_mm_mul_ps($_a, $_b), $_c))
        };
    }

    macro_rules! ozz_maddx {
        ($_a:expr, $_b:expr, $_c:expr) => {
            _mm_add_ss(_mm_mul_ss($_a, $_b), $_c)
        };
    }

    macro_rules! ozz_msubx {
        ($_a:expr, $_b:expr, $_c:expr) => {
            _mm_sub_ss(_mm_mul_ss($_a, $_b), $_c)
        };
    }

    macro_rules! ozz_nmaddx {
        ($_a:expr, $_b:expr, $_c:expr) => {
            _mm_sub_ss($_c, _mm_mul_ss($_a, $_b))
        };
    }

    macro_rules! ozz_nmsubx {
        ($_a:expr, $_b:expr, $_c:expr) => {
            (-_mm_add_ss(_mm_mul_ss($_a, $_b), $_c))
        };
    }

    macro_rules! ozz_sse_select_f {
        ($_b:expr, $_true:expr, $_false:expr) => {
            _mm_blendv_ps($_false, $_true, _mm_castsi128_ps($_b))
        };
    }

    macro_rules! ozz_sse_select_i {
        ($_b:expr, $_true:expr, $_false:expr) => {
            _mm_blendv_epi8($_false, $_true, $_b)
        };
    }

    //----------------------------------------------------------------------------------------------
    // Returns a SimdFloat4 vector with all components set to 0.
    #[inline]
    pub fn zero() -> SimdFloat4 {
        unsafe {
            return _mm_setzero_ps();
        }
    }

    // Returns a SimdFloat4 vector with all components set to 1.
    #[inline]
    pub fn one() -> SimdFloat4 {
        unsafe {
            let zero = _mm_setzero_si128();
            return _mm_castsi128_ps(
                _mm_srli_epi32(_mm_slli_epi32(_mm_cmpeq_epi32(zero, zero), 25), 2));
        }
    }

    // Returns a SimdFloat4 vector with the x component set to 1 and all the others to 0.
    #[inline]
    pub fn x_axis() -> SimdFloat4 {
        unsafe {
            let zero = _mm_setzero_si128();
            let one = _mm_srli_epi32(_mm_slli_epi32(_mm_cmpeq_epi32(zero, zero), 25), 2);
            return _mm_castsi128_ps(_mm_srli_si128(one, 12));
        }
    }

    // Returns a SimdFloat4 vector with the y component set to 1 and all the others to 0.
    #[inline]
    pub fn y_axis() -> SimdFloat4 {
        unsafe {
            let zero = _mm_setzero_si128();
            let one = _mm_srli_epi32(_mm_slli_epi32(_mm_cmpeq_epi32(zero, zero), 25), 2);
            return _mm_castsi128_ps(_mm_slli_si128(_mm_srli_si128(one, 12), 4));
        }
    }

    // Returns a SimdFloat4 vector with the z component set to 1 and all the others to 0.
    #[inline]
    pub fn z_axis() -> SimdFloat4 {
        unsafe {
            let zero = _mm_setzero_si128();
            let one = _mm_srli_epi32(_mm_slli_epi32(_mm_cmpeq_epi32(zero, zero), 25), 2);
            return _mm_castsi128_ps(_mm_slli_si128(_mm_srli_si128(one, 12), 8));
        }
    }

    // Returns a SimdFloat4 vector with the w component set to 1 and all the others to 0.
    #[inline]
    pub fn w_axis() -> SimdFloat4 {
        unsafe {
            let zero = _mm_setzero_si128();
            let one = _mm_srli_epi32(_mm_slli_epi32(_mm_cmpeq_epi32(zero, zero), 25), 2);
            return _mm_castsi128_ps(_mm_slli_si128(one, 12));
        }
    }

    // Loads _x, _y, _z, _w to the returned vector.
    // r.x = _x
    // r.y = _y
    // r.z = _z
    // r.w = _w
    #[inline]
    pub fn load(_x: f32, _y: f32, _z: f32, _w: f32) -> SimdFloat4 {
        unsafe {
            return _mm_set_ps(_w, _z, _y, _x);
        }
    }

    // Loads _x to the x component of the returned vector, and sets y, z and w to 0.
    // r.x = _x
    // r.y = 0
    // r.z = 0
    // r.w = 0
    #[inline]
    pub fn load_x(_x: f32) -> SimdFloat4 {
        unsafe {
            return _mm_set_ss(_x);
        }
    }

    // Loads _x to the all the components of the returned vector.
    // r.x = _x
    // r.y = _x
    // r.z = _x
    // r.w = _x
    #[inline]
    pub fn load1(_x: f32) -> SimdFloat4 {
        unsafe {
            return _mm_set_ps1(_x);
        }
    }

    // Loads the 4 values of _f to the returned vector.
    // _f must be aligned to 16 bytes.
    // r.x = _f[0]
    // r.y = _f[1]
    // r.z = _f[2]
    // r.w = _f[3]
    #[inline]
    pub fn load_ptr(_f: [f32; 4]) -> SimdFloat4 {
        unsafe {
            return _mm_load_ps(_f.as_ptr());
        }
    }

    // Loads the 4 values of _f to the returned vector.
    // _f must be aligned to 4 bytes.
    // r.x = _f[0]
    // r.y = _f[1]
    // r.z = _f[2]
    // r.w = _f[3]
    #[inline]
    pub fn load_ptr_u(_f: [f32; 4]) -> SimdFloat4 {
        unsafe {
            return _mm_loadu_ps(_f.as_ptr());
        }
    }

    // Loads _f[0] to the x component of the returned vector, and sets y, z and w
    // to 0.
    // _f must be aligned to 4 bytes.
    // r.x = _f[0]
    // r.y = 0
    // r.z = 0
    // r.w = 0
    pub fn load_x_ptr_u(_f: [f32; 4]) -> SimdFloat4 {
        unsafe {
            return _mm_load_ss(_f.as_ptr());
        }
    }

    // Loads _f[0] to all the components of the returned vector.
    // _f must be aligned to 4 bytes.
    // r.x = _f[0]
    // r.y = _f[0]
    // r.z = _f[0]
    // r.w = _f[0]
    pub fn load1ptr_u(_f: [f32; 4]) -> SimdFloat4 {
        unsafe {
            return _mm_load_ps1(_f.as_ptr());
        }
    }

    // Loads the 2 first value of _f to the x and y components of the returned
    // vector. The remaining components are set to 0.
    // _f must be aligned to 4 bytes.
    // r.x = _f[0]
    // r.y = _f[1]
    // r.z = 0
    // r.w = 0
    pub fn load2ptr_u(_f: [f32; 4]) -> SimdFloat4 {
        unsafe {
            return _mm_unpacklo_ps(_mm_load_ss(_f.as_ptr().add(0)), _mm_load_ss(_f.as_ptr().add(1)));
        }
    }

    // Loads the 3 first value of _f to the x, y and z components of the returned
    // vector. The remaining components are set to 0.
    // _f must be aligned to 4 bytes.
    // r.x = _f[0]
    // r.y = _f[1]
    // r.z = _f[2]
    // r.w = 0
    pub fn load3ptr_u(_f: [f32; 4]) -> SimdFloat4 {
        unsafe {
            return _mm_movelh_ps(
                _mm_unpacklo_ps(_mm_load_ss(_f.as_ptr().add(0)),
                                _mm_load_ss(_f.as_ptr().add(1))),
                _mm_load_ss(_f.as_ptr().add(2)));
        }
    }

    // Convert from integer to float.
    pub fn from_int(_i: SimdInt4) -> SimdFloat4 {
        unsafe {
            return _mm_cvtepi32_ps(_i);
        }
    }
}

// Returns the x component of _v as a float.
pub fn get_x(_v: SimdFloat4) -> f32 {
    todo!()
}

// Returns the y component of _v as a float.
pub fn get_y(_v: SimdFloat4) -> f32 {
    todo!()
}

// Returns the z component of _v as a float.
pub fn get_z(_v: SimdFloat4) -> f32 {
    todo!()
}

// Returns the w component of _v as a float.
pub fn get_w(_v: SimdFloat4) -> f32 {
    todo!()
}

// Returns _v with the x component set to x component of _f.
pub fn set_x(_v: SimdFloat4, _f: SimdFloat4) -> SimdFloat4 {
    todo!()
}

// Returns _v with the y component set to  x component of _f.
pub fn set_y(_v: SimdFloat4, _f: SimdFloat4) -> SimdFloat4 {
    todo!()
}


// Returns _v with the z component set to  x component of _f.
pub fn set_z(_v: SimdFloat4, _f: SimdFloat4) -> SimdFloat4 {
    todo!()
}


// Returns _v with the w component set to  x component of _f.
pub fn set_w(_v: SimdFloat4, _f: SimdFloat4) -> SimdFloat4 {
    todo!()
}