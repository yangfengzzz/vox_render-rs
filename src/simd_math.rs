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

#[macro_export]
macro_rules! _mm_shuffle {
        ($z:expr, $y:expr, $x:expr, $w:expr) => {
            (($z << 6) | ($y << 4) | ($x << 2) | $w)
        };
    }

#[macro_export]
macro_rules! ozz_shuffle_ps1 {
        ($_v:expr, $_m:expr) => {
            _mm_shuffle_ps($_v, $_v, $_m)
        };
    }

#[macro_export]
macro_rules! ozz_sse_splat_f {
        ($_v:expr, $_i:expr) => {
            ozz_shuffle_ps1!($_v, _mm_shuffle!($_i,$_i,$_i,$_i))
        };
    }

#[macro_export]
macro_rules! ozz_sse_splat_i {
        ($_v:expr, $_i:expr) => {
            _mm_shuffle_epi32($_v, _mm_shuffle!($_i, $_i, $_i, $_i))
        };
    }

#[macro_export]
macro_rules! ozz_sse_hadd2_f {
        ($_v:expr) => {
            _mm_add_ss($_v, ozz_sse_splat_i($_v, 1))
        };
    }

#[macro_export]
macro_rules! ozz_sse_hadd3_f {
        ($_v:expr) => {
            _mm_add_ss(_mm_add_ss($_v, ozz_sse_splat_f($_v, 2)), ozz_sse_splat_f($_v, 1))
        };
    }

#[macro_export]
macro_rules! ozz_sse_hadd4_f {
        ($_v:expr, $_r:expr) => {
            {
                let haddxyzw = _mm_add_ps($_v, _mm_movehl_ps($_v, $_v));
                $_r = _mm_add_ss(haddxyzw, ozz_sse_splat_f(haddxyzw, 1));
            }
        };
    }

#[macro_export]
macro_rules! ozz_sse_dot2_f {
        ($_a:expr, $_b:expr, $_r:expr) => {
            {
                let ab = _mm_mul_ps($_a, $_b);
                $_r = _mm_add_ss(ab, ozz_sse_splat_f(ab, 1));
            }
        };
    }

#[macro_export]
macro_rules! ozz_sse_dot3_f {
        ($_a:expr, $_b:expr, $_r:expr) => {
            $_r = _mm_dp_ps($_a, $_b, 0x7f);
        };
    }

#[macro_export]
macro_rules! ozz_sse_dot4_f {
        ($_a:expr, $_b:expr, $_r:expr) => {
            $_r = _mm_dp_ps($_a, $_b, 0xff);
        };
    }

#[macro_export]
macro_rules! ozz_madd {
        ($_a:expr, $_b:expr, $_c:expr) => {
            _mm_add_ps(_mm_mul_ps($_a, $_b), $_c)
        };
    }

#[macro_export]
macro_rules! ozz_msub {
        ($_a:expr, $_b:expr, $_c:expr) => {
            _mm_sub_ps(_mm_mul_ps($_a, $_b), $_c)
        };
    }

#[macro_export]
macro_rules! ozz_nmadd {
        ($_a:expr, $_b:expr, $_c:expr) => {
            _mm_sub_ps($_c, _mm_mul_ps($_a, $_b))
        };
    }

#[macro_export]
macro_rules! ozz_nmsub {
        ($_a:expr, $_b:expr, $_c:expr) => {
            (-_mm_add_ps(_mm_mul_ps($_a, $_b), $_c))
        };
    }

#[macro_export]
macro_rules! ozz_maddx {
        ($_a:expr, $_b:expr, $_c:expr) => {
            _mm_add_ss(_mm_mul_ss($_a, $_b), $_c)
        };
    }

#[macro_export]
macro_rules! ozz_msubx {
        ($_a:expr, $_b:expr, $_c:expr) => {
            _mm_sub_ss(_mm_mul_ss($_a, $_b), $_c)
        };
    }

#[macro_export]
macro_rules! ozz_nmaddx {
        ($_a:expr, $_b:expr, $_c:expr) => {
            _mm_sub_ss($_c, _mm_mul_ss($_a, $_b))
        };
    }

#[macro_export]
macro_rules! ozz_nmsubx {
        ($_a:expr, $_b:expr, $_c:expr) => {
            (-_mm_add_ss(_mm_mul_ss($_a, $_b), $_c))
        };
    }

#[macro_export]
macro_rules! ozz_sse_select_f {
        ($_b:expr, $_true:expr, $_false:expr) => {
            _mm_blendv_ps($_false, $_true, _mm_castsi128_ps($_b))
        };
    }

#[macro_export]
macro_rules! ozz_sse_select_i {
        ($_b:expr, $_true:expr, $_false:expr) => {
            _mm_blendv_epi8($_false, $_true, $_b)
        };
    }

//----------------------------------------------------------------------------------------------
pub mod simd_float4 {
    use std::arch::x86_64::*;
    use crate::simd_math::{SimdInt4, SimdFloat4};

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
    #[inline]
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
    #[inline]
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
    #[inline]
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
    #[inline]
    pub fn load3ptr_u(_f: [f32; 4]) -> SimdFloat4 {
        unsafe {
            return _mm_movelh_ps(
                _mm_unpacklo_ps(_mm_load_ss(_f.as_ptr().add(0)),
                                _mm_load_ss(_f.as_ptr().add(1))),
                _mm_load_ss(_f.as_ptr().add(2)));
        }
    }

    // Convert from integer to float.
    #[inline]
    pub fn from_int(_i: SimdInt4) -> SimdFloat4 {
        unsafe {
            return _mm_cvtepi32_ps(_i);
        }
    }
}

// Returns the x component of _v as a float.
#[inline]
pub fn get_x(_v: SimdFloat4) -> f32 {
    unsafe {
        return _mm_cvtss_f32(_v);
    }
}

// Returns the y component of _v as a float.
#[inline]
pub fn get_y(_v: SimdFloat4) -> f32 {
    unsafe {
        return _mm_cvtss_f32(ozz_sse_splat_f!(_v, 1));
    }
}

// Returns the z component of _v as a float.
#[inline]
pub fn get_z(_v: SimdFloat4) -> f32 {
    unsafe {
        return _mm_cvtss_f32(_mm_movehl_ps(_v, _v));
    }
}

// Returns the w component of _v as a float.
#[inline]
pub fn get_w(_v: SimdFloat4) -> f32 {
    unsafe {
        return _mm_cvtss_f32(ozz_sse_splat_f!(_v, 3));
    }
}

// Returns _v with the x component set to x component of _f.
#[inline]
pub fn set_x(_v: SimdFloat4, _f: SimdFloat4) -> SimdFloat4 {
    unsafe {
        return _mm_move_ss(_v, _f);
    }
}

// Returns _v with the y component set to  x component of _f.
#[inline]
pub fn set_y(_v: SimdFloat4, _f: SimdFloat4) -> SimdFloat4 {
    unsafe {
        let xfnn = _mm_unpacklo_ps(_v, _f);
        return _mm_shuffle_ps(xfnn, _v, _mm_shuffle!(3, 2, 1, 0));
    }
}


// Returns _v with the z component set to  x component of _f.
#[inline]
pub fn set_z(_v: SimdFloat4, _f: SimdFloat4) -> SimdFloat4 {
    unsafe {
        let ffww = _mm_shuffle_ps(_f, _v, _mm_shuffle!(3, 3, 0, 0));
        return _mm_shuffle_ps(_v, ffww, _mm_shuffle!(2, 0, 1, 0));
    }
}


// Returns _v with the w component set to  x component of _f.
#[inline]
pub fn set_w(_v: SimdFloat4, _f: SimdFloat4) -> SimdFloat4 {
    unsafe {
        let ffzz = _mm_shuffle_ps(_f, _v, _mm_shuffle!(2, 2, 0, 0));
        return _mm_shuffle_ps(_v, ffzz, _mm_shuffle!(0, 2, 1, 0));
    }
}

// Returns _v with the _i th component set to _f.
// _i must be in range [0,3]
pub union SimdFloat4Union {
    ret: SimdFloat4,
    af: [f32; 4],
}

#[inline]
pub fn set_i(_v: SimdFloat4, _f: SimdFloat4, _ith: usize) -> SimdFloat4 {
    unsafe {
        let mut u = SimdFloat4Union {
            ret: _v,
        };

        u.af[_ith] = _mm_cvtss_f32(_f);
        return u.ret;
    }
}

// Stores the 4 components of _v to the four first floats of _f.
// _f must be aligned to 16 bytes.
// _f[0] = _v.x
// _f[1] = _v.y
// _f[2] = _v.z
// _f[3] = _v.w
#[inline]
pub fn store_ptr(_v: SimdFloat4, _f: &mut [f32; 4]) {
    todo!()
}

// Stores the x component of _v to the first float of _f.
// _f must be aligned to 16 bytes.
// _f[0] = _v.x
#[inline]
pub fn store1ptr(_v: SimdFloat4, _f: &mut [f32; 4]) {
    todo!()
}

// Stores x and y components of _v to the two first floats of _f.
// _f must be aligned to 16 bytes.
// _f[0] = _v.x
// _f[1] = _v.y
#[inline]
pub fn store2ptr(_v: SimdFloat4, _f: &mut [f32; 4]) {
    todo!()
}

// Stores x, y and z components of _v to the three first floats of _f.
// _f must be aligned to 16 bytes.
// _f[0] = _v.x
// _f[1] = _v.y
// _f[2] = _v.z
#[inline]
pub fn store3ptr(_v: SimdFloat4, _f: &mut [f32; 4]) {
    todo!()
}

// Stores the 4 components of _v to the four first floats of _f.
// _f must be aligned to 4 bytes.
// _f[0] = _v.x
// _f[1] = _v.y
// _f[2] = _v.z
// _f[3] = _v.w
#[inline]
pub fn store_ptr_u(_v: SimdFloat4, _f: &mut [f32; 4]) {
    todo!()
}

// Stores the x component of _v to the first float of _f.
// _f must be aligned to 4 bytes.
// _f[0] = _v.x
#[inline]
pub fn store1ptr_u(_v: SimdFloat4, _f: &mut [f32; 4]) {
    todo!()
}

// Stores x and y components of _v to the two first floats of _f.
// _f must be aligned to 4 bytes.
// _f[0] = _v.x
// _f[1] = _v.y
#[inline]
pub fn store2ptr_u(_v: SimdFloat4, _f: &mut [f32; 4]) {
    todo!()
}

// Stores x, y and z components of _v to the three first floats of _f.
// _f must be aligned to 4 bytes.
// _f[0] = _v.x
// _f[1] = _v.y
// _f[2] = _v.z
#[inline]
pub fn store3ptr_u(_v: SimdFloat4, _f: &mut [f32; 4]) {
    todo!()
}

// Replicates x of _a to all the components of the returned vector.
#[inline]
pub fn splat_x(_v: SimdFloat4) -> SimdFloat4 {
    todo!()
}

// Replicates y of _a to all the components of the returned vector.
#[inline]
pub fn splat_y(_v: SimdFloat4) -> SimdFloat4 {
    todo!()
}

// Replicates z of _a to all the components of the returned vector.
#[inline]
pub fn splat_z(_v: SimdFloat4) -> SimdFloat4 {
    todo!()
}

// Replicates w of _a to all the components of the returned vector.
#[inline]
pub fn splat_w(_v: SimdFloat4) -> SimdFloat4 {
    todo!()
}


// swizzle X, y, z and w components based on compile time arguments _X, _Y, _Z
// and _W. Arguments can vary from 0 (X), to 3 (w).
#[inline]
pub fn swizzle0123(_v: SimdFloat4) -> SimdFloat4 {
    return _v;
}

#[inline]
pub fn swizzle0101(_v: SimdFloat4) -> SimdFloat4 {
    unsafe {
        return _mm_movelh_ps(_v, _v);
    }
}

#[inline]
pub fn swizzle2323(_v: SimdFloat4) -> SimdFloat4 {
    unsafe {
        return _mm_movehl_ps(_v, _v);
    }
}

#[inline]
pub fn swizzle0011(_v: SimdFloat4) -> SimdFloat4 {
    unsafe {
        return _mm_unpacklo_ps(_v, _v);
    }
}

#[inline]
pub fn swizzle2233(_v: SimdFloat4) -> SimdFloat4 {
    unsafe {
        return _mm_unpackhi_ps(_v, _v);
    }
}

// Transposes the x components of the 4 SimdFloat4 of _in into the 1
// SimdFloat4 of _out.
#[inline]
pub fn transpose4x1(_in: [SimdFloat4; 4], _out: &mut [SimdFloat4; 1]) {
    todo!()
}

// Transposes x, y, z and w components of _in to the x components of _out.
// Remaining y, z and w are set to 0.
#[inline]
pub fn transpose1x4(_in: [SimdFloat4; 1], _out: &mut [SimdFloat4; 4]) {
    todo!()
}

// Transposes the 1 SimdFloat4 of _in into the x components of the 4
// SimdFloat4 of _out. Remaining y, z and w are set to 0.
#[inline]
pub fn transpose2x4(_in: [SimdFloat4; 2], _out: &mut [SimdFloat4; 4]) {
    todo!()
}

// Transposes the x and y components of the 4 SimdFloat4 of _in into the 2
// SimdFloat4 of _out.
#[inline]
pub fn transpose4x2(_in: [SimdFloat4; 4], _out: &mut [SimdFloat4; 2]) {
    todo!()
}

// Transposes the x, y and z components of the 4 SimdFloat4 of _in into the 3
// SimdFloat4 of _out.
#[inline]
pub fn transpose4x3(_in: [SimdFloat4; 4], _out: &mut [SimdFloat4; 3]) {
    todo!()
}

// Transposes the 3 SimdFloat4 of _in into the x, y and z components of the 4
// SimdFloat4 of _out. Remaining w are set to 0.
#[inline]
pub fn transpose3x4(_in: [SimdFloat4; 3], _out: &mut [SimdFloat4; 4]) {
    todo!()
}

// Transposes the 4 SimdFloat4 of _in into the 4 SimdFloat4 of _out.
#[inline]
pub fn transpose4x4(_in: [SimdFloat4; 4], _out: &mut [SimdFloat4; 4]) {
    todo!()
}

// Transposes the 16 SimdFloat4 of _in into the 16 SimdFloat4 of _out.
#[inline]
pub fn transpose16x16(_in: [SimdFloat4; 16], _out: &mut [SimdFloat4; 16]) {
    todo!()
}

// Multiplies _a and _b, then adds _c.
// v = (_a * _b) + _c
#[inline]
pub fn madd(_a: SimdFloat4, _b: SimdFloat4, _c: SimdFloat4) -> SimdFloat4 {
    todo!()
}

// Multiplies _a and _b, then subs _c.
// v = (_a * _b) + _c
#[inline]
pub fn msub(_a: SimdFloat4, _b: SimdFloat4, _c: SimdFloat4) -> SimdFloat4 {
    todo!()
}

// Multiplies _a and _b, negate it, then adds _c.
// v = -(_a * _b) + _c
#[inline]
pub fn nmadd(_a: SimdFloat4, _b: SimdFloat4, _c: SimdFloat4) -> SimdFloat4 {
    todo!()
}

// Multiplies _a and _b, negate it, then subs _c.
// v = -(_a * _b) + _c
#[inline]
pub fn nmsub(_a: SimdFloat4, _b: SimdFloat4, _c: SimdFloat4) -> SimdFloat4 {
    todo!()
}

// Divides the x component of _a by the _x component of _b and stores it in the
// x component of the returned vector. y, z, w of the returned vector are the
// same as _a respective components.
// r.x = _a.x / _b.x
// r.y = _a.y
// r.z = _a.z
// r.w = _a.w
#[inline]
pub fn div_x(_a: SimdFloat4, _b: SimdFloat4) -> SimdFloat4 {
    todo!()
}

// Computes the (horizontal) addition of x and y components of _v. The result is
// stored in the x component of the returned value. y, z, w of the returned
// vector are the same as their respective components in _v.
// r.x = _a.x + _a.y
// r.y = _a.y
// r.z = _a.z
// r.w = _a.w
#[inline]
pub fn hadd2(_v: SimdFloat4) -> SimdFloat4 {
    todo!()
}

// Computes the (horizontal) addition of x, y and z components of _v. The result
// is stored in the x component of the returned value. y, z, w of the returned
// vector are the same as their respective components in _v.
// r.x = _a.x + _a.y + _a.z
// r.y = _a.y
// r.z = _a.z
// r.w = _a.w
#[inline]
pub fn hadd3(_v: SimdFloat4) -> SimdFloat4 {
    todo!()
}

// Computes the (horizontal) addition of x and y components of _v. The result is
// stored in the x component of the returned value. y, z, w of the returned
// vector are the same as their respective components in _v.
// r.x = _a.x + _a.y + _a.z + _a.w
// r.y = _a.y
// r.z = _a.z
// r.w = _a.w
#[inline]
pub fn hadd4(_v: SimdFloat4) -> SimdFloat4 {
    todo!()
}

// Computes the dot product of x and y components of _v. The result is
// stored in the x component of the returned value. y, z, w of the returned
// vector are undefined.
// r.x = _a.x * _a.x + _a.y * _a.y
// r.y = ?
// r.z = ?
// r.w = ?
#[inline]
pub fn dot2(_a: SimdFloat4, _b: SimdFloat4) -> SimdFloat4 {
    todo!()
}

// Computes the dot product of x, y and z components of _v. The result is
// stored in the x component of the returned value. y, z, w of the returned
// vector are undefined.
// r.x = _a.x * _a.x + _a.y * _a.y + _a.z * _a.z
// r.y = ?
// r.z = ?
// r.w = ?
#[inline]
pub fn dot3(_a: SimdFloat4, _b: SimdFloat4) -> SimdFloat4 {
    todo!()
}

// Computes the dot product of x, y, z and w components of _v. The result is
// stored in the x component of the returned value. y, z, w of the returned
// vector are undefined.
// r.x = _a.x * _a.x + _a.y * _a.y + _a.z * _a.z + _a.w * _a.w
// r.y = ?
// r.z = ?
// r.w = ?
#[inline]
pub fn dot4(_a: SimdFloat4, _b: SimdFloat4) -> SimdFloat4 {
    todo!()
}

// Computes the cross product of x, y and z components of _v. The result is
// stored in the x, y and z components of the returned value. w of the returned
// vector is undefined.
// r.x = _a.y * _b.z - _a.z * _b.y
// r.y = _a.z * _b.x - _a.x * _b.z
// r.z = _a.x * _b.y - _a.y * _b.x
// r.w = ?
#[inline]
pub fn cross3(_a: SimdFloat4, _b: SimdFloat4) -> SimdFloat4 {
    todo!()
}















