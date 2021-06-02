/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use std::arch::x86_64::*;
use std::ops::{Mul, Add, Sub};

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
            _mm_add_ss($_v, ozz_sse_splat_f!($_v, 1))
        };
    }

#[macro_export]
macro_rules! ozz_sse_hadd3_f {
        ($_v:expr) => {
            _mm_add_ss(_mm_add_ss($_v, ozz_sse_splat_f!($_v, 2)), ozz_sse_splat_f!($_v, 1))
        };
    }

#[macro_export]
macro_rules! ozz_sse_hadd4_f {
        ($_v:expr, $_r:expr) => {
            {
                let haddxyzw = _mm_add_ps($_v, _mm_movehl_ps($_v, $_v));
                $_r = _mm_add_ss(haddxyzw, ozz_sse_splat_f!(haddxyzw, 1));
            }
        };
    }

#[macro_export]
macro_rules! ozz_sse_dot2_f {
        ($_a:expr, $_b:expr, $_r:expr) => {
            {
                let ab = _mm_mul_ps($_a, $_b);
                $_r = _mm_add_ss(ab, ozz_sse_splat_f!(ab, 1));
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

    //----------------------------------------------------------------------------------------------
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
        unsafe {
            _mm_store_ps(_f.as_mut_ptr(), _v);
        }
    }

    // Stores the x component of _v to the first float of _f.
    // _f must be aligned to 16 bytes.
    // _f[0] = _v.x
    #[inline]
    pub fn store1ptr(_v: SimdFloat4, _f: &mut [f32; 4]) {
        unsafe {
            _mm_store_ss(_f.as_mut_ptr(), _v);
        }
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
        unsafe {
            _mm_storeu_ps(_f.as_mut_ptr(), _v);
        }
    }

    // Stores the x component of _v to the first float of _f.
    // _f must be aligned to 4 bytes.
    // _f[0] = _v.x
    #[inline]
    pub fn store1ptr_u(_v: SimdFloat4, _f: &mut [f32; 4]) {
        unsafe {
            _mm_store_ss(_f.as_mut_ptr(), _v);
        }
    }

    // Stores x and y components of _v to the two first floats of _f.
    // _f must be aligned to 4 bytes.
    // _f[0] = _v.x
    // _f[1] = _v.y
    #[inline]
    pub fn store2ptr_u(_v: SimdFloat4, _f: &mut [f32; 4]) {
        unsafe {
            _mm_store_ss(_f.as_mut_ptr().add(0), _v);
            _mm_store_ss(_f.as_mut_ptr().add(1), ozz_sse_splat_f!(_v, 1));
        }
    }

    // Stores x, y and z components of _v to the three first floats of _f.
    // _f must be aligned to 4 bytes.
    // _f[0] = _v.x
    // _f[1] = _v.y
    // _f[2] = _v.z
    #[inline]
    pub fn store3ptr_u(_v: SimdFloat4, _f: &mut [f32; 4]) {
        unsafe {
            _mm_store_ss(_f.as_mut_ptr().add(0), _v);
            _mm_store_ss(_f.as_mut_ptr().add(1), ozz_sse_splat_f!(_v, 1));
            _mm_store_ss(_f.as_mut_ptr().add(2), _mm_movehl_ps(_v, _v));
        }
    }

    // Replicates x of _a to all the components of the returned vector.
    #[inline]
    pub fn splat_x(_v: SimdFloat4) -> SimdFloat4 {
        unsafe {
            return ozz_sse_splat_f!(_v, 0);
        }
    }

    // Replicates y of _a to all the components of the returned vector.
    #[inline]
    pub fn splat_y(_v: SimdFloat4) -> SimdFloat4 {
        unsafe {
            return ozz_sse_splat_f!(_v, 1);
        }
    }

    // Replicates z of _a to all the components of the returned vector.
    #[inline]
    pub fn splat_z(_v: SimdFloat4) -> SimdFloat4 {
        unsafe {
            return ozz_sse_splat_f!(_v, 2);
        }
    }

    // Replicates w of _a to all the components of the returned vector.
    #[inline]
    pub fn splat_w(_v: SimdFloat4) -> SimdFloat4 {
        unsafe {
            return ozz_sse_splat_f!(_v, 3);
        }
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
        unsafe {
            let xz = _mm_unpacklo_ps(_in[0], _in[2]);
            let yw = _mm_unpacklo_ps(_in[1], _in[3]);
            _out[0] = _mm_unpacklo_ps(xz, yw);
        }
    }

    // Transposes x, y, z and w components of _in to the x components of _out.
    // Remaining y, z and w are set to 0.
    #[inline]
    pub fn transpose1x4(_in: [SimdFloat4; 1], _out: &mut [SimdFloat4; 4]) {
        unsafe {
            let zwzw = _mm_movehl_ps(_in[0], _in[0]);
            let yyyy = ozz_sse_splat_f!(_in[0], 1);
            let wwww = ozz_sse_splat_f!(_in[0], 3);
            let zero = _mm_setzero_ps();
            _out[0] = _mm_move_ss(zero, _in[0]);
            _out[1] = _mm_move_ss(zero, yyyy);
            _out[2] = _mm_move_ss(zero, zwzw);
            _out[3] = _mm_move_ss(zero, wwww);
        }
    }

    // Transposes the 1 SimdFloat4 of _in into the x components of the 4
    // SimdFloat4 of _out. Remaining y, z and w are set to 0.
    #[inline]
    pub fn transpose2x4(_in: [SimdFloat4; 2], _out: &mut [SimdFloat4; 4]) {
        unsafe {
            let tmp0 = _mm_unpacklo_ps(_in[0], _in[1]);
            let tmp1 = _mm_unpackhi_ps(_in[0], _in[1]);
            let zero = _mm_setzero_ps();
            _out[0] = _mm_movelh_ps(tmp0, zero);
            _out[1] = _mm_movehl_ps(zero, tmp0);
            _out[2] = _mm_movelh_ps(tmp1, zero);
            _out[3] = _mm_movehl_ps(zero, tmp1);
        }
    }

    // Transposes the x and y components of the 4 SimdFloat4 of _in into the 2
    // SimdFloat4 of _out.
    #[inline]
    pub fn transpose4x2(_in: [SimdFloat4; 4], _out: &mut [SimdFloat4; 2]) {
        unsafe {
            let tmp0 = _mm_unpacklo_ps(_in[0], _in[2]);
            let tmp1 = _mm_unpacklo_ps(_in[1], _in[3]);
            _out[0] = _mm_unpacklo_ps(tmp0, tmp1);
            _out[1] = _mm_unpackhi_ps(tmp0, tmp1);
        }
    }

    // Transposes the x, y and z components of the 4 SimdFloat4 of _in into the 3
    // SimdFloat4 of _out.
    #[inline]
    pub fn transpose4x3(_in: [SimdFloat4; 4], _out: &mut [SimdFloat4; 3]) {
        unsafe {
            let tmp0 = _mm_unpacklo_ps(_in[0], _in[2]);
            let tmp1 = _mm_unpacklo_ps(_in[1], _in[3]);
            let tmp2 = _mm_unpackhi_ps(_in[0], _in[2]);
            let tmp3 = _mm_unpackhi_ps(_in[1], _in[3]);
            _out[0] = _mm_unpacklo_ps(tmp0, tmp1);
            _out[1] = _mm_unpackhi_ps(tmp0, tmp1);
            _out[2] = _mm_unpacklo_ps(tmp2, tmp3);
        }
    }

    // Transposes the 3 SimdFloat4 of _in into the x, y and z components of the 4
    // SimdFloat4 of _out. Remaining w are set to 0.
    #[inline]
    pub fn transpose3x4(_in: [SimdFloat4; 3], _out: &mut [SimdFloat4; 4]) {
        unsafe {
            let zero = _mm_setzero_ps();
            let temp0 = _mm_unpacklo_ps(_in[0], _in[1]);
            let temp1 = _mm_unpacklo_ps(_in[2], zero);
            let temp2 = _mm_unpackhi_ps(_in[0], _in[1]);
            let temp3 = _mm_unpackhi_ps(_in[2], zero);
            _out[0] = _mm_movelh_ps(temp0, temp1);
            _out[1] = _mm_movehl_ps(temp1, temp0);
            _out[2] = _mm_movelh_ps(temp2, temp3);
            _out[3] = _mm_movehl_ps(temp3, temp2);
        }
    }

    // Transposes the 4 SimdFloat4 of _in into the 4 SimdFloat4 of _out.
    #[inline]
    pub fn transpose4x4(_in: [SimdFloat4; 4], _out: &mut [SimdFloat4; 4]) {
        unsafe {
            let tmp0 = _mm_unpacklo_ps(_in[0], _in[2]);
            let tmp1 = _mm_unpacklo_ps(_in[1], _in[3]);
            let tmp2 = _mm_unpackhi_ps(_in[0], _in[2]);
            let tmp3 = _mm_unpackhi_ps(_in[1], _in[3]);
            _out[0] = _mm_unpacklo_ps(tmp0, tmp1);
            _out[1] = _mm_unpackhi_ps(tmp0, tmp1);
            _out[2] = _mm_unpacklo_ps(tmp2, tmp3);
            _out[3] = _mm_unpackhi_ps(tmp2, tmp3);
        }
    }

    // Transposes the 16 SimdFloat4 of _in into the 16 SimdFloat4 of _out.
    #[inline]
    pub fn transpose16x16(_in: [SimdFloat4; 16], _out: &mut [SimdFloat4; 16]) {
        unsafe {
            let tmp0 = _mm_unpacklo_ps(_in[0], _in[2]);
            let tmp1 = _mm_unpacklo_ps(_in[1], _in[3]);
            _out[0] = _mm_unpacklo_ps(tmp0, tmp1);
            _out[4] = _mm_unpackhi_ps(tmp0, tmp1);
            let tmp2 = _mm_unpackhi_ps(_in[0], _in[2]);
            let tmp3 = _mm_unpackhi_ps(_in[1], _in[3]);
            _out[8] = _mm_unpacklo_ps(tmp2, tmp3);
            _out[12] = _mm_unpackhi_ps(tmp2, tmp3);
            let tmp4 = _mm_unpacklo_ps(_in[4], _in[6]);
            let tmp5 = _mm_unpacklo_ps(_in[5], _in[7]);
            _out[1] = _mm_unpacklo_ps(tmp4, tmp5);
            _out[5] = _mm_unpackhi_ps(tmp4, tmp5);
            let tmp6 = _mm_unpackhi_ps(_in[4], _in[6]);
            let tmp7 = _mm_unpackhi_ps(_in[5], _in[7]);
            _out[9] = _mm_unpacklo_ps(tmp6, tmp7);
            _out[13] = _mm_unpackhi_ps(tmp6, tmp7);
            let tmp8 = _mm_unpacklo_ps(_in[8], _in[10]);
            let tmp9 = _mm_unpacklo_ps(_in[9], _in[11]);
            _out[2] = _mm_unpacklo_ps(tmp8, tmp9);
            _out[6] = _mm_unpackhi_ps(tmp8, tmp9);
            let tmp10 = _mm_unpackhi_ps(_in[8], _in[10]);
            let tmp11 = _mm_unpackhi_ps(_in[9], _in[11]);
            _out[10] = _mm_unpacklo_ps(tmp10, tmp11);
            _out[14] = _mm_unpackhi_ps(tmp10, tmp11);
            let tmp12 = _mm_unpacklo_ps(_in[12], _in[14]);
            let tmp13 = _mm_unpacklo_ps(_in[13], _in[15]);
            _out[3] = _mm_unpacklo_ps(tmp12, tmp13);
            _out[7] = _mm_unpackhi_ps(tmp12, tmp13);
            let tmp14 = _mm_unpackhi_ps(_in[12], _in[14]);
            let tmp15 = _mm_unpackhi_ps(_in[13], _in[15]);
            _out[11] = _mm_unpacklo_ps(tmp14, tmp15);
            _out[15] = _mm_unpackhi_ps(tmp14, tmp15);
        }
    }

    // Multiplies _a and _b, then adds _c.
    // v = (_a * _b) + _c
    #[inline]
    pub fn madd(_a: SimdFloat4, _b: SimdFloat4, _c: SimdFloat4) -> SimdFloat4 {
        unsafe {
            return ozz_madd!(_a, _b, _c);
        }
    }

    // Multiplies _a and _b, then subs _c.
    // v = (_a * _b) + _c
    #[inline]
    pub fn msub(_a: SimdFloat4, _b: SimdFloat4, _c: SimdFloat4) -> SimdFloat4 {
        unsafe {
            return ozz_msub!(_a, _b, _c);
        }
    }

    // Multiplies _a and _b, negate it, then adds _c.
    // v = -(_a * _b) + _c
    #[inline]
    pub fn nmadd(_a: SimdFloat4, _b: SimdFloat4, _c: SimdFloat4) -> SimdFloat4 {
        unsafe {
            return ozz_nmadd!(_a, _b, _c);
        }
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
        unsafe {
            return _mm_div_ss(_a, _b);
        }
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
        unsafe {
            return ozz_sse_hadd2_f!(_v);
        }
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
        unsafe {
            return ozz_sse_hadd3_f!(_v);
        }
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
        unsafe {
            let hadd4: __m128;
            ozz_sse_hadd4_f!(_v, hadd4);
            return hadd4;
        }
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
        unsafe {
            let dot2: __m128;
            ozz_sse_dot2_f!(_a, _b, dot2);
            return dot2;
        }
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
        unsafe {
            let dot3: __m128;
            ozz_sse_dot3_f!(_a, _b, dot3);
            return dot3;
        }
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
        unsafe {
            let dot4: __m128;
            ozz_sse_dot4_f!(_a, _b, dot4);
            return dot4;
        }
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
        unsafe {
            // Implementation with 3 shuffles only is based on:
            // https://geometrian.com/programming/tutorials/cross-product
            let shufa = ozz_shuffle_ps1!(_a, _mm_shuffle!(3, 0, 2, 1));
            let shufb = ozz_shuffle_ps1!(_b, _mm_shuffle!(3, 0, 2, 1));
            let shufc = ozz_msub!(_a, shufb, _mm_mul_ps(_b, shufa));
            return ozz_shuffle_ps1!(shufc, _mm_shuffle!(3, 0, 2, 1));
        }
    }

    // Returns the per component estimated reciprocal of _v.
    #[inline]
    pub fn rcp_est(_v: SimdFloat4) -> SimdFloat4 {
        unsafe {
            return _mm_rcp_ps(_v);
        }
    }

    // Returns the per component estimated reciprocal of _v, where approximation is
    // improved with one more new Newton-Raphson step.
    #[inline]
    pub fn rcp_est_nr(_v: SimdFloat4) -> SimdFloat4 {
        unsafe {
            let nr = _mm_rcp_ps(_v);
            // Do one more Newton-Raphson step to improve precision.
            return ozz_nmadd!(_mm_mul_ps(nr, nr), _v, _mm_add_ps(nr, nr));
        }
    }

    // Returns the estimated reciprocal of the x component of _v and stores it in
    // the x component of the returned vector. y, z, w of the returned vector are
    // the same as their respective components in _v.
    #[inline]
    pub fn rcp_est_x(_v: SimdFloat4) -> SimdFloat4 {
        unsafe {
            return _mm_rcp_ss(_v);
        }
    }

    // Returns the estimated reciprocal of the x component of _v, where
    // approximation is improved with one more new Newton-Raphson step. y, z, w of
    // the returned vector are undefined.
    #[inline]
    pub fn rcp_est_xnr(_v: SimdFloat4) -> SimdFloat4 {
        unsafe {
            let nr = _mm_rcp_ss(_v);
            // Do one more Newton-Raphson step to improve precision.
            return ozz_nmaddx!(_mm_mul_ss(nr, nr), _v, _mm_add_ss(nr, nr));
        }
    }

    // Returns the per component square root of _v.
    #[inline]
    pub fn sqrt(_v: SimdFloat4) -> SimdFloat4 {
        unsafe {
            return _mm_sqrt_ps(_v);
        }
    }

    // Returns the square root of the x component of _v and stores it in the x
    // component of the returned vector. y, z, w of the returned vector are the
    // same as their respective components in _v.
    #[inline]
    pub fn sqrt_x(_v: SimdFloat4) -> SimdFloat4 {
        unsafe {
            return _mm_sqrt_ss(_v);
        }
    }

    // Returns the per component estimated reciprocal square root of _v.
    #[inline]
    pub fn rsqrt_est(_v: SimdFloat4) -> SimdFloat4 {
        unsafe {
            return _mm_rsqrt_ps(_v);
        }
    }

    // Returns the per component estimated reciprocal square root of _v, where
    // approximation is improved with one more new Newton-Raphson step.
    #[inline]
    pub fn rsqrt_est_nr(_v: SimdFloat4) -> SimdFloat4 {
        unsafe {
            let nr = _mm_rsqrt_ps(_v);
            // Do one more Newton-Raphson step to improve precision.
            return _mm_mul_ps(_mm_mul_ps(_mm_set_ps1(0.5), nr),
                              ozz_nmadd!(_mm_mul_ps(_v, nr), nr, _mm_set_ps1(3.0)));
        }
    }

    // Returns the estimated reciprocal square root of the x component of _v and
    // stores it in the x component of the returned vector. y, z, w of the returned
    // vector are the same as their respective components in _v.
    #[inline]
    pub fn rsqrt_est_x(_v: SimdFloat4) -> SimdFloat4 {
        unsafe {
            return _mm_rsqrt_ss(_v);
        }
    }

    // Returns the estimated reciprocal square root of the x component of _v, where
    // approximation is improved with one more new Newton-Raphson step. y, z, w of
    // the returned vector are undefined.
    #[inline]
    pub fn rsqrt_est_xnr(_v: SimdFloat4) -> SimdFloat4 {
        unsafe {
            let nr = _mm_rsqrt_ss(_v);
            // Do one more Newton-Raphson step to improve precision.
            return _mm_mul_ss(_mm_mul_ss(_mm_set_ps1(0.5), nr),
                              ozz_nmaddx!(_mm_mul_ss(_v, nr), nr, _mm_set_ps1(3.0)));
        }
    }

    // Returns the per element absolute value of _v.
    #[inline]
    pub fn abs(_v: SimdFloat4) -> SimdFloat4 {
        unsafe {
            let zero = _mm_setzero_si128();
            return _mm_and_ps(
                _mm_castsi128_ps(_mm_srli_epi32(_mm_cmpeq_epi32(zero, zero), 1)), _v);
        }
    }

    // Returns the sign bit of _v.
    #[inline]
    pub fn sign(_v: SimdFloat4) -> SimdInt4 {
        unsafe {
            return _mm_slli_epi32(_mm_srli_epi32(_mm_castps_si128(_v), 31), 31);
        }
    }

    // Returns the per component minimum of _a and _b.
    #[inline]
    pub fn min(_a: SimdFloat4, _b: SimdFloat4) -> SimdFloat4 {
        todo!()
    }

    // Returns the per component maximum of _a and _b.
    #[inline]
    pub fn max(_a: SimdFloat4, _b: SimdFloat4) -> SimdFloat4 {
        todo!()
    }

    // Returns the per component minimum of _v and 0.
    #[inline]
    pub fn min0(_v: SimdFloat4) -> SimdFloat4 {
        todo!()
    }

    // Returns the per component maximum of _v and 0.
    #[inline]
    pub fn max0(_v: SimdFloat4) -> SimdFloat4 {
        todo!()
    }

    // Clamps each element of _x between _a and _b.
    // Result is unknown if _a is not less or equal to _b.
    #[inline]
    pub fn clamp(_a: SimdFloat4, _v: SimdFloat4, _b: SimdFloat4) -> SimdFloat4 {
        todo!()
    }

    // Computes the length of the components x and y of _v, and stores it in the x
    // component of the returned vector. y, z, w of the returned vector are
    // undefined.
    #[inline]
    pub fn length2(_v: SimdFloat4) -> SimdFloat4 {
        unsafe {
            let sq_len;
            ozz_sse_dot2_f!(_v, _v, sq_len);
            return _mm_sqrt_ss(sq_len);
        }
    }

    // Computes the length of the components x, y and z of _v, and stores it in the
    // x component of the returned vector. undefined.
    #[inline]
    pub fn length3(_v: SimdFloat4) -> SimdFloat4 {
        unsafe {
            let sq_len;
            ozz_sse_dot3_f!(_v, _v, sq_len);
            return _mm_sqrt_ss(sq_len);
        }
    }

    // Computes the length of _v, and stores it in the x component of the returned
    // vector. y, z, w of the returned vector are undefined.
    #[inline]
    pub fn length4(_v: SimdFloat4) -> SimdFloat4 {
        unsafe {
            let sq_len;
            ozz_sse_dot4_f!(_v, _v, sq_len);
            return _mm_sqrt_ss(sq_len);
        }
    }

    // Computes the square length of the components x and y of _v, and stores it
    // in the x component of the returned vector. y, z, w of the returned vector are
    // undefined.
    #[inline]
    pub fn length2sqr(_v: SimdFloat4) -> SimdFloat4 {
        unsafe {
            let sq_len;
            ozz_sse_dot2_f!(_v, _v, sq_len);
            return sq_len;
        }
    }


    // Computes the square length of the components x, y and z of _v, and stores it
    // in the x component of the returned vector. y, z, w of the returned vector are
    // undefined.
    #[inline]
    pub fn length3sqr(_v: SimdFloat4) -> SimdFloat4 {
        unsafe {
            let sq_len;
            ozz_sse_dot3_f!(_v, _v, sq_len);
            return sq_len;
        }
    }


    // Computes the square length of the components x, y, z and w of _v, and stores
    // it in the x component of the returned vector. y, z, w of the returned vector
    // undefined.
    #[inline]
    pub fn length4sqr(_v: SimdFloat4) -> SimdFloat4 {
        unsafe {
            let sq_len;
            ozz_sse_dot4_f!(_v, _v, sq_len);
            return sq_len;
        }
    }


    // Returns the normalized vector of the components x and y of _v, and stores
    // it in the x and y components of the returned vector. z and w of the returned
    // vector are the same as their respective components in _v.
    #[inline]
    pub fn normalize2(_v: SimdFloat4) -> SimdFloat4 {
        unsafe {
            let sq_len;
            ozz_sse_dot2_f!(_v, _v, sq_len);
            debug_assert!(_mm_cvtss_f32(sq_len) != 0.0 && "_v is not normalizable".parse().unwrap());
            let inv_len = _mm_div_ss(one(), _mm_sqrt_ss(sq_len));
            let inv_lenxxxx = ozz_sse_splat_f!(inv_len, 0);
            let norm = _mm_mul_ps(_v, inv_lenxxxx);
            return _mm_movelh_ps(norm, _mm_movehl_ps(_v, _v));
        }
    }


    // Returns the normalized vector of the components x, y and z of _v, and stores
    // it in the x, y and z components of the returned vector. w of the returned
    // vector is the same as its respective component in _v.
    #[inline]
    pub fn normalize3(_v: SimdFloat4) -> SimdFloat4 {
        unsafe {
            let sq_len;
            ozz_sse_dot3_f!(_v, _v, sq_len);
            debug_assert!(_mm_cvtss_f32(sq_len) != 0.0 && "_v is not normalizable".parse().unwrap());
            let inv_len = _mm_div_ss(one(), _mm_sqrt_ss(sq_len));
            let vwxyz = ozz_shuffle_ps1!(_v, _mm_shuffle!(0, 1, 2, 3));
            let inv_lenxxxx = ozz_sse_splat_f!(inv_len, 0);
            let normwxyz = _mm_move_ss(_mm_mul_ps(vwxyz, inv_lenxxxx), vwxyz);
            return ozz_shuffle_ps1!(normwxyz, _mm_shuffle!(0, 1, 2, 3));
        }
    }


    // Returns the normalized vector _v.
    #[inline]
    pub fn normalize4(_v: SimdFloat4) -> SimdFloat4 {
        unsafe {
            let sq_len;
            ozz_sse_dot4_f!(_v, _v, sq_len);
            debug_assert!(_mm_cvtss_f32(sq_len) != 0.0 && "_v is not normalizable".parse().unwrap());
            let inv_len = _mm_div_ss(one(), _mm_sqrt_ss(sq_len));
            let inv_lenxxxx = ozz_sse_splat_f!(inv_len, 0);
            return _mm_mul_ps(_v, inv_lenxxxx);
        }
    }


    // Returns the estimated normalized vector of the components x and y of _v, and
    // stores it in the x and y components of the returned vector. z and w of the
    // returned vector are the same as their respective components in _v.
    #[inline]
    pub fn normalize_est2(_v: SimdFloat4) -> SimdFloat4 {
        unsafe {
            let sq_len;
            ozz_sse_dot2_f!(_v, _v, sq_len);
            debug_assert!(_mm_cvtss_f32(sq_len) != 0.0 && "_v is not normalizable".parse().unwrap());
            let inv_len = _mm_rsqrt_ss(sq_len);
            let inv_lenxxxx = ozz_sse_splat_f!(inv_len, 0);
            let norm = _mm_mul_ps(_v, inv_lenxxxx);
            return _mm_movelh_ps(norm, _mm_movehl_ps(_v, _v));
        }
    }


    // Returns the estimated normalized vector of the components x, y and z of _v,
    // and stores it in the x, y and z components of the returned vector. w of the
    // returned vector is the same as its respective component in _v.
    #[inline]
    pub fn normalize_est3(_v: SimdFloat4) -> SimdFloat4 {
        unsafe {
            let sq_len;
            ozz_sse_dot3_f!(_v, _v, sq_len);
            debug_assert!(_mm_cvtss_f32(sq_len) != 0.0 && "_v is not normalizable".parse().unwrap());
            let inv_len = _mm_rsqrt_ss(sq_len);
            let vwxyz = ozz_shuffle_ps1!(_v, _mm_shuffle!(0, 1, 2, 3));
            let inv_lenxxxx = ozz_sse_splat_f!(inv_len, 0);
            let normwxyz = _mm_move_ss(_mm_mul_ps(vwxyz, inv_lenxxxx), vwxyz);
            return ozz_shuffle_ps1!(normwxyz, _mm_shuffle!(0, 1, 2, 3));
        }
    }


    // Returns the estimated normalized vector _v.
    #[inline]
    pub fn normalize_est4(_v: SimdFloat4) -> SimdFloat4 {
        unsafe {
            let sq_len;
            ozz_sse_dot4_f!(_v, _v, sq_len);
            debug_assert!(_mm_cvtss_f32(sq_len) != 0.0 && "_v is not normalizable".parse().unwrap());
            let inv_len = _mm_rsqrt_ss(sq_len);
            let inv_lenxxxx = ozz_sse_splat_f!(inv_len, 0);
            return _mm_mul_ps(_v, inv_lenxxxx);
        }
    }

    // Tests if the components x and y of _v forms a normalized vector.
    // Returns the result in the x component of the returned vector. y, z and w are
    // set to 0.
    #[inline]
    pub fn is_normalized2(_v: SimdFloat4) -> SimdInt4 {
        unsafe {
            let max = _mm_set_ss(1.0 + crate::math_constant::K_NORMALIZATION_TOLERANCE_SQ);
            let min = _mm_set_ss(1.0 - crate::math_constant::K_NORMALIZATION_TOLERANCE_SQ);
            let dot;
            ozz_sse_dot2_f!(_v, _v, dot);
            let dotx000 = _mm_move_ss(_mm_setzero_ps(), dot);
            return _mm_castps_si128(
                _mm_and_ps(_mm_cmplt_ss(dotx000, max), _mm_cmpgt_ss(dotx000, min)));
        }
    }

    // Tests if the components x, y and z of _v forms a normalized vector.
    // Returns the result in the x component of the returned vector. y, z and w are
    // set to 0.
    #[inline]
    pub fn is_normalized3(_v: SimdFloat4) -> SimdInt4 {
        unsafe {
            let max = _mm_set_ss(1.0 + crate::math_constant::K_NORMALIZATION_TOLERANCE_SQ);
            let min = _mm_set_ss(1.0 - crate::math_constant::K_NORMALIZATION_TOLERANCE_SQ);
            let dot;
            ozz_sse_dot3_f!(_v, _v, dot);
            let dotx000 = _mm_move_ss(_mm_setzero_ps(), dot);
            return _mm_castps_si128(
                _mm_and_ps(_mm_cmplt_ss(dotx000, max), _mm_cmpgt_ss(dotx000, min)));
        }
    }

    // Tests if the _v is a normalized vector.
    // Returns the result in the x component of the returned vector. y, z and w are
    // set to 0.
    #[inline]
    pub fn is_normalized4(_v: SimdFloat4) -> SimdInt4 {
        unsafe {
            let max = _mm_set_ss(1.0 + crate::math_constant::K_NORMALIZATION_TOLERANCE_SQ);
            let min = _mm_set_ss(1.0 - crate::math_constant::K_NORMALIZATION_TOLERANCE_SQ);
            let dot;
            ozz_sse_dot4_f!(_v, _v, dot);
            let dotx000 = _mm_move_ss(_mm_setzero_ps(), dot);
            return _mm_castps_si128(
                _mm_and_ps(_mm_cmplt_ss(dotx000, max), _mm_cmpgt_ss(dotx000, min)));
        }
    }

    // Tests if the components x and y of _v forms a normalized vector.
    // Uses the estimated normalization coefficient, that matches estimated math
    // functions (RecpEst, MormalizeEst...).
    // Returns the result in the x component of the returned vector. y, z and w are
    // set to 0.
    #[inline]
    pub fn is_normalized_est2(_v: SimdFloat4) -> SimdInt4 {
        unsafe {
            let max = _mm_set_ss(1.0 + crate::math_constant::K_NORMALIZATION_TOLERANCE_EST_SQ);
            let min = _mm_set_ss(1.0 - crate::math_constant::K_NORMALIZATION_TOLERANCE_EST_SQ);
            let dot;
            ozz_sse_dot2_f!(_v, _v, dot);
            let dotx000 = _mm_move_ss(_mm_setzero_ps(), dot);
            return _mm_castps_si128(
                _mm_and_ps(_mm_cmplt_ss(dotx000, max), _mm_cmpgt_ss(dotx000, min)));
        }
    }

    // Tests if the components x, y and z of _v forms a normalized vector.
    // Uses the estimated normalization coefficient, that matches estimated math
    // functions (RecpEst, MormalizeEst...).
    // Returns the result in the x component of the returned vector. y, z and w are
    // set to 0.
    #[inline]
    pub fn is_normalized_est3(_v: SimdFloat4) -> SimdInt4 {
        unsafe {
            let max = _mm_set_ss(1.0 + crate::math_constant::K_NORMALIZATION_TOLERANCE_EST_SQ);
            let min = _mm_set_ss(1.0 - crate::math_constant::K_NORMALIZATION_TOLERANCE_EST_SQ);
            let dot;
            ozz_sse_dot3_f!(_v, _v, dot);
            let dotx000 = _mm_move_ss(_mm_setzero_ps(), dot);
            return _mm_castps_si128(
                _mm_and_ps(_mm_cmplt_ss(dotx000, max), _mm_cmpgt_ss(dotx000, min)));
        }
    }

    // Tests if the _v is a normalized vector.
    // Uses the estimated normalization coefficient, that matches estimated math
    // functions (RecpEst, MormalizeEst...).
    // Returns the result in the x component of the returned vector. y, z and w are
    // set to 0.
    #[inline]
    pub fn is_normalized_est4(_v: SimdFloat4) -> SimdInt4 {
        unsafe {
            let max = _mm_set_ss(1.0 + crate::math_constant::K_NORMALIZATION_TOLERANCE_EST_SQ);
            let min = _mm_set_ss(1.0 - crate::math_constant::K_NORMALIZATION_TOLERANCE_EST_SQ);
            let dot;
            ozz_sse_dot4_f!(_v, _v, dot);
            let dotx000 = _mm_move_ss(_mm_setzero_ps(), dot);
            return _mm_castps_si128(
                _mm_and_ps(_mm_cmplt_ss(dotx000, max), _mm_cmpgt_ss(dotx000, min)));
        }
    }

    // Returns the normalized vector of the components x and y of _v if it is
    // normalizable, otherwise returns _safe. z and w of the returned vector are
    // the same as their respective components in _v.
    #[inline]
    pub fn normalize_safe2(_v: SimdFloat4, _safe: SimdFloat4) -> SimdFloat4 {
        unsafe {
            let sq_len;
            ozz_sse_dot2_f!(_v, _v, sq_len);
            let inv_len = _mm_div_ss(one(), _mm_sqrt_ss(sq_len));
            let inv_lenxxxx = ozz_sse_splat_f!(inv_len, 0);
            let norm = _mm_mul_ps(_v, inv_lenxxxx);
            let cond = _mm_castps_si128(
                _mm_cmple_ps(ozz_sse_splat_f!(sq_len, 0), _mm_setzero_ps()));
            let cfalse = _mm_movelh_ps(norm, _mm_movehl_ps(_v, _v));
            return ozz_sse_select_f!(cond, _safe, cfalse);
        }
    }

    // Returns the normalized vector of the components x, y, z and w of _v if it is
    // normalizable, otherwise returns _safe. w of the returned vector is the same
    // as its respective components in _v.
    #[inline]
    pub fn normalize_safe3(_v: SimdFloat4, _safe: SimdFloat4) -> SimdFloat4 {
        unsafe {
            let sq_len;
            ozz_sse_dot3_f!(_v, _v, sq_len);
            let inv_len = _mm_div_ss(one(), _mm_sqrt_ss(sq_len));
            let vwxyz = ozz_shuffle_ps1!(_v, _mm_shuffle!(0, 1, 2, 3));
            let inv_lenxxxx = ozz_sse_splat_f!(inv_len, 0);
            let normwxyz = _mm_move_ss(_mm_mul_ps(vwxyz, inv_lenxxxx), vwxyz);
            let cond = _mm_castps_si128(
                _mm_cmple_ps(ozz_sse_splat_f!(sq_len, 0), _mm_setzero_ps()));
            let cfalse = ozz_shuffle_ps1!(normwxyz, _mm_shuffle!(0, 1, 2, 3));
            return ozz_sse_select_f!(cond, _safe, cfalse);
        }
    }

    // Returns the normalized vector _v if it is normalizable, otherwise returns
    // _safe.
    #[inline]
    pub fn normalize_safe4(_v: SimdFloat4, _safe: SimdFloat4) -> SimdFloat4 {
        unsafe {
            let sq_len;
            ozz_sse_dot4_f!(_v, _v, sq_len);
            let inv_len = _mm_div_ss(one(), _mm_sqrt_ss(sq_len));
            let inv_lenxxxx = ozz_sse_splat_f!(inv_len, 0);
            let cond = _mm_castps_si128(
                _mm_cmple_ps(ozz_sse_splat_f!(sq_len, 0), _mm_setzero_ps()));
            let cfalse = _mm_mul_ps(_v, inv_lenxxxx);
            return ozz_sse_select_f!(cond, _safe, cfalse);
        }
    }

    // Returns the estimated normalized vector of the components x and y of _v if it
    // is normalizable, otherwise returns _safe. z and w of the returned vector are
    // the same as their respective components in _v.
    #[inline]
    pub fn normalize_safe_est2(_v: SimdFloat4, _safe: SimdFloat4) -> SimdFloat4 {
        unsafe {
            let sq_len;
            ozz_sse_dot2_f!(_v, _v, sq_len);
            let inv_len = _mm_rsqrt_ss(sq_len);
            let inv_lenxxxx = ozz_sse_splat_f!(inv_len, 0);
            let norm = _mm_mul_ps(_v, inv_lenxxxx);
            let cond = _mm_castps_si128(
                _mm_cmple_ps(ozz_sse_splat_f!(sq_len, 0), _mm_setzero_ps()));
            let cfalse = _mm_movelh_ps(norm, _mm_movehl_ps(_v, _v));
            return ozz_sse_select_f!(cond, _safe, cfalse);
        }
    }

    // Returns the estimated normalized vector of the components x, y, z and w of _v
    // if it is normalizable, otherwise returns _safe. w of the returned vector is
    // the same as its respective components in _v.
    #[inline]
    pub fn normalize_safe_est3(_v: SimdFloat4, _safe: SimdFloat4) -> SimdFloat4 {
        unsafe {
            let sq_len;
            ozz_sse_dot3_f!(_v, _v, sq_len);
            let inv_len = _mm_rsqrt_ss(sq_len);
            let vwxyz = ozz_shuffle_ps1!(_v, _mm_shuffle!(0, 1, 2, 3));
            let inv_lenxxxx = ozz_sse_splat_f!(inv_len, 0);
            let normwxyz = _mm_move_ss(_mm_mul_ps(vwxyz, inv_lenxxxx), vwxyz);
            let cond = _mm_castps_si128(
                _mm_cmple_ps(ozz_sse_splat_f!(sq_len, 0), _mm_setzero_ps()));
            let cfalse = ozz_shuffle_ps1!(normwxyz, _mm_shuffle!(0, 1, 2, 3));
            return ozz_sse_select_f!(cond, _safe, cfalse);
        }
    }

    // Returns the estimated normalized vector _v if it is normalizable, otherwise
    // returns _safe.
    #[inline]
    pub fn normalize_safe_est4(_v: SimdFloat4, _safe: SimdFloat4) -> SimdFloat4 {
        unsafe {
            let sq_len;
            ozz_sse_dot4_f!(_v, _v, sq_len);
            let inv_len = _mm_rsqrt_ss(sq_len);
            let inv_lenxxxx = ozz_sse_splat_f!(inv_len, 0);
            let cond = _mm_castps_si128(
                _mm_cmple_ps(ozz_sse_splat_f!(sq_len, 0), _mm_setzero_ps()));
            let cfalse = _mm_mul_ps(_v, inv_lenxxxx);
            return ozz_sse_select_f!(cond, _safe, cfalse);
        }
    }

    // Computes the per element linear interpolation of _a and _b, where _alpha is
    // not bound to range [0,1].
    #[inline]
    pub fn lerp(_a: SimdFloat4, _b: SimdFloat4, _alpha: SimdFloat4) -> SimdFloat4 {
        todo!()
    }

    // Computes the per element cosine of _v.
    #[inline]
    pub fn cos(_v: SimdFloat4) -> SimdFloat4 {
        todo!()
    }

    // Computes the cosine of the x component of _v and stores it in the x
    // component of the returned vector. y, z and w of the returned vector are the
    // same as their respective components in _v.
    #[inline]
    pub fn cos_x(_v: SimdFloat4) -> SimdFloat4 {
        todo!()
    }

    // Computes the per element arccosine of _v.
    #[inline]
    pub fn acos(_v: SimdFloat4) -> SimdFloat4 {
        todo!()
    }

    // Computes the arccosine of the x component of _v and stores it in the x
    // component of the returned vector. y, z and w of the returned vector are the
    // same as their respective components in _v.
    #[inline]
    pub fn acos_x(_v: SimdFloat4) -> SimdFloat4 {
        todo!()
    }

    // Computes the per element sines of _v.
    #[inline]
    pub fn sin(_v: SimdFloat4) -> SimdFloat4 {
        todo!()
    }

    // Computes the sines of the x component of _v and stores it in the x
    // component of the returned vector. y, z and w of the returned vector are the
    // same as their respective components in _v.
    #[inline]
    pub fn sin_x(_v: SimdFloat4) -> SimdFloat4 {
        todo!()
    }

    // Computes the per element arcsine of _v.
    #[inline]
    pub fn asin(_v: SimdFloat4) -> SimdFloat4 {
        todo!()
    }

    // Computes the arcsine of the x component of _v and stores it in the x
    // component of the returned vector. y, z and w of the returned vector are the
    // same as their respective components in _v.
    #[inline]
    pub fn asin_x(_v: SimdFloat4) -> SimdFloat4 {
        todo!()
    }

    // Computes the per element tangent of _v.
    #[inline]
    pub fn tan(_v: SimdFloat4) -> SimdFloat4 {
        todo!()
    }

    // Computes the tangent of the x component of _v and stores it in the x
    // component of the returned vector. y, z and w of the returned vector are the
    // same as their respective components in _v.
    #[inline]
    pub fn tan_x(_v: SimdFloat4) -> SimdFloat4 {
        todo!()
    }

    // Computes the per element arctangent of _v.
    #[inline]
    pub fn atan(_v: SimdFloat4) -> SimdFloat4 {
        todo!()
    }

    // Computes the arctangent of the x component of _v and stores it in the x
    // component of the returned vector. y, z and w of the returned vector are the
    // same as their respective components in _v.
    #[inline]
    pub fn atan_x(_v: SimdFloat4) -> SimdFloat4 {
        todo!()
    }

    // Returns boolean selection of vectors _true and _false according to condition
    // _b. All bits a each component of _b must have the same value (O or
    // 0xffffffff) to ensure portability.
    #[inline]
    pub fn select(_b: SimdInt4, _true: SimdFloat4, _false: SimdFloat4) -> SimdFloat4 {
        todo!()
    }

    // Per element "equal" comparison of _a and _b.
    #[inline]
    pub fn cmp_eq(_a: SimdFloat4, _b: SimdFloat4) -> SimdInt4 {
        todo!()
    }

    // Per element "not equal" comparison of _a and _b.
    #[inline]
    pub fn cmp_ne(_a: SimdFloat4, _b: SimdFloat4) -> SimdInt4 {
        todo!()
    }

    // Per element "less than" comparison of _a and _b.
    #[inline]
    pub fn cmp_lt(_a: SimdFloat4, _b: SimdFloat4) -> SimdInt4 {
        todo!()
    }

    // Per element "less than or equal" comparison of _a and _b.
    #[inline]
    pub fn cmp_le(_a: SimdFloat4, _b: SimdFloat4) -> SimdInt4 {
        todo!()
    }

    // Per element "greater than" comparison of _a and _b.
    #[inline]
    pub fn cmp_gt(_a: SimdFloat4, _b: SimdFloat4) -> SimdInt4 {
        todo!()
    }

    // Per element "greater than or equal" comparison of _a and _b.
    #[inline]
    pub fn cmp_ge(_a: SimdFloat4, _b: SimdFloat4) -> SimdInt4 {
        todo!()
    }

    // Returns per element binary and operation of _a and _b.
    // _v[0...127] = _a[0...127] & _b[0...127]
    #[inline]
    pub fn and_ff(_a: SimdFloat4, _b: SimdFloat4) -> SimdFloat4 {
        todo!()
    }

    // Returns per element binary or operation of _a and _b.
    // _v[0...127] = _a[0...127] | _b[0...127]
    #[inline]
    pub fn or_ff(_a: SimdFloat4, _b: SimdFloat4) -> SimdFloat4 {
        todo!()
    }

    // Returns per element binary logical xor operation of _a and _b.
    // _v[0...127] = _a[0...127] ^ _b[0...127]
    #[inline]
    pub fn xor_ff(_a: SimdFloat4, _b: SimdFloat4) -> SimdFloat4 {
        todo!()
    }

    // Returns per element binary and operation of _a and _b.
    // _v[0...127] = _a[0...127] & _b[0...127]
    #[inline]
    pub fn and_fi(_a: SimdFloat4, _b: SimdInt4) -> SimdFloat4 {
        todo!()
    }

    // Returns per element binary and operation of _a and ~_b.
    // _v[0...127] = _a[0...127] & ~_b[0...127]
    #[inline]
    pub fn and_not(_a: SimdFloat4, _b: SimdInt4) -> SimdFloat4 {
        todo!()
    }

    // Returns per element binary or operation of _a and _b.
    // _v[0...127] = _a[0...127] | _b[0...127]
    #[inline]
    pub fn or_fi(_a: SimdFloat4, _b: SimdInt4) -> SimdFloat4 {
        todo!()
    }

    // Returns per element binary logical xor operation of _a and _b.
    // _v[0...127] = _a[0...127] ^ _b[0...127]
    #[inline]
    pub fn xor_fi(_a: SimdFloat4, _b: SimdInt4) -> SimdFloat4 {
        todo!()
    }
}


//--------------------------------------------------------------------------------------------------
pub mod simd_int4 {
    use std::arch::x86_64::*;
    use crate::simd_math::{SimdInt4, SimdFloat4};

    // Returns a SimdInt4 vector with all components set to 0.
    #[inline]
    pub fn zero() -> SimdInt4 {
        todo!()
    }

    // Returns a SimdInt4 vector with all components set to 1.
    #[inline]
    pub fn one() -> SimdInt4 {
        todo!()
    }

    // Returns a SimdInt4 vector with the x component set to 1 and all the others
    // to 0.
    #[inline]
    pub fn x_axis() -> SimdInt4 {
        todo!()
    }

    // Returns a SimdInt4 vector with the y component set to 1 and all the others
    // to 0.
    #[inline]
    pub fn y_axis() -> SimdInt4 {
        todo!()
    }

    // Returns a SimdInt4 vector with the z component set to 1 and all the others
    // to 0.
    #[inline]
    pub fn z_axis() -> SimdInt4 {
        todo!()
    }

    // Returns a SimdInt4 vector with the w component set to 1 and all the others
    // to 0.
    #[inline]
    pub fn w_axis() -> SimdInt4 {
        todo!()
    }

    // Returns a SimdInt4 vector with all components set to true (0xffffffff).
    #[inline]
    pub fn all_true() -> SimdInt4 {
        todo!()
    }

    // Returns a SimdInt4 vector with all components set to false (0).
    #[inline]
    pub fn all_false() -> SimdInt4 {
        todo!()
    }

    // Returns a SimdInt4 vector with sign bits set to 1.
    #[inline]
    pub fn mask_sign() -> SimdInt4 {
        todo!()
    }

    // Returns a SimdInt4 vector with all bits set to 1 except sign.
    #[inline]
    pub fn mask_not_sign() -> SimdInt4 {
        todo!()
    }

    // Returns a SimdInt4 vector with sign bits of x, y and z components set to 1.
    #[inline]
    pub fn mask_sign_xyz() -> SimdInt4 {
        todo!()
    }

    // Returns a SimdInt4 vector with sign bits of w component set to 1.
    #[inline]
    pub fn mask_sign_w() -> SimdInt4 {
        todo!()
    }

    // Returns a SimdInt4 vector with all bits set to 1.
    #[inline]
    pub fn mask_ffff() -> SimdInt4 {
        todo!()
    }

    // Returns a SimdInt4 vector with all bits set to 0.
    #[inline]
    pub fn mask_0000() -> SimdInt4 {
        todo!()
    }

    // Returns a SimdInt4 vector with all the bits of the x, y, z components set to
    // 1, while z is set to 0.
    #[inline]
    pub fn mask_fff0() -> SimdInt4 {
        todo!()
    }

    // Returns a SimdInt4 vector with all the bits of the x component set to 1,
    // while the others are set to 0.
    pub fn mask_f000() -> SimdInt4 {
        todo!()
    }

    // Returns a SimdInt4 vector with all the bits of the y component set to 1,
    // while the others are set to 0.
    #[inline]
    pub fn mask_0f00() -> SimdInt4 {
        todo!()
    }

    // Returns a SimdInt4 vector with all the bits of the z component set to 1,
    // while the others are set to 0.
    #[inline]
    pub fn mask_00f0() -> SimdInt4 {
        todo!()
    }

    // Returns a SimdInt4 vector with all the bits of the w component set to 1,
    // while the others are set to 0.
    #[inline]
    pub fn mask_000f() -> SimdInt4 {
        todo!()
    }

    // Loads _x, _y, _z, _w to the returned vector.
    // r.x = _x
    // r.y = _y
    // r.z = _z
    // r.w = _w
    #[inline]
    pub fn load_i32(_x: i32, _y: i32, _z: i32, _w: i32) -> SimdInt4 {
        todo!()
    }

    // Loads _x, _y, _z, _w to the returned vector using the following conversion
    // rule.
    // r.x = _x ? 0xffffffff:0
    // r.y = _y ? 0xffffffff:0
    // r.z = _z ? 0xffffffff:0
    // r.w = _w ? 0xffffffff:0
    #[inline]
    pub fn load_bool(_x: bool, _y: bool, _z: bool, _w: bool) -> SimdInt4 {
        todo!()
    }

    // Loads _x to the x component of the returned vector using the following
    // conversion rule, and sets y, z and w to 0.
    // r.x = _x ? 0xffffffff:0
    // r.y = 0
    // r.z = 0
    // r.w = 0
    #[inline]
    pub fn load_x(_x: bool) -> SimdInt4 {
        todo!()
    }

    // Loads _x to the all the components of the returned vector using the following
    // conversion rule.
    // r.x = _x ? 0xffffffff:0
    // r.y = _x ? 0xffffffff:0
    // r.z = _x ? 0xffffffff:0
    // r.w = _x ? 0xffffffff:0
    #[inline]
    pub fn load1(_x: bool) -> SimdInt4 {
        todo!()
    }

    // Loads the 4 values of _f to the returned vector.
    // _i must be aligned to 16 bytes.
    // r.x = _i[0]
    // r.y = _i[1]
    // r.z = _i[2]
    // r.w = _i[3]
    #[inline]
    pub fn load_ptr(_i: [i32; 4]) -> SimdInt4 {
        todo!()
    }

    // Loads _i[0] to the x component of the returned vector, and sets y, z and w
    // to 0.
    // _i must be aligned to 16 bytes.
    // r.x = _i[0]
    // r.y = 0
    // r.z = 0
    // r.w = 0
    #[inline]
    pub fn load_xptr(_i: [i32; 4]) -> SimdInt4 {
        todo!()
    }

    // Loads _i[0] to all the components of the returned vector.
    // _i must be aligned to 16 bytes.
    // r.x = _i[0]
    // r.y = _i[0]
    // r.z = _i[0]
    // r.w = _i[0]
    #[inline]
    pub fn load1ptr(_i: [i32; 4]) -> SimdInt4 {
        todo!()
    }

    // Loads the 2 first value of _i to the x and y components of the returned
    // vector. The remaining components are set to 0.
    // _f must be aligned to 4 bytes.
    // r.x = _i[0]
    // r.y = _i[1]
    // r.z = 0
    // r.w = 0
    #[inline]
    pub fn load2ptr(_i: [i32; 4]) -> SimdInt4 {
        todo!()
    }

    // Loads the 3 first value of _i to the x, y and z components of the returned
    // vector. The remaining components are set to 0.
    // _f must be aligned to 16 bytes.
    // r.x = _i[0]
    // r.y = _i[1]
    // r.z = _i[2]
    // r.w = 0
    #[inline]
    pub fn load3ptr(_i: [i32; 4]) -> SimdInt4 {
        todo!()
    }

    // Loads the 4 values of _f to the returned vector.
    // _i must be aligned to 16 bytes.
    // r.x = _i[0]
    // r.y = _i[1]
    // r.z = _i[2]
    // r.w = _i[3]
    #[inline]
    pub fn load_ptr_u(_i: [i32; 4]) -> SimdInt4 {
        todo!()
    }

    // Loads _i[0] to the x component of the returned vector, and sets y, z and w
    // to 0.
    // _f must be aligned to 4 bytes.
    // r.x = _i[0]
    // r.y = 0
    // r.z = 0
    // r.w = 0
    #[inline]
    pub fn load_xptr_u(_i: [i32; 4]) -> SimdInt4 {
        todo!()
    }

    // Loads the 4 values of _i to the returned vector.
    // _i must be aligned to 4 bytes.
    // r.x = _i[0]
    // r.y = _i[0]
    // r.z = _i[0]
    // r.w = _i[0]
    #[inline]
    pub fn load1ptr_u(_i: [i32; 4]) -> SimdInt4 {
        todo!()
    }

    // Loads the 2 first value of _i to the x and y components of the returned
    // vector. The remaining components are set to 0.
    // _f must be aligned to 4 bytes.
    // r.x = _i[0]
    // r.y = _i[1]
    // r.z = 0
    // r.w = 0
    #[inline]
    pub fn load2ptr_u(_i: [i32; 4]) -> SimdInt4 {
        todo!()
    }

    // Loads the 3 first value of _i to the x, y and z components of the returned
    // vector. The remaining components are set to 0.
    // _f must be aligned to 4 bytes.
    // r.x = _i[0]
    // r.y = _i[1]
    // r.z = _i[2]
    // r.w = 0
    #[inline]
    pub fn load3ptr_u(_i: [i32; 4]) -> SimdInt4 {
        todo!()
    }

    // Convert from float to integer by rounding the nearest value.
    #[inline]
    pub fn from_float_round(_f: SimdFloat4) -> SimdInt4 {
        todo!()
    }

    // Convert from float to integer by truncating.
    #[inline]
    pub fn from_float_trunc(_f: SimdFloat4) -> SimdInt4 {
        todo!()
    }

    //----------------------------------------------------------------------------------------------
    // Returns the x component of _v as an integer.
    #[inline]
    pub fn get_x(_v: SimdInt4) -> i32 {
        todo!()
    }

    // Returns the y component of _v as a integer.
    #[inline]
    pub fn get_y(_v: SimdInt4) -> i32 {
        todo!()
    }

    // Returns the z component of _v as a integer.
    #[inline]
    pub fn get_z(_v: SimdInt4) -> i32 {
        todo!()
    }

    // Returns the w component of _v as a integer.
    #[inline]
    pub fn get_w(_v: SimdInt4) -> i32 {
        todo!()
    }

    // Returns _v with the x component set to x component of _i.
    #[inline]
    pub fn set_x(_v: SimdInt4, _i: SimdInt4) -> SimdInt4 {
        todo!()
    }

    // Returns _v with the y component set to x component of _i.
    #[inline]
    pub fn set_y(_v: SimdInt4, _i: SimdInt4) -> SimdInt4 {
        todo!()
    }

    // Returns _v with the z component set to x component of _i.
    #[inline]
    pub fn set_z(_v: SimdInt4, _i: SimdInt4) -> SimdInt4 {
        todo!()
    }

    // Returns _v with the w component set to x component of _i.
    #[inline]
    pub fn set_w(_v: SimdInt4, _i: SimdInt4) -> SimdInt4 {
        todo!()
    }

    // Returns _v with the _ith component set to _i.
    // _i must be in range [0,3]
    #[inline]
    pub fn set_i(_v: SimdInt4, _i: SimdInt4, _ith: usize) -> SimdInt4 {
        todo!()
    }

    // Stores the 4 components of _v to the four first integers of _i.
    // _i must be aligned to 16 bytes.
    // _i[0] = _v.x
    // _i[1] = _v.y
    // _i[2] = _v.z
    // _i[3] = _v.w
    #[inline]
    pub fn store_ptr(_v: SimdInt4, _i: &mut [i32; 4]) {
        todo!()
    }

    // Stores the x component of _v to the first integers of _i.
    // _i must be aligned to 16 bytes.
    // _i[0] = _v.x
    #[inline]
    pub fn store1ptr(_v: SimdInt4, _i: &mut [i32; 4]) {
        todo!()
    }

    // Stores x and y components of _v to the two first integers of _i.
    // _i must be aligned to 16 bytes.
    // _i[0] = _v.x
    // _i[1] = _v.y
    #[inline]
    pub fn store2ptr(_v: SimdInt4, _i: &mut [i32; 4]) {
        todo!()
    }

    // Stores x, y and z components of _v to the three first integers of _i.
    // _i must be aligned to 16 bytes.
    // _i[0] = _v.x
    // _i[1] = _v.y
    // _i[2] = _v.z
    #[inline]
    pub fn store3ptr(_v: SimdInt4, _i: &mut [i32; 4]) {
        todo!()
    }

    // Stores the 4 components of _v to the four first integers of _i.
    // _i must be aligned to 4 bytes.
    // _i[0] = _v.x
    // _i[1] = _v.y
    // _i[2] = _v.z
    // _i[3] = _v.w
    #[inline]
    pub fn store_ptr_u(_v: SimdInt4, _i: &mut [i32; 4]) {
        todo!()
    }

    // Stores the x component of _v to the first float of _i.
    // _i must be aligned to 4 bytes.
    // _i[0] = _v.x
    #[inline]
    pub fn store1ptr_u(_v: SimdInt4, _i: &mut [i32; 4]) {
        todo!()
    }

    // Stores x and y components of _v to the two first integers of _i.
    // _i must be aligned to 4 bytes.
    // _i[0] = _v.x
    // _i[1] = _v.y
    #[inline]
    pub fn store2ptr_u(_v: SimdInt4, _i: &mut [i32; 4]) {
        todo!()
    }

    // Stores x, y and z components of _v to the three first integers of _i.
    // _i must be aligned to 4 bytes.
    // _i[0] = _v.x
    // _i[1] = _v.y
    // _i[2] = _v.z
    #[inline]
    pub fn store3ptr_u(_v: SimdInt4, _i: &mut [i32; 4]) {
        todo!()
    }

    // Replicates x of _a to all the components of the returned vector.
    #[inline]
    pub fn splat_x(_v: SimdInt4) -> SimdInt4 {
        todo!()
    }

    // Replicates y of _a to all the components of the returned vector.
    #[inline]
    pub fn splat_y(_v: SimdInt4) -> SimdInt4 {
        todo!()
    }

    // Replicates z of _a to all the components of the returned vector.
    #[inline]
    pub fn splat_z(_v: SimdInt4) -> SimdInt4 {
        todo!()
    }

    // Replicates w of _a to all the components of the returned vector.
    #[inline]
    pub fn splat_w(_v: SimdInt4) -> SimdInt4 {
        todo!()
    }

    // Swizzle x, y, z and w components based on compile time arguments _X, _Y, _Z
    // and _W. Arguments can vary from 0 (x), to 3 (w).
    #[inline]
    pub fn swizzle0123(_v: SimdInt4) -> SimdInt4 {
        todo!()
    }

    // Creates a 4-bit mask from the most significant bits of each component of _v.
    // i := sign(a3)<<3 | sign(a2)<<2 | sign(a1)<<1 | sign(a0)
    #[inline]
    pub fn move_mask(_v: SimdInt4) -> i32 {
        todo!()
    }

    // Returns true if all the components of _v are not 0.
    #[inline]
    pub fn are_all_true(_v: SimdInt4) -> bool {
        todo!()
    }

    // Returns true if x, y and z components of _v are not 0.
    #[inline]
    pub fn are_all_true3(_v: SimdInt4) -> bool {
        todo!()
    }

    // Returns true if x and y components of _v are not 0.
    #[inline]
    pub fn are_all_true2(_v: SimdInt4) -> bool {
        todo!()
    }

    // Returns true if x component of _v is not 0.
    #[inline]
    pub fn are_all_true1(_v: SimdInt4) -> bool {
        todo!()
    }

    // Returns true if all the components of _v are 0.
    #[inline]
    pub fn are_all_false(_v: SimdInt4) -> bool {
        todo!()
    }

    // Returns true if x, y and z components of _v are 0.
    #[inline]
    pub fn are_all_false3(_v: SimdInt4) -> bool {
        todo!()
    }

    // Returns true if x and y components of _v are 0.
    #[inline]
    pub fn are_all_false2(_v: SimdInt4) -> bool {
        todo!()
    }

    // Returns true if x component of _v is 0.
    #[inline]
    pub fn are_all_false1(_v: SimdInt4) -> bool {
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
    pub fn hadd2(_v: SimdInt4) -> SimdInt4 {
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
    pub fn hadd3(_v: SimdInt4) -> SimdInt4 {
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
    pub fn hadd4(_v: SimdInt4) -> SimdInt4 {
        todo!()
    }

    // Returns the per element absolute value of _v.
    #[inline]
    pub fn abs(_v: SimdInt4) -> SimdInt4 {
        todo!()
    }

    // Returns the sign bit of _v.
    #[inline]
    pub fn sign(_v: SimdInt4) -> SimdInt4 {
        todo!()
    }

    // Returns the per component minimum of _a and _b.
    #[inline]
    pub fn min(_a: SimdInt4, _b: SimdInt4) -> SimdInt4 {
        todo!()
    }

    // Returns the per component maximum of _a and _b.
    #[inline]
    pub fn max(_a: SimdInt4, _b: SimdInt4) -> SimdInt4 {
        todo!()
    }

    // Returns the per component minimum of _v and 0.
    #[inline]
    pub fn min0(_v: SimdInt4) -> SimdInt4 {
        todo!()
    }

    // Returns the per component maximum of _v and 0.
    #[inline]
    pub fn max0(_v: SimdInt4) -> SimdInt4 {
        todo!()
    }

    // Clamps each element of _x between _a and _b.
    // Result is unknown if _a is not less or equal to _b.
    #[inline]
    pub fn clamp(_a: SimdInt4, _v: SimdInt4, _b: SimdInt4) -> SimdInt4 {
        todo!()
    }

    // Returns boolean selection of vectors _true and _false according to consition
    // _b. All bits a each component of _b must have the same value (O or
    // 0xffffffff) to ensure portability.
    #[inline]
    pub fn select(_b: SimdInt4, _true: SimdInt4, _false: SimdInt4) -> SimdInt4 {
        todo!()
    }

    // Returns per element binary and operation of _a and _b.
    // _v[0...127] = _a[0...127] & _b[0...127]
    #[inline]
    pub fn and(_a: SimdInt4, _b: SimdInt4) -> SimdInt4 {
        todo!()
    }

    // Returns per element binary and operation of _a and ~_b.
    // _v[0...127] = _a[0...127] & ~_b[0...127]
    #[inline]
    pub fn and_not(_a: SimdInt4, _b: SimdInt4) -> SimdInt4 {
        todo!()
    }

    // Returns per element binary or operation of _a and _b.
    // _v[0...127] = _a[0...127] | _b[0...127]
    #[inline]
    pub fn or(_a: SimdInt4, _b: SimdInt4) -> SimdInt4 {
        todo!()
    }

    // Returns per element binary logical xor operation of _a and _b.
    // _v[0...127] = _a[0...127] ^ _b[0...127]
    #[inline]
    pub fn xor(_a: SimdInt4, _b: SimdInt4) -> SimdInt4 {
        todo!()
    }

    // Returns per element binary complement of _v.
    // _v[0...127] = ~_b[0...127]
    #[inline]
    pub fn not(_v: SimdInt4) -> SimdInt4 {
        todo!()
    }

    // Shifts the 4 signed or unsigned 32-bit integers in a left by count _bits
    // while shifting in zeros.
    #[inline]
    pub fn shift_l(_v: SimdInt4, _bits: i32) -> SimdInt4 {
        todo!()
    }

    // Shifts the 4 signed 32-bit integers in a right by count bits while shifting
    // in the sign bit.
    #[inline]
    pub fn shift_r(_v: SimdInt4, _bits: i32) -> SimdInt4 {
        todo!()
    }

    // Shifts the 4 signed or unsigned 32-bit integers in a right by count bits
    // while shifting in zeros.
    #[inline]
    pub fn shift_ru(_v: SimdInt4, _bits: i32) -> SimdInt4 {
        todo!()
    }

    // Per element "equal" comparison of _a and _b.
    #[inline]
    pub fn cmp_eq(_a: SimdInt4, _b: SimdInt4) -> SimdInt4 {
        todo!()
    }

    // Per element "not equal" comparison of _a and _b.
    #[inline]
    pub fn cmp_ne(_a: SimdInt4, _b: SimdInt4) -> SimdInt4 {
        todo!()
    }

    // Per element "less than" comparison of _a and _b.
    #[inline]
    pub fn cmp_lt(_a: SimdInt4, _b: SimdInt4) -> SimdInt4 {
        todo!()
    }

    // Per element "less than or equal" comparison of _a and _b.
    #[inline]
    pub fn cmp_le(_a: SimdInt4, _b: SimdInt4) -> SimdInt4 {
        todo!()
    }

    // Per element "greater than" comparison of _a and _b.
    #[inline]
    pub fn cmp_gt(_a: SimdInt4, _b: SimdInt4) -> SimdInt4 {
        todo!()
    }

    // Per element "greater than or equal" comparison of _a and _b.
    #[inline]
    pub fn cmp_ge(_a: SimdInt4, _b: SimdInt4) -> SimdInt4 {
        todo!()
    }
}

//--------------------------------------------------------------------------------------------------
// Declare the 4x4 matrix type. Uses the column major convention where the
// matrix-times-vector is written v'=Mv:
// [ m.cols[0].x m.cols[1].x m.cols[2].x m.cols[3].x ]   {v.x}
// | m.cols[0].y m.cols[1].y m.cols[2].y m.cols[3].y | * {v.y}
// | m.cols[0].z m.cols[1].y m.cols[2].y m.cols[3].y |   {v.z}
// [ m.cols[0].w m.cols[1].w m.cols[2].w m.cols[3].w ]   {v.1}
pub struct Float4x4 {
    // Matrix columns.
    pub cols: [SimdFloat4; 4],
}

impl Float4x4 {
    // Returns the identity matrix.
    #[inline]
    pub fn identity() -> Float4x4 {
        todo!()
    }

    // Returns a translation matrix.
    // _v.w is ignored.
    #[inline]
    pub fn translation(_v: SimdFloat4) -> Float4x4 {
        todo!()
    }

    // Returns a scaling matrix that scales along _v.
    // _v.w is ignored.
    #[inline]
    pub fn scaling(_v: SimdFloat4) -> Float4x4 {
        todo!()
    }

    // Returns the rotation matrix built from Euler angles defined by x, y and z
    // components of _v. Euler angles are ordered Heading, Elevation and Bank, or
    // Yaw, Pitch and Roll. _v.w is ignored.
    #[inline]
    pub fn from_euler(_v: SimdFloat4) -> Float4x4 {
        todo!()
    }

    // Returns the rotation matrix built from axis defined by _axis.xyz and
    // _angle.x
    #[inline]
    pub fn from_axis_angle(_axis: SimdFloat4, _angle: SimdFloat4) -> Float4x4 {
        todo!()
    }

    // Returns the rotation matrix built from quaternion defined by x, y, z and w
    // components of _v.
    #[inline]
    pub fn from_quaternion(_v: SimdFloat4) -> Float4x4 {
        todo!()
    }

    // Returns the affine transformation matrix built from split translation,
    // rotation (quaternion) and scale.
    #[inline]
    pub fn from_affine(_translation: SimdFloat4,
                       _quaternion: SimdFloat4,
                       _scale: SimdFloat4) -> Float4x4 {
        todo!()
    }
}

// Returns the transpose of matrix _m.
#[inline]
pub fn transpose(_m: &Float4x4) -> Float4x4 {
    todo!()
}

// Returns the inverse of matrix _m.
// If _invertible is not nullptr, its x component will be set to true if matrix is
// invertible. If _invertible is nullptr, then an assert is triggered in case the
// matrix isn't invertible.
#[inline]
pub fn invert(_m: &Float4x4, _invertible: Option<SimdInt4>) -> Float4x4 {
    todo!()
}

// Translates matrix _m along the axis defined by _v components.
// _v.w is ignored.
#[inline]
pub fn translate(_m: &Float4x4, _v: SimdFloat4) -> Float4x4 {
    todo!()
}

// Scales matrix _m along each axis with x, y, z components of _v.
// _v.w is ignored.
#[inline]
pub fn scale(_m: &Float4x4, _v: SimdFloat4) -> Float4x4 {
    todo!()
}

// Multiply each column of matrix _m with vector _v.
#[inline]
pub fn column_multiply(_m: &Float4x4, _v: SimdFloat4) -> Float4x4 {
    todo!()
}

// Tests if each 3 column of upper 3x3 matrix of _m is a normal matrix.
// Returns the result in the x, y and z component of the returned vector. w is
// set to 0.
#[inline]
pub fn is_normalized(_m: &Float4x4) -> SimdInt4 {
    todo!()
}

// Tests if each 3 column of upper 3x3 matrix of _m is a normal matrix.
// Uses the estimated tolerance
// Returns the result in the x, y and z component of the returned vector. w is
// set to 0.
#[inline]
pub fn is_normalized_est(_m: &Float4x4) -> SimdInt4 {
    todo!()
}

// Tests if the upper 3x3 matrix of _m is an orthogonal matrix.
// A matrix that contains a reflexion cannot be considered orthogonal.
// Returns the result in the x component of the returned vector. y, z and w are
// set to 0.
#[inline]
pub fn is_orthogonal(_m: &Float4x4) -> SimdInt4 {
    todo!()
}

// Returns the quaternion that represent the rotation of matrix _m.
// _m must be normalized and orthogonal.
// the return quaternion is normalized.
#[inline]
pub fn to_quaternion(_m: &Float4x4) -> SimdFloat4 {
    todo!()
}

// Decompose a general 3D transformation matrix _m into its scalar, rotational
// and translational components.
// Returns false if it was not possible to decompose the matrix. This would be
// because more than 1 of the 3 first column of _m are scaled to 0.
#[inline]
pub fn to_affine(_m: &Float4x4, _translation: &mut SimdFloat4, _quaternion: &mut SimdFloat4, _scale: &mut SimdFloat4) -> bool {
    todo!()
}

// Computes the transformation of a Float4x4 matrix and a point _p.
// This is equivalent to multiplying a matrix by a SimdFloat4 with a w component
// of 1.
#[inline]
pub fn transform_point(_m: &Float4x4, _v: SimdFloat4) -> SimdFloat4 {
    todo!()
}

// Computes the transformation of a Float4x4 matrix and a vector _v.
// This is equivalent to multiplying a matrix by a SimdFloat4 with a w component
// of 0.
#[inline]
pub fn transform_vector(_m: &Float4x4, _v: SimdFloat4) -> SimdFloat4 {
    todo!()
}

// Computes the multiplication of matrix Float4x4 and vector _v.
impl Mul<SimdFloat4> for Float4x4 {
    type Output = SimdFloat4;
    #[inline]
    fn mul(self, rhs: SimdFloat4) -> Self::Output {
        todo!()
    }
}

// Computes the multiplication of two matrices _a and _b.
impl Mul for Float4x4 {
    type Output = Float4x4;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        todo!()
    }
}

// Computes the per element addition of two matrices _a and _b.
impl Add for Float4x4 {
    type Output = Float4x4;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        todo!()
    }
}

// Computes the per element subtraction of two matrices _a and _b.
impl Sub for Float4x4 {
    type Output = Float4x4;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        todo!()
    }
}

// Converts from a float to a half.
#[inline]
pub fn float_to_half(_f: f32) -> u16 {
    todo!()
}

// Converts from a half to a float.
#[inline]
pub fn half_to_float(_h: u16) -> f32 {
    todo!()
}

// Converts from a float to a half.
#[inline]
pub fn float_to_half_simd(_f: SimdFloat4) -> SimdInt4 {
    todo!()
}

// Converts from a half to a float.
#[inline]
pub fn half_to_float_simd(_h: SimdInt4) -> SimdFloat4 {
    todo!()
}