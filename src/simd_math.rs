/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use std::arch::x86_64::*;
use std::ops::{Mul, Add, Sub};

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
            ozz_shuffle_ps1!($_v, _mm_shuffle!($_i,$_i,$_i,$_i))
        };
    }

macro_rules! ozz_sse_splat_i {
        ($_v:expr, $_i:expr) => {
            _mm_shuffle_epi32($_v, _mm_shuffle!($_i, $_i, $_i, $_i))
        };
    }

macro_rules! ozz_sse_hadd2_f {
        ($_v:expr) => {
            _mm_add_ss($_v, ozz_sse_splat_f!($_v, 1))
        };
    }

macro_rules! ozz_sse_hadd3_f {
        ($_v:expr) => {
            _mm_add_ss(_mm_add_ss($_v, ozz_sse_splat_f!($_v, 2)), ozz_sse_splat_f!($_v, 1))
        };
    }

macro_rules! ozz_sse_hadd4_f {
        ($_v:expr, $_r:expr) => {
            {
                let haddxyzw = _mm_add_ps($_v, _mm_movehl_ps($_v, $_v));
                $_r = _mm_add_ss(haddxyzw, ozz_sse_splat_f!(haddxyzw, 1));
            }
        };
    }

macro_rules! ozz_sse_dot2_f {
        ($_a:expr, $_b:expr, $_r:expr) => {
            {
                let ab = _mm_mul_ps($_a, $_b);
                $_r = _mm_add_ss(ab, ozz_sse_splat_f!(ab, 1));
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

macro_rules! ozz_nmaddx {
        ($_a:expr, $_b:expr, $_c:expr) => {
            _mm_sub_ss($_c, _mm_mul_ss($_a, $_b))
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
// Vector of four floating point values.
pub struct SimdFloat4 {
    pub data: __m128,
}

pub union SimdFloat4Union {
    ret: __m128,
    af: [f32; 4],
}

impl SimdFloat4 {
    #[inline]
    pub fn new(data: __m128) -> SimdFloat4 {
        return SimdFloat4 {
            data
        };
    }

    // Returns a SimdFloat4 vector with all components set to 0.
    #[inline]
    pub fn zero() -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(_mm_setzero_ps());
        }
    }

    // Returns a SimdFloat4 vector with all components set to 1.
    #[inline]
    pub fn one() -> SimdFloat4 {
        unsafe {
            let zero = _mm_setzero_si128();
            return SimdFloat4::new(_mm_castsi128_ps(
                _mm_srli_epi32(_mm_slli_epi32(_mm_cmpeq_epi32(zero, zero), 25), 2)));
        }
    }

    // Returns a SimdFloat4 vector with the x component set to 1 and all the others to 0.
    #[inline]
    pub fn x_axis() -> SimdFloat4 {
        unsafe {
            let zero = _mm_setzero_si128();
            let one = _mm_srli_epi32(_mm_slli_epi32(_mm_cmpeq_epi32(zero, zero), 25), 2);
            return SimdFloat4::new(_mm_castsi128_ps(_mm_srli_si128(one, 12)));
        }
    }

    // Returns a SimdFloat4 vector with the y component set to 1 and all the others to 0.
    #[inline]
    pub fn y_axis() -> SimdFloat4 {
        unsafe {
            let zero = _mm_setzero_si128();
            let one = _mm_srli_epi32(_mm_slli_epi32(_mm_cmpeq_epi32(zero, zero), 25), 2);
            return SimdFloat4::new(_mm_castsi128_ps(_mm_slli_si128(_mm_srli_si128(one, 12), 4)));
        }
    }

    // Returns a SimdFloat4 vector with the z component set to 1 and all the others to 0.
    #[inline]
    pub fn z_axis() -> SimdFloat4 {
        unsafe {
            let zero = _mm_setzero_si128();
            let one = _mm_srli_epi32(_mm_slli_epi32(_mm_cmpeq_epi32(zero, zero), 25), 2);
            return SimdFloat4::new(_mm_castsi128_ps(_mm_slli_si128(_mm_srli_si128(one, 12), 8)));
        }
    }

    // Returns a SimdFloat4 vector with the w component set to 1 and all the others to 0.
    #[inline]
    pub fn w_axis() -> SimdFloat4 {
        unsafe {
            let zero = _mm_setzero_si128();
            let one = _mm_srli_epi32(_mm_slli_epi32(_mm_cmpeq_epi32(zero, zero), 25), 2);
            return SimdFloat4::new(_mm_castsi128_ps(_mm_slli_si128(one, 12)));
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
            return SimdFloat4::new(_mm_set_ps(_w, _z, _y, _x));
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
            return SimdFloat4::new(_mm_set_ss(_x));
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
            return SimdFloat4::new(_mm_set_ps1(_x));
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
            return SimdFloat4::new(_mm_load_ps(_f.as_ptr()));
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
            return SimdFloat4::new(_mm_loadu_ps(_f.as_ptr()));
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
            return SimdFloat4::new(_mm_load_ss(_f.as_ptr()));
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
            return SimdFloat4::new(_mm_load_ps1(_f.as_ptr()));
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
            return SimdFloat4::new(_mm_unpacklo_ps(_mm_load_ss(_f.as_ptr().add(0)), _mm_load_ss(_f.as_ptr().add(1))));
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
            return SimdFloat4::new(_mm_movelh_ps(
                _mm_unpacklo_ps(_mm_load_ss(_f.as_ptr().add(0)),
                                _mm_load_ss(_f.as_ptr().add(1))),
                _mm_load_ss(_f.as_ptr().add(2))));
        }
    }

    // Convert from integer to float.
    #[inline]
    pub fn from_int(_i: SimdInt4) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(_mm_cvtepi32_ps(_i.data));
        }
    }

    //----------------------------------------------------------------------------------------------
    // Returns the x component of _v as a float.
    #[inline]
    pub fn get_x(&self) -> f32 {
        unsafe {
            return _mm_cvtss_f32(self.data);
        }
    }

    // Returns the y component of _v as a float.
    #[inline]
    pub fn get_y(&self) -> f32 {
        unsafe {
            return _mm_cvtss_f32(ozz_sse_splat_f!(self.data, 1));
        }
    }

    // Returns the z component of _v as a float.
    #[inline]
    pub fn get_z(&self) -> f32 {
        unsafe {
            return _mm_cvtss_f32(_mm_movehl_ps(self.data, self.data));
        }
    }

    // Returns the w component of _v as a float.
    #[inline]
    pub fn get_w(&self) -> f32 {
        unsafe {
            return _mm_cvtss_f32(ozz_sse_splat_f!(self.data, 3));
        }
    }

    // Returns _v with the x component set to x component of _f.
    #[inline]
    pub fn set_x(&mut self, _f: SimdFloat4) {
        unsafe {
            self.data = _mm_move_ss(self.data, _f.data);
        }
    }

    // Returns _v with the y component set to  x component of _f.
    #[inline]
    pub fn set_y(&mut self, _f: SimdFloat4) {
        unsafe {
            let xfnn = _mm_unpacklo_ps(self.data, _f.data);
            self.data = _mm_shuffle_ps(xfnn, self.data, _mm_shuffle!(3, 2, 1, 0));
        }
    }


    // Returns _v with the z component set to  x component of _f.
    #[inline]
    pub fn set_z(&mut self, _f: SimdFloat4) {
        unsafe {
            let ffww = _mm_shuffle_ps(_f.data, self.data, _mm_shuffle!(3, 3, 0, 0));
            self.data = _mm_shuffle_ps(self.data, ffww, _mm_shuffle!(2, 0, 1, 0));
        }
    }


    // Returns _v with the w component set to  x component of _f.
    #[inline]
    pub fn set_w(&mut self, _f: SimdFloat4) {
        unsafe {
            let ffzz = _mm_shuffle_ps(_f.data, self.data, _mm_shuffle!(2, 2, 0, 0));
            self.data = _mm_shuffle_ps(self.data, ffzz, _mm_shuffle!(0, 2, 1, 0));
        }
    }

    // Returns _v with the _i th component set to _f.
    // _i must be in range [0,3]
    #[inline]
    pub fn set_i(&mut self, _f: SimdFloat4, _ith: usize) {
        unsafe {
            let mut u = SimdFloat4Union {
                ret: self.data,
            };

            u.af[_ith] = _mm_cvtss_f32(_f.data);
            self.data = u.ret;
        }
    }

    // Stores the 4 components of _v to the four first floats of _f.
    // _f must be aligned to 16 bytes.
    // _f[0] = _v.x
    // _f[1] = _v.y
    // _f[2] = _v.z
    // _f[3] = _v.w
    #[inline]
    pub fn store_ptr(&self, _f: &mut [f32; 4]) {
        unsafe {
            _mm_store_ps(_f.as_mut_ptr(), self.data);
        }
    }

    // Stores the x component of _v to the first float of _f.
    // _f must be aligned to 16 bytes.
    // _f[0] = _v.x
    #[inline]
    pub fn store1ptr(&self, _f: &mut [f32; 4]) {
        unsafe {
            _mm_store_ss(_f.as_mut_ptr(), self.data);
        }
    }

    // Stores x and y components of _v to the two first floats of _f.
    // _f must be aligned to 16 bytes.
    // _f[0] = _v.x
    // _f[1] = _v.y
    #[inline]
    pub fn store2ptr(&self, _f: &mut [f32; 4]) {
        todo!()
    }

    // Stores x, y and z components of _v to the three first floats of _f.
    // _f must be aligned to 16 bytes.
    // _f[0] = _v.x
    // _f[1] = _v.y
    // _f[2] = _v.z
    #[inline]
    pub fn store3ptr(&self, _f: &mut [f32; 4]) {
        todo!()
    }

    // Stores the 4 components of _v to the four first floats of _f.
    // _f must be aligned to 4 bytes.
    // _f[0] = _v.x
    // _f[1] = _v.y
    // _f[2] = _v.z
    // _f[3] = _v.w
    #[inline]
    pub fn store_ptr_u(&self, _f: &mut [f32; 4]) {
        unsafe {
            _mm_storeu_ps(_f.as_mut_ptr(), self.data);
        }
    }

    // Stores the x component of _v to the first float of _f.
    // _f must be aligned to 4 bytes.
    // _f[0] = _v.x
    #[inline]
    pub fn store1ptr_u(&self, _f: &mut [f32; 4]) {
        unsafe {
            _mm_store_ss(_f.as_mut_ptr(), self.data);
        }
    }

    // Stores x and y components of _v to the two first floats of _f.
    // _f must be aligned to 4 bytes.
    // _f[0] = _v.x
    // _f[1] = _v.y
    #[inline]
    pub fn store2ptr_u(&self, _f: &mut [f32; 4]) {
        unsafe {
            _mm_store_ss(_f.as_mut_ptr().add(0), self.data);
            _mm_store_ss(_f.as_mut_ptr().add(1), ozz_sse_splat_f!(self.data, 1));
        }
    }

    // Stores x, y and z components of _v to the three first floats of _f.
    // _f must be aligned to 4 bytes.
    // _f[0] = _v.x
    // _f[1] = _v.y
    // _f[2] = _v.z
    #[inline]
    pub fn store3ptr_u(&self, _f: &mut [f32; 4]) {
        unsafe {
            _mm_store_ss(_f.as_mut_ptr().add(0), self.data);
            _mm_store_ss(_f.as_mut_ptr().add(1), ozz_sse_splat_f!(self.data, 1));
            _mm_store_ss(_f.as_mut_ptr().add(2), _mm_movehl_ps(self.data, self.data));
        }
    }

    // Replicates x of _a to all the components of the returned vector.
    #[inline]
    pub fn splat_x(&self) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(ozz_sse_splat_f!(self.data, 0));
        }
    }

    // Replicates y of _a to all the components of the returned vector.
    #[inline]
    pub fn splat_y(&self) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(ozz_sse_splat_f!(self.data, 1));
        }
    }

    // Replicates z of _a to all the components of the returned vector.
    #[inline]
    pub fn splat_z(&self) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(ozz_sse_splat_f!(self.data, 2));
        }
    }

    // Replicates w of _a to all the components of the returned vector.
    #[inline]
    pub fn splat_w(&self) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(ozz_sse_splat_f!(self.data, 3));
        }
    }

    // swizzle X, y, z and w components based on compile time arguments _X, _Y, _Z
    // and _W. Arguments can vary from 0 (X), to 3 (w).
    #[inline]
    pub fn swizzle0123(&self) -> SimdFloat4 {
        return SimdFloat4::new(self.data);
    }

    #[inline]
    pub fn swizzle3332(&self) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(ozz_shuffle_ps1!(self.data, _mm_shuffle!(3, 3, 3, 2)));
        }
    }

    #[inline]
    pub fn swizzle3330(&self) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(ozz_shuffle_ps1!(self.data, _mm_shuffle!(3, 3, 3, 0)));
        }
    }

    #[inline]
    pub fn swizzle0122(&self) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(ozz_shuffle_ps1!(self.data, _mm_shuffle!(0, 1, 2, 2)));
        }
    }

    #[inline]
    pub fn swizzle0120(&self) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(ozz_shuffle_ps1!(self.data, _mm_shuffle!(0, 1, 2, 0)));
        }
    }

    #[inline]
    pub fn swizzle1201(&self) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(ozz_shuffle_ps1!(self.data, _mm_shuffle!(1, 2, 0, 1)));
        }
    }

    #[inline]
    pub fn swizzle2011(&self) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(ozz_shuffle_ps1!(self.data, _mm_shuffle!(2, 0, 1, 1)));
        }
    }

    #[inline]
    pub fn swizzle2013(&self) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(ozz_shuffle_ps1!(self.data, _mm_shuffle!(2, 0, 1, 3)));
        }
    }

    #[inline]
    pub fn swizzle1203(&self) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(ozz_shuffle_ps1!(self.data, _mm_shuffle!(1, 2, 0, 3)));
        }
    }

    #[inline]
    pub fn swizzle0101(&self) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(_mm_movelh_ps(self.data, self.data));
        }
    }

    #[inline]
    pub fn swizzle2323(&self) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(_mm_movehl_ps(self.data, self.data));
        }
    }

    #[inline]
    pub fn swizzle0011(&self) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(_mm_unpacklo_ps(self.data, self.data));
        }
    }

    #[inline]
    pub fn swizzle2233(&self) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(_mm_unpackhi_ps(self.data, self.data));
        }
    }

    // Transposes the x components of the 4 SimdFloat4 of _in into the 1
    // SimdFloat4 of _out.
    #[inline]
    pub fn transpose4x1(_in: [SimdFloat4; 4], _out: &mut [SimdFloat4; 1]) {
        unsafe {
            let xz = _mm_unpacklo_ps(_in[0].data, _in[2].data);
            let yw = _mm_unpacklo_ps(_in[1].data, _in[3].data);
            _out[0].data = _mm_unpacklo_ps(xz, yw);
        }
    }

    // Transposes x, y, z and w components of _in to the x components of _out.
    // Remaining y, z and w are set to 0.
    #[inline]
    pub fn transpose1x4(_in: [SimdFloat4; 1], _out: &mut [SimdFloat4; 4]) {
        unsafe {
            let zwzw = _mm_movehl_ps(_in[0].data, _in[0].data);
            let yyyy = ozz_sse_splat_f!(_in[0].data, 1);
            let wwww = ozz_sse_splat_f!(_in[0].data, 3);
            let zero = _mm_setzero_ps();
            _out[0].data = _mm_move_ss(zero, _in[0].data);
            _out[1].data = _mm_move_ss(zero, yyyy);
            _out[2].data = _mm_move_ss(zero, zwzw);
            _out[3].data = _mm_move_ss(zero, wwww);
        }
    }

    // Transposes the 1 SimdFloat4 of _in into the x components of the 4
    // SimdFloat4 of _out. Remaining y, z and w are set to 0.
    #[inline]
    pub fn transpose2x4(_in: [SimdFloat4; 2], _out: &mut [SimdFloat4; 4]) {
        unsafe {
            let tmp0 = _mm_unpacklo_ps(_in[0].data, _in[1].data);
            let tmp1 = _mm_unpackhi_ps(_in[0].data, _in[1].data);
            let zero = _mm_setzero_ps();
            _out[0].data = _mm_movelh_ps(tmp0, zero);
            _out[1].data = _mm_movehl_ps(zero, tmp0);
            _out[2].data = _mm_movelh_ps(tmp1, zero);
            _out[3].data = _mm_movehl_ps(zero, tmp1);
        }
    }

    // Transposes the x and y components of the 4 SimdFloat4 of _in into the 2
    // SimdFloat4 of _out.
    #[inline]
    pub fn transpose4x2(_in: [SimdFloat4; 4], _out: &mut [SimdFloat4; 2]) {
        unsafe {
            let tmp0 = _mm_unpacklo_ps(_in[0].data, _in[2].data);
            let tmp1 = _mm_unpacklo_ps(_in[1].data, _in[3].data);
            _out[0].data = _mm_unpacklo_ps(tmp0, tmp1);
            _out[1].data = _mm_unpackhi_ps(tmp0, tmp1);
        }
    }

    // Transposes the x, y and z components of the 4 SimdFloat4 of _in into the 3
    // SimdFloat4 of _out.
    #[inline]
    pub fn transpose4x3(_in: [SimdFloat4; 4], _out: &mut [SimdFloat4; 3]) {
        unsafe {
            let tmp0 = _mm_unpacklo_ps(_in[0].data, _in[2].data);
            let tmp1 = _mm_unpacklo_ps(_in[1].data, _in[3].data);
            let tmp2 = _mm_unpackhi_ps(_in[0].data, _in[2].data);
            let tmp3 = _mm_unpackhi_ps(_in[1].data, _in[3].data);
            _out[0].data = _mm_unpacklo_ps(tmp0, tmp1);
            _out[1].data = _mm_unpackhi_ps(tmp0, tmp1);
            _out[2].data = _mm_unpacklo_ps(tmp2, tmp3);
        }
    }

    // Transposes the 3 SimdFloat4 of _in into the x, y and z components of the 4
    // SimdFloat4 of _out. Remaining w are set to 0.
    #[inline]
    pub fn transpose3x4(_in: [SimdFloat4; 3], _out: &mut [SimdFloat4; 4]) {
        unsafe {
            let zero = _mm_setzero_ps();
            let temp0 = _mm_unpacklo_ps(_in[0].data, _in[1].data);
            let temp1 = _mm_unpacklo_ps(_in[2].data, zero);
            let temp2 = _mm_unpackhi_ps(_in[0].data, _in[1].data);
            let temp3 = _mm_unpackhi_ps(_in[2].data, zero);
            _out[0].data = _mm_movelh_ps(temp0, temp1);
            _out[1].data = _mm_movehl_ps(temp1, temp0);
            _out[2].data = _mm_movelh_ps(temp2, temp3);
            _out[3].data = _mm_movehl_ps(temp3, temp2);
        }
    }

    // Transposes the 4 SimdFloat4 of _in into the 4 SimdFloat4 of _out.
    #[inline]
    pub fn transpose4x4(_in: [SimdFloat4; 4], _out: &mut [SimdFloat4; 4]) {
        unsafe {
            let tmp0 = _mm_unpacklo_ps(_in[0].data, _in[2].data);
            let tmp1 = _mm_unpacklo_ps(_in[1].data, _in[3].data);
            let tmp2 = _mm_unpackhi_ps(_in[0].data, _in[2].data);
            let tmp3 = _mm_unpackhi_ps(_in[1].data, _in[3].data);
            _out[0].data = _mm_unpacklo_ps(tmp0, tmp1);
            _out[1].data = _mm_unpackhi_ps(tmp0, tmp1);
            _out[2].data = _mm_unpacklo_ps(tmp2, tmp3);
            _out[3].data = _mm_unpackhi_ps(tmp2, tmp3);
        }
    }

    // Transposes the 16 SimdFloat4 of _in into the 16 SimdFloat4 of _out.
    #[inline]
    pub fn transpose16x16(_in: [SimdFloat4; 16], _out: &mut [SimdFloat4; 16]) {
        unsafe {
            let tmp0 = _mm_unpacklo_ps(_in[0].data, _in[2].data);
            let tmp1 = _mm_unpacklo_ps(_in[1].data, _in[3].data);
            _out[0].data = _mm_unpacklo_ps(tmp0, tmp1);
            _out[4].data = _mm_unpackhi_ps(tmp0, tmp1);
            let tmp2 = _mm_unpackhi_ps(_in[0].data, _in[2].data);
            let tmp3 = _mm_unpackhi_ps(_in[1].data, _in[3].data);
            _out[8].data = _mm_unpacklo_ps(tmp2, tmp3);
            _out[12].data = _mm_unpackhi_ps(tmp2, tmp3);
            let tmp4 = _mm_unpacklo_ps(_in[4].data, _in[6].data);
            let tmp5 = _mm_unpacklo_ps(_in[5].data, _in[7].data);
            _out[1].data = _mm_unpacklo_ps(tmp4, tmp5);
            _out[5].data = _mm_unpackhi_ps(tmp4, tmp5);
            let tmp6 = _mm_unpackhi_ps(_in[4].data, _in[6].data);
            let tmp7 = _mm_unpackhi_ps(_in[5].data, _in[7].data);
            _out[9].data = _mm_unpacklo_ps(tmp6, tmp7);
            _out[13].data = _mm_unpackhi_ps(tmp6, tmp7);
            let tmp8 = _mm_unpacklo_ps(_in[8].data, _in[10].data);
            let tmp9 = _mm_unpacklo_ps(_in[9].data, _in[11].data);
            _out[2].data = _mm_unpacklo_ps(tmp8, tmp9);
            _out[6].data = _mm_unpackhi_ps(tmp8, tmp9);
            let tmp10 = _mm_unpackhi_ps(_in[8].data, _in[10].data);
            let tmp11 = _mm_unpackhi_ps(_in[9].data, _in[11].data);
            _out[10].data = _mm_unpacklo_ps(tmp10, tmp11);
            _out[14].data = _mm_unpackhi_ps(tmp10, tmp11);
            let tmp12 = _mm_unpacklo_ps(_in[12].data, _in[14].data);
            let tmp13 = _mm_unpacklo_ps(_in[13].data, _in[15].data);
            _out[3].data = _mm_unpacklo_ps(tmp12, tmp13);
            _out[7].data = _mm_unpackhi_ps(tmp12, tmp13);
            let tmp14 = _mm_unpackhi_ps(_in[12].data, _in[14].data);
            let tmp15 = _mm_unpackhi_ps(_in[13].data, _in[15].data);
            _out[11].data = _mm_unpacklo_ps(tmp14, tmp15);
            _out[15].data = _mm_unpackhi_ps(tmp14, tmp15);
        }
    }

    // Multiplies _a and _b, then adds _c.
    // v = (_a * _b) + _c
    #[inline]
    pub fn madd(&self, _b: SimdFloat4, _c: SimdFloat4) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(ozz_madd!(self.data, _b.data, _c.data));
        }
    }

    // Multiplies _a and _b, then subs _c.
    // v = (_a * _b) + _c
    #[inline]
    pub fn msub(&self, _b: SimdFloat4, _c: SimdFloat4) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(ozz_msub!(self.data, _b.data, _c.data));
        }
    }

    // Multiplies _a and _b, negate it, then adds _c.
    // v = -(_a * _b) + _c
    #[inline]
    pub fn nmadd(&self, _b: SimdFloat4, _c: SimdFloat4) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(ozz_nmadd!(self.data, _b.data, _c.data));
        }
    }

    // Multiplies _a and _b, negate it, then subs _c.
    // v = -(_a * _b) + _c
    #[inline]
    pub fn nmsub(&self, _b: SimdFloat4, _c: SimdFloat4) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(_mm_sub_ps(SimdFloat4::zero().data,
                                              _mm_add_ps(_mm_mul_ps(self.data, _b.data), _c.data)));
        }
    }

    // Divides the x component of _a by the _x component of _b and stores it in the
    // x component of the returned vector. y, z, w of the returned vector are the
    // same as _a respective components.
    // r.x = _a.x / _b.x
    // r.y = _a.y
    // r.z = _a.z
    // r.w = _a.w
    #[inline]
    pub fn div_x(&self, _b: SimdFloat4) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(_mm_div_ss(self.data, _b.data));
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
    pub fn hadd2(&self) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(ozz_sse_hadd2_f!(self.data));
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
    pub fn hadd3(&self) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(ozz_sse_hadd3_f!(self.data));
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
    pub fn hadd4(&self) -> SimdFloat4 {
        unsafe {
            let hadd4: __m128;
            ozz_sse_hadd4_f!(self.data, hadd4);
            return SimdFloat4::new(hadd4);
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
    pub fn dot2(&self, _b: SimdFloat4) -> SimdFloat4 {
        unsafe {
            let dot2: __m128;
            ozz_sse_dot2_f!(self.data, _b.data, dot2);
            return SimdFloat4::new(dot2);
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
    pub fn dot3(&self, _b: SimdFloat4) -> SimdFloat4 {
        unsafe {
            let dot3: __m128;
            ozz_sse_dot3_f!(self.data, _b.data, dot3);
            return SimdFloat4::new(dot3);
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
    pub fn dot4(&self, _b: SimdFloat4) -> SimdFloat4 {
        unsafe {
            let dot4: __m128;
            ozz_sse_dot4_f!(self.data, _b.data, dot4);
            return SimdFloat4::new(dot4);
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
    pub fn cross3(&self, _b: SimdFloat4) -> SimdFloat4 {
        unsafe {
            // Implementation with 3 shuffles only is based on:
            // https://geometrian.com/programming/tutorials/cross-product
            let shufa = ozz_shuffle_ps1!(self.data, _mm_shuffle!(3, 0, 2, 1));
            let shufb = ozz_shuffle_ps1!(_b.data, _mm_shuffle!(3, 0, 2, 1));
            let shufc = ozz_msub!(self.data, shufb, _mm_mul_ps(_b.data, shufa));
            return SimdFloat4::new(ozz_shuffle_ps1!(shufc, _mm_shuffle!(3, 0, 2, 1)));
        }
    }

    // Returns the per component estimated reciprocal of _v.
    #[inline]
    pub fn rcp_est(&self) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(_mm_rcp_ps(self.data));
        }
    }

    // Returns the per component estimated reciprocal of _v, where approximation is
    // improved with one more new Newton-Raphson step.
    #[inline]
    pub fn rcp_est_nr(&self) -> SimdFloat4 {
        unsafe {
            let nr = _mm_rcp_ps(self.data);
            // Do one more Newton-Raphson step to improve precision.
            return SimdFloat4::new(ozz_nmadd!(_mm_mul_ps(nr, nr), self.data, _mm_add_ps(nr, nr)));
        }
    }

    // Returns the estimated reciprocal of the x component of _v and stores it in
    // the x component of the returned vector. y, z, w of the returned vector are
    // the same as their respective components in _v.
    #[inline]
    pub fn rcp_est_x(&self) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(_mm_rcp_ss(self.data));
        }
    }

    // Returns the estimated reciprocal of the x component of _v, where
    // approximation is improved with one more new Newton-Raphson step. y, z, w of
    // the returned vector are undefined.
    #[inline]
    pub fn rcp_est_xnr(&self) -> SimdFloat4 {
        unsafe {
            let nr = _mm_rcp_ss(self.data);
            // Do one more Newton-Raphson step to improve precision.
            return SimdFloat4::new(ozz_nmaddx!(_mm_mul_ss(nr, nr), self.data, _mm_add_ss(nr, nr)));
        }
    }

    // Returns the per component square root of _v.
    #[inline]
    pub fn sqrt(&self) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(_mm_sqrt_ps(self.data));
        }
    }

    // Returns the square root of the x component of _v and stores it in the x
    // component of the returned vector. y, z, w of the returned vector are the
    // same as their respective components in _v.
    #[inline]
    pub fn sqrt_x(&self) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(_mm_sqrt_ss(self.data));
        }
    }

    // Returns the per component estimated reciprocal square root of _v.
    #[inline]
    pub fn rsqrt_est(&self) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(_mm_rsqrt_ps(self.data));
        }
    }

    // Returns the per component estimated reciprocal square root of _v, where
    // approximation is improved with one more new Newton-Raphson step.
    #[inline]
    pub fn rsqrt_est_nr(&self) -> SimdFloat4 {
        unsafe {
            let nr = _mm_rsqrt_ps(self.data);
            // Do one more Newton-Raphson step to improve precision.
            return SimdFloat4::new(_mm_mul_ps(_mm_mul_ps(_mm_set_ps1(0.5), nr),
                                              ozz_nmadd!(_mm_mul_ps(self.data, nr), nr, _mm_set_ps1(3.0))));
        }
    }

    // Returns the estimated reciprocal square root of the x component of _v and
    // stores it in the x component of the returned vector. y, z, w of the returned
    // vector are the same as their respective components in _v.
    #[inline]
    pub fn rsqrt_est_x(&self) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(_mm_rsqrt_ss(self.data));
        }
    }

    // Returns the estimated reciprocal square root of the x component of _v, where
    // approximation is improved with one more new Newton-Raphson step. y, z, w of
    // the returned vector are undefined.
    #[inline]
    pub fn rsqrt_est_xnr(&self) -> SimdFloat4 {
        unsafe {
            let nr = _mm_rsqrt_ss(self.data);
            // Do one more Newton-Raphson step to improve precision.
            return SimdFloat4::new(_mm_mul_ss(_mm_mul_ss(_mm_set_ps1(0.5), nr),
                                              ozz_nmaddx!(_mm_mul_ss(self.data, nr), nr, _mm_set_ps1(3.0))));
        }
    }

    // Returns the per element absolute value of _v.
    #[inline]
    pub fn abs(&self) -> SimdFloat4 {
        unsafe {
            let zero = _mm_setzero_si128();
            return SimdFloat4::new(_mm_and_ps(
                _mm_castsi128_ps(_mm_srli_epi32(_mm_cmpeq_epi32(zero, zero), 1)), self.data));
        }
    }

    // Returns the sign bit of _v.
    #[inline]
    pub fn sign(&self) -> SimdInt4 {
        unsafe {
            return SimdInt4::new(_mm_slli_epi32(_mm_srli_epi32(_mm_castps_si128(self.data), 31), 31));
        }
    }

    // Returns the per component minimum of _a and _b.
    #[inline]
    pub fn min(&self, _b: SimdFloat4) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(_mm_min_ps(self.data, _b.data));
        }
    }

    // Returns the per component maximum of _a and _b.
    #[inline]
    pub fn max(&self, _b: SimdFloat4) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(_mm_max_ps(self.data, _b.data));
        }
    }

    // Returns the per component minimum of _v and 0.
    #[inline]
    pub fn min0(&self) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(_mm_min_ps(_mm_setzero_ps(), self.data));
        }
    }

    // Returns the per component maximum of _v and 0.
    #[inline]
    pub fn max0(&self) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(_mm_max_ps(_mm_setzero_ps(), self.data));
        }
    }

    // Clamps each element of _x between _a and _b.
    // Result is unknown if _a is not less or equal to _b.
    #[inline]
    pub fn clamp(&self, _a: SimdFloat4, _b: SimdFloat4) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(_mm_max_ps(_a.data, _mm_min_ps(self.data, _b.data)));
        }
    }

    // Computes the length of the components x and y of _v, and stores it in the x
    // component of the returned vector. y, z, w of the returned vector are
    // undefined.
    #[inline]
    pub fn length2(&self) -> SimdFloat4 {
        unsafe {
            let sq_len;
            ozz_sse_dot2_f!(self.data, self.data, sq_len);
            return SimdFloat4::new(_mm_sqrt_ss(sq_len));
        }
    }

    // Computes the length of the components x, y and z of _v, and stores it in the
    // x component of the returned vector. undefined.
    #[inline]
    pub fn length3(&self) -> SimdFloat4 {
        unsafe {
            let sq_len;
            ozz_sse_dot3_f!(self.data, self.data, sq_len);
            return SimdFloat4::new(_mm_sqrt_ss(sq_len));
        }
    }

    // Computes the length of _v, and stores it in the x component of the returned
    // vector. y, z, w of the returned vector are undefined.
    #[inline]
    pub fn length4(&self) -> SimdFloat4 {
        unsafe {
            let sq_len;
            ozz_sse_dot4_f!(self.data, self.data, sq_len);
            return SimdFloat4::new(_mm_sqrt_ss(sq_len));
        }
    }

    // Computes the square length of the components x and y of _v, and stores it
    // in the x component of the returned vector. y, z, w of the returned vector are
    // undefined.
    #[inline]
    pub fn length2sqr(&self) -> SimdFloat4 {
        unsafe {
            let sq_len;
            ozz_sse_dot2_f!(self.data, self.data, sq_len);
            return SimdFloat4::new(sq_len);
        }
    }


    // Computes the square length of the components x, y and z of _v, and stores it
    // in the x component of the returned vector. y, z, w of the returned vector are
    // undefined.
    #[inline]
    pub fn length3sqr(&self) -> SimdFloat4 {
        unsafe {
            let sq_len;
            ozz_sse_dot3_f!(self.data, self.data, sq_len);
            return SimdFloat4::new(sq_len);
        }
    }


    // Computes the square length of the components x, y, z and w of _v, and stores
    // it in the x component of the returned vector. y, z, w of the returned vector
    // undefined.
    #[inline]
    pub fn length4sqr(&self) -> SimdFloat4 {
        unsafe {
            let sq_len;
            ozz_sse_dot4_f!(self.data, self.data, sq_len);
            return SimdFloat4::new(sq_len);
        }
    }


    // Returns the normalized vector of the components x and y of self, and stores
    // it in the x and y components of the returned vector. z and w of the returned
    // vector are the same as their respective components in self.
    #[inline]
    pub fn normalize2(&self) -> SimdFloat4 {
        unsafe {
            let sq_len;
            ozz_sse_dot2_f!(self.data, self.data, sq_len);
            debug_assert!(_mm_cvtss_f32(sq_len) != 0.0 && "self is not normalizable".parse().unwrap());
            let inv_len = _mm_div_ss(SimdFloat4::one().data, _mm_sqrt_ss(sq_len));
            let inv_lenxxxx = ozz_sse_splat_f!(inv_len, 0);
            let norm = _mm_mul_ps(self.data, inv_lenxxxx);
            return SimdFloat4::new(_mm_movelh_ps(norm, _mm_movehl_ps(self.data, self.data)));
        }
    }


    // Returns the normalized vector of the components x, y and z of self, and stores
    // it in the x, y and z components of the returned vector. w of the returned
    // vector is the same as its respective component in self.
    #[inline]
    pub fn normalize3(&self) -> SimdFloat4 {
        unsafe {
            let sq_len;
            ozz_sse_dot3_f!(self.data, self.data, sq_len);
            debug_assert!(_mm_cvtss_f32(sq_len) != 0.0 && "self is not normalizable".parse().unwrap());
            let inv_len = _mm_div_ss(SimdFloat4::one().data, _mm_sqrt_ss(sq_len));
            let vwxyz = ozz_shuffle_ps1!(self.data, _mm_shuffle!(0, 1, 2, 3));
            let inv_lenxxxx = ozz_sse_splat_f!(inv_len, 0);
            let normwxyz = _mm_move_ss(_mm_mul_ps(vwxyz, inv_lenxxxx), vwxyz);
            return SimdFloat4::new(ozz_shuffle_ps1!(normwxyz, _mm_shuffle!(0, 1, 2, 3)));
        }
    }


    // Returns the normalized vector self.
    #[inline]
    pub fn normalize4(&self) -> SimdFloat4 {
        unsafe {
            let sq_len;
            ozz_sse_dot4_f!(self.data, self.data, sq_len);
            debug_assert!(_mm_cvtss_f32(sq_len) != 0.0 && "self is not normalizable".parse().unwrap());
            let inv_len = _mm_div_ss(SimdFloat4::one().data, _mm_sqrt_ss(sq_len));
            let inv_lenxxxx = ozz_sse_splat_f!(inv_len, 0);
            return SimdFloat4::new(_mm_mul_ps(self.data, inv_lenxxxx));
        }
    }


    // Returns the estimated normalized vector of the components x and y of self, and
    // stores it in the x and y components of the returned vector. z and w of the
    // returned vector are the same as their respective components in self.
    #[inline]
    pub fn normalize_est2(&self) -> SimdFloat4 {
        unsafe {
            let sq_len;
            ozz_sse_dot2_f!(self.data, self.data, sq_len);
            debug_assert!(_mm_cvtss_f32(sq_len) != 0.0 && "self is not normalizable".parse().unwrap());
            let inv_len = _mm_rsqrt_ss(sq_len);
            let inv_lenxxxx = ozz_sse_splat_f!(inv_len, 0);
            let norm = _mm_mul_ps(self.data, inv_lenxxxx);
            return SimdFloat4::new(_mm_movelh_ps(norm, _mm_movehl_ps(self.data, self.data)));
        }
    }


    // Returns the estimated normalized vector of the components x, y and z of self,
    // and stores it in the x, y and z components of the returned vector. w of the
    // returned vector is the same as its respective component in self.
    #[inline]
    pub fn normalize_est3(&self) -> SimdFloat4 {
        unsafe {
            let sq_len;
            ozz_sse_dot3_f!(self.data, self.data, sq_len);
            debug_assert!(_mm_cvtss_f32(sq_len) != 0.0 && "self is not normalizable".parse().unwrap());
            let inv_len = _mm_rsqrt_ss(sq_len);
            let vwxyz = ozz_shuffle_ps1!(self.data, _mm_shuffle!(0, 1, 2, 3));
            let inv_lenxxxx = ozz_sse_splat_f!(inv_len, 0);
            let normwxyz = _mm_move_ss(_mm_mul_ps(vwxyz, inv_lenxxxx), vwxyz);
            return SimdFloat4::new(ozz_shuffle_ps1!(normwxyz, _mm_shuffle!(0, 1, 2, 3)));
        }
    }


    // Returns the estimated normalized vector self.
    #[inline]
    pub fn normalize_est4(&self) -> SimdFloat4 {
        unsafe {
            let sq_len;
            ozz_sse_dot4_f!(self.data, self.data, sq_len);
            debug_assert!(_mm_cvtss_f32(sq_len) != 0.0 && "self is not normalizable".parse().unwrap());
            let inv_len = _mm_rsqrt_ss(sq_len);
            let inv_lenxxxx = ozz_sse_splat_f!(inv_len, 0);
            return SimdFloat4::new(_mm_mul_ps(self.data, inv_lenxxxx));
        }
    }

    // Tests if the components x and y of self forms a normalized vector.
    // Returns the result in the x component of the returned vector. y, z and w are
    // set to 0.
    #[inline]
    pub fn is_normalized2(&self) -> SimdInt4 {
        unsafe {
            let max = _mm_set_ss(1.0 + crate::math_constant::K_NORMALIZATION_TOLERANCE_SQ);
            let min = _mm_set_ss(1.0 - crate::math_constant::K_NORMALIZATION_TOLERANCE_SQ);
            let dot;
            ozz_sse_dot2_f!(self.data, self.data, dot);
            let dotx000 = _mm_move_ss(_mm_setzero_ps(), dot);
            return SimdInt4::new(_mm_castps_si128(
                _mm_and_ps(_mm_cmplt_ss(dotx000, max), _mm_cmpgt_ss(dotx000, min))));
        }
    }

    // Tests if the components x, y and z of self forms a normalized vector.
    // Returns the result in the x component of the returned vector. y, z and w are
    // set to 0.
    #[inline]
    pub fn is_normalized3(&self) -> SimdInt4 {
        unsafe {
            let max = _mm_set_ss(1.0 + crate::math_constant::K_NORMALIZATION_TOLERANCE_SQ);
            let min = _mm_set_ss(1.0 - crate::math_constant::K_NORMALIZATION_TOLERANCE_SQ);
            let dot;
            ozz_sse_dot3_f!(self.data, self.data, dot);
            let dotx000 = _mm_move_ss(_mm_setzero_ps(), dot);
            return SimdInt4::new(_mm_castps_si128(
                _mm_and_ps(_mm_cmplt_ss(dotx000, max), _mm_cmpgt_ss(dotx000, min))));
        }
    }

    // Tests if the self is a normalized vector.
    // Returns the result in the x component of the returned vector. y, z and w are
    // set to 0.
    #[inline]
    pub fn is_normalized4(&self) -> SimdInt4 {
        unsafe {
            let max = _mm_set_ss(1.0 + crate::math_constant::K_NORMALIZATION_TOLERANCE_SQ);
            let min = _mm_set_ss(1.0 - crate::math_constant::K_NORMALIZATION_TOLERANCE_SQ);
            let dot;
            ozz_sse_dot4_f!(self.data, self.data, dot);
            let dotx000 = _mm_move_ss(_mm_setzero_ps(), dot);
            return SimdInt4::new(_mm_castps_si128(
                _mm_and_ps(_mm_cmplt_ss(dotx000, max), _mm_cmpgt_ss(dotx000, min))));
        }
    }

    // Tests if the components x and y of self forms a normalized vector.
    // Uses the estimated normalization coefficient, that matches estimated math
    // functions (RecpEst, MormalizeEst...).
    // Returns the result in the x component of the returned vector. y, z and w are
    // set to 0.
    #[inline]
    pub fn is_normalized_est2(&self) -> SimdInt4 {
        unsafe {
            let max = _mm_set_ss(1.0 + crate::math_constant::K_NORMALIZATION_TOLERANCE_EST_SQ);
            let min = _mm_set_ss(1.0 - crate::math_constant::K_NORMALIZATION_TOLERANCE_EST_SQ);
            let dot;
            ozz_sse_dot2_f!(self.data, self.data, dot);
            let dotx000 = _mm_move_ss(_mm_setzero_ps(), dot);
            return SimdInt4::new(_mm_castps_si128(
                _mm_and_ps(_mm_cmplt_ss(dotx000, max), _mm_cmpgt_ss(dotx000, min))));
        }
    }

    // Tests if the components x, y and z of _v forms a normalized vector.
    // Uses the estimated normalization coefficient, that matches estimated math
    // functions (RecpEst, MormalizeEst...).
    // Returns the result in the x component of the returned vector. y, z and w are
    // set to 0.
    #[inline]
    pub fn is_normalized_est3(&self) -> SimdInt4 {
        unsafe {
            let max = _mm_set_ss(1.0 + crate::math_constant::K_NORMALIZATION_TOLERANCE_EST_SQ);
            let min = _mm_set_ss(1.0 - crate::math_constant::K_NORMALIZATION_TOLERANCE_EST_SQ);
            let dot;
            ozz_sse_dot3_f!(self.data, self.data, dot);
            let dotx000 = _mm_move_ss(_mm_setzero_ps(), dot);
            return SimdInt4::new(_mm_castps_si128(
                _mm_and_ps(_mm_cmplt_ss(dotx000, max), _mm_cmpgt_ss(dotx000, min))));
        }
    }

    // Tests if the _v is a normalized vector.
    // Uses the estimated normalization coefficient, that matches estimated math
    // functions (RecpEst, MormalizeEst...).
    // Returns the result in the x component of the returned vector. y, z and w are
    // set to 0.
    #[inline]
    pub fn is_normalized_est4(&self) -> SimdInt4 {
        unsafe {
            let max = _mm_set_ss(1.0 + crate::math_constant::K_NORMALIZATION_TOLERANCE_EST_SQ);
            let min = _mm_set_ss(1.0 - crate::math_constant::K_NORMALIZATION_TOLERANCE_EST_SQ);
            let dot;
            ozz_sse_dot4_f!(self.data, self.data, dot);
            let dotx000 = _mm_move_ss(_mm_setzero_ps(), dot);
            return SimdInt4::new(_mm_castps_si128(
                _mm_and_ps(_mm_cmplt_ss(dotx000, max), _mm_cmpgt_ss(dotx000, min))));
        }
    }

    // Returns the normalized vector of the components x and y of _v if it is
    // normalizable, otherwise returns _safe. z and w of the returned vector are
    // the same as their respective components in _v.
    #[inline]
    pub fn normalize_safe2(&self, _safe: SimdFloat4) -> SimdFloat4 {
        unsafe {
            let sq_len;
            ozz_sse_dot2_f!(self.data, self.data, sq_len);
            let inv_len = _mm_div_ss(SimdFloat4::one().data, _mm_sqrt_ss(sq_len));
            let inv_lenxxxx = ozz_sse_splat_f!(inv_len, 0);
            let norm = _mm_mul_ps(self.data, inv_lenxxxx);
            let cond = _mm_castps_si128(
                _mm_cmple_ps(ozz_sse_splat_f!(sq_len, 0), _mm_setzero_ps()));
            let cfalse = _mm_movelh_ps(norm, _mm_movehl_ps(self.data, self.data));
            return SimdFloat4::new(ozz_sse_select_f!(cond, _safe.data, cfalse));
        }
    }

    // Returns the normalized vector of the components x, y, z and w of _v if it is
    // normalizable, otherwise returns _safe. w of the returned vector is the same
    // as its respective components in _v.
    #[inline]
    pub fn normalize_safe3(&self, _safe: SimdFloat4) -> SimdFloat4 {
        unsafe {
            let sq_len;
            ozz_sse_dot3_f!(self.data, self.data, sq_len);
            let inv_len = _mm_div_ss(SimdFloat4::one().data, _mm_sqrt_ss(sq_len));
            let vwxyz = ozz_shuffle_ps1!(self.data, _mm_shuffle!(0, 1, 2, 3));
            let inv_lenxxxx = ozz_sse_splat_f!(inv_len, 0);
            let normwxyz = _mm_move_ss(_mm_mul_ps(vwxyz, inv_lenxxxx), vwxyz);
            let cond = _mm_castps_si128(
                _mm_cmple_ps(ozz_sse_splat_f!(sq_len, 0), _mm_setzero_ps()));
            let cfalse = ozz_shuffle_ps1!(normwxyz, _mm_shuffle!(0, 1, 2, 3));
            return SimdFloat4::new(ozz_sse_select_f!(cond, _safe.data, cfalse));
        }
    }

    // Returns the normalized vector _v if it is normalizable, otherwise returns
    // _safe.
    #[inline]
    pub fn normalize_safe4(&self, _safe: SimdFloat4) -> SimdFloat4 {
        unsafe {
            let sq_len;
            ozz_sse_dot4_f!(self.data, self.data, sq_len);
            let inv_len = _mm_div_ss(SimdFloat4::one().data, _mm_sqrt_ss(sq_len));
            let inv_lenxxxx = ozz_sse_splat_f!(inv_len, 0);
            let cond = _mm_castps_si128(
                _mm_cmple_ps(ozz_sse_splat_f!(sq_len, 0), _mm_setzero_ps()));
            let cfalse = _mm_mul_ps(self.data, inv_lenxxxx);
            return SimdFloat4::new(ozz_sse_select_f!(cond, _safe.data, cfalse));
        }
    }

    // Returns the estimated normalized vector of the components x and y of _v if it
    // is normalizable, otherwise returns _safe. z and w of the returned vector are
    // the same as their respective components in _v.
    #[inline]
    pub fn normalize_safe_est2(&self, _safe: SimdFloat4) -> SimdFloat4 {
        unsafe {
            let sq_len;
            ozz_sse_dot2_f!(self.data, self.data, sq_len);
            let inv_len = _mm_rsqrt_ss(sq_len);
            let inv_lenxxxx = ozz_sse_splat_f!(inv_len, 0);
            let norm = _mm_mul_ps(self.data, inv_lenxxxx);
            let cond = _mm_castps_si128(
                _mm_cmple_ps(ozz_sse_splat_f!(sq_len, 0), _mm_setzero_ps()));
            let cfalse = _mm_movelh_ps(norm, _mm_movehl_ps(self.data, self.data));
            return SimdFloat4::new(ozz_sse_select_f!(cond, _safe.data, cfalse));
        }
    }

    // Returns the estimated normalized vector of the components x, y, z and w of _v
    // if it is normalizable, otherwise returns _safe. w of the returned vector is
    // the same as its respective components in _v.
    #[inline]
    pub fn normalize_safe_est3(&self, _safe: SimdFloat4) -> SimdFloat4 {
        unsafe {
            let sq_len;
            ozz_sse_dot3_f!(self.data, self.data, sq_len);
            let inv_len = _mm_rsqrt_ss(sq_len);
            let vwxyz = ozz_shuffle_ps1!(self.data, _mm_shuffle!(0, 1, 2, 3));
            let inv_lenxxxx = ozz_sse_splat_f!(inv_len, 0);
            let normwxyz = _mm_move_ss(_mm_mul_ps(vwxyz, inv_lenxxxx), vwxyz);
            let cond = _mm_castps_si128(
                _mm_cmple_ps(ozz_sse_splat_f!(sq_len, 0), _mm_setzero_ps()));
            let cfalse = ozz_shuffle_ps1!(normwxyz, _mm_shuffle!(0, 1, 2, 3));
            return SimdFloat4::new(ozz_sse_select_f!(cond, _safe.data, cfalse));
        }
    }

    // Returns the estimated normalized vector _v if it is normalizable, otherwise
    // returns _safe.
    #[inline]
    pub fn normalize_safe_est4(&self, _safe: SimdFloat4) -> SimdFloat4 {
        unsafe {
            let sq_len;
            ozz_sse_dot4_f!(self.data, self.data, sq_len);
            let inv_len = _mm_rsqrt_ss(sq_len);
            let inv_lenxxxx = ozz_sse_splat_f!(inv_len, 0);
            let cond = _mm_castps_si128(
                _mm_cmple_ps(ozz_sse_splat_f!(sq_len, 0), _mm_setzero_ps()));
            let cfalse = _mm_mul_ps(self.data, inv_lenxxxx);
            return SimdFloat4::new(ozz_sse_select_f!(cond, _safe.data, cfalse));
        }
    }

    // Computes the per element linear interpolation of _a and _b, where _alpha is
    // not bound to range [0,1].
    #[inline]
    pub fn lerp(&self, _b: SimdFloat4, _alpha: SimdFloat4) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(ozz_madd!(_alpha.data, _mm_sub_ps(_b.data, self.data), self.data));
        }
    }

    // Computes the per element cosine of _v.
    #[inline]
    pub fn cos(&self) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(_mm_set_ps(f32::cos(self.get_w()), f32::cos(self.get_z()),
                                              f32::cos(self.get_y()), f32::cos(self.get_x())));
        }
    }

    // Computes the cosine of the x component of _v and stores it in the x
    // component of the returned vector. y, z and w of the returned vector are the
    // same as their respective components in _v.
    #[inline]
    pub fn cos_x(&self) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(_mm_move_ss(self.data, _mm_set_ps1(f32::cos(self.get_x()))));
        }
    }

    // Computes the per element arccosine of _v.
    #[inline]
    pub fn acos(&self) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(_mm_set_ps(f32::acos(self.get_w()), f32::acos(self.get_z()),
                                              f32::acos(self.get_y()), f32::acos(self.get_x())));
        }
    }

    // Computes the arccosine of the x component of _v and stores it in the x
    // component of the returned vector. y, z and w of the returned vector are the
    // same as their respective components in _v.
    #[inline]
    pub fn acos_x(&self) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(_mm_move_ss(self.data, _mm_set_ps1(f32::acos(self.get_x()))));
        }
    }

    // Computes the per element sines of _v.
    #[inline]
    pub fn sin(&self) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(_mm_set_ps(f32::sin(self.get_w()), f32::sin(self.get_z()),
                                              f32::sin(self.get_y()), f32::sin(self.get_x())));
        }
    }

    // Computes the sines of the x component of _v and stores it in the x
    // component of the returned vector. y, z and w of the returned vector are the
    // same as their respective components in _v.
    #[inline]
    pub fn sin_x(&self) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(_mm_move_ss(self.data, _mm_set_ps1(f32::sin(self.get_x()))));
        }
    }

    // Computes the per element arcsine of _v.
    #[inline]
    pub fn asin(&self) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(_mm_set_ps(f32::asin(self.get_w()), f32::asin(self.get_z()),
                                              f32::asin(self.get_y()), f32::asin(self.get_x())));
        }
    }

    // Computes the arcsine of the x component of _v and stores it in the x
    // component of the returned vector. y, z and w of the returned vector are the
    // same as their respective components in _v.
    #[inline]
    pub fn asin_x(&self) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(_mm_move_ss(self.data, _mm_set_ps1(f32::asin(self.get_x()))));
        }
    }

    // Computes the per element tangent of _v.
    #[inline]
    pub fn tan(&self) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(_mm_set_ps(f32::tan(self.get_w()), f32::tan(self.get_z()),
                                              f32::tan(self.get_y()), f32::tan(self.get_x())));
        }
    }

    // Computes the tangent of the x component of _v and stores it in the x
    // component of the returned vector. y, z and w of the returned vector are the
    // same as their respective components in _v.
    #[inline]
    pub fn tan_x(&self) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(_mm_move_ss(self.data, _mm_set_ps1(f32::tan(self.get_x()))));
        }
    }

    // Computes the per element arctangent of _v.
    #[inline]
    pub fn atan(&self) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(_mm_set_ps(f32::atan(self.get_w()), f32::atan(self.get_z()),
                                              f32::atan(self.get_y()), f32::atan(self.get_x())));
        }
    }

    // Computes the arctangent of the x component of _v and stores it in the x
    // component of the returned vector. y, z and w of the returned vector are the
    // same as their respective components in _v.
    #[inline]
    pub fn atan_x(&self) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(_mm_move_ss(self.data, _mm_set_ps1(f32::atan(self.get_x()))));
        }
    }

    // Returns boolean selection of vectors _true and _false according to condition
    // _b. All bits a each component of _b must have the same value (O or
    // 0xffffffff) to ensure portability.
    #[inline]
    pub fn select(_b: SimdInt4, _true: SimdFloat4, _false: SimdFloat4) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(ozz_sse_select_f!(_b.data, _true.data, _false.data));
        }
    }

    // Per element "equal" comparison of _a and _b.
    #[inline]
    pub fn cmp_eq(&self, _b: SimdFloat4) -> SimdInt4 {
        unsafe {
            return SimdInt4::new(_mm_castps_si128(_mm_cmpeq_ps(self.data, _b.data)));
        }
    }

    // Per element "not equal" comparison of _a and _b.
    #[inline]
    pub fn cmp_ne(&self, _b: SimdFloat4) -> SimdInt4 {
        unsafe {
            return SimdInt4::new(_mm_castps_si128(_mm_cmpneq_ps(self.data, _b.data)));
        }
    }

    // Per element "less than" comparison of _a and _b.
    #[inline]
    pub fn cmp_lt(&self, _b: SimdFloat4) -> SimdInt4 {
        unsafe {
            return SimdInt4::new(_mm_castps_si128(_mm_cmplt_ps(self.data, _b.data)));
        }
    }

    // Per element "less than or equal" comparison of _a and _b.
    #[inline]
    pub fn cmp_le(&self, _b: SimdFloat4) -> SimdInt4 {
        unsafe {
            return SimdInt4::new(_mm_castps_si128(_mm_cmple_ps(self.data, _b.data)));
        }
    }

    // Per element "greater than" comparison of _a and _b.
    #[inline]
    pub fn cmp_gt(&self, _b: SimdFloat4) -> SimdInt4 {
        unsafe {
            return SimdInt4::new(_mm_castps_si128(_mm_cmpgt_ps(self.data, _b.data)));
        }
    }

    // Per element "greater than or equal" comparison of _a and _b.
    #[inline]
    pub fn cmp_ge(&self, _b: SimdFloat4) -> SimdInt4 {
        unsafe {
            return SimdInt4::new(_mm_castps_si128(_mm_cmpge_ps(self.data, _b.data)));
        }
    }

    // Returns per element binary and operation of _a and _b.
    // _v[0...127] = _a[0...127] & _b[0...127]
    #[inline]
    pub fn and_ff(&self, _b: SimdFloat4) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(_mm_and_ps(self.data, _b.data));
        }
    }

    // Returns per element binary or operation of _a and _b.
    // _v[0...127] = _a[0...127] | _b[0...127]
    #[inline]
    pub fn or_ff(&self, _b: SimdFloat4) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(_mm_or_ps(self.data, _b.data));
        }
    }

    // Returns per element binary logical xor operation of _a and _b.
    // _v[0...127] = _a[0...127] ^ _b[0...127]
    #[inline]
    pub fn xor_ff(&self, _b: SimdFloat4) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(_mm_xor_ps(self.data, _b.data));
        }
    }

    // Returns per element binary and operation of _a and _b.
    // _v[0...127] = _a[0...127] & _b[0...127]
    #[inline]
    pub fn and_fi(&self, _b: SimdInt4) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(_mm_and_ps(self.data, _mm_castsi128_ps(_b.data)));
        }
    }

    // Returns per element binary and operation of _a and ~_b.
    // _v[0...127] = _a[0...127] & ~_b[0...127]
    #[inline]
    pub fn and_not(&self, _b: SimdInt4) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(_mm_andnot_ps(_mm_castsi128_ps(_b.data), self.data));
        }
    }

    // Returns per element binary or operation of _a and _b.
    // _v[0...127] = _a[0...127] | _b[0...127]
    #[inline]
    pub fn or_fi(&self, _b: SimdInt4) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(_mm_or_ps(self.data, _mm_castsi128_ps(_b.data)));
        }
    }

    // Returns per element binary logical xor operation of _a and _b.
    // _v[0...127] = _a[0...127] ^ _b[0...127]
    #[inline]
    pub fn xor_fi(&self, _b: SimdInt4) -> SimdFloat4 {
        unsafe {
            return SimdFloat4::new(_mm_xor_ps(self.data, _mm_castsi128_ps(_b.data)));
        }
    }
}

// //--------------------------------------------------------------------------------------------------
// Vector of four integer values.
pub struct SimdInt4 {
    pub data: __m128i,
}

pub union SimdInt4Union {
    ret: __m128i,
    af: [i32; 4],
}

impl SimdInt4 {
    #[inline]
    pub fn new(data: __m128i) -> SimdInt4 {
        return SimdInt4 {
            data
        };
    }

    // Returns a SimdInt4 vector with all components set to 0.
    #[inline]
    pub fn zero() -> SimdInt4 {
        unsafe {
            return SimdInt4::new(_mm_setzero_si128());
        }
    }

    // Returns a SimdInt4 vector with all components set to 1.
    #[inline]
    pub fn one() -> SimdInt4 {
        unsafe {
            let zero = _mm_setzero_si128();
            return SimdInt4::new(_mm_sub_epi32(zero, _mm_cmpeq_epi32(zero, zero)));
        }
    }

    // Returns a SimdInt4 vector with the x component set to 1 and all the others
    // to 0.
    #[inline]
    pub fn x_axis() -> SimdInt4 {
        unsafe {
            let zero = _mm_setzero_si128();
            return SimdInt4::new(_mm_srli_si128(_mm_sub_epi32(zero, _mm_cmpeq_epi32(zero, zero)), 12));
        }
    }

    // Returns a SimdInt4 vector with the y component set to 1 and all the others
    // to 0.
    #[inline]
    pub fn y_axis() -> SimdInt4 {
        unsafe {
            let zero = _mm_setzero_si128();
            return SimdInt4::new(_mm_slli_si128(
                _mm_srli_si128(_mm_sub_epi32(zero, _mm_cmpeq_epi32(zero, zero)), 12), 4));
        }
    }

    // Returns a SimdInt4 vector with the z component set to 1 and all the others
    // to 0.
    #[inline]
    pub fn z_axis() -> SimdInt4 {
        unsafe {
            let zero = _mm_setzero_si128();
            return SimdInt4::new(_mm_slli_si128(
                _mm_srli_si128(_mm_sub_epi32(zero, _mm_cmpeq_epi32(zero, zero)), 12), 8));
        }
    }

    // Returns a SimdInt4 vector with the w component set to 1 and all the others
    // to 0.
    #[inline]
    pub fn w_axis() -> SimdInt4 {
        unsafe {
            let zero = _mm_setzero_si128();
            return SimdInt4::new(_mm_slli_si128(_mm_sub_epi32(zero, _mm_cmpeq_epi32(zero, zero)), 12));
        }
    }

    // Returns a SimdInt4 vector with all components set to true (0xffffffff).
    #[inline]
    pub fn all_true() -> SimdInt4 {
        unsafe {
            let zero = _mm_setzero_si128();
            return SimdInt4::new(_mm_cmpeq_epi32(zero, zero));
        }
    }

    // Returns a SimdInt4 vector with all components set to false (0).
    #[inline]
    pub fn all_false() -> SimdInt4 {
        unsafe {
            return SimdInt4::new(_mm_setzero_si128());
        }
    }

    // Returns a SimdInt4 vector with sign bits set to 1.
    #[inline]
    pub fn mask_sign() -> SimdInt4 {
        unsafe {
            let zero = _mm_setzero_si128();
            return SimdInt4::new(_mm_slli_epi32(_mm_cmpeq_epi32(zero, zero), 31));
        }
    }

    // Returns a SimdInt4 vector with all bits set to 1 except sign.
    #[inline]
    pub fn mask_not_sign() -> SimdInt4 {
        unsafe {
            let zero = _mm_setzero_si128();
            return SimdInt4::new(_mm_srli_epi32(_mm_cmpeq_epi32(zero, zero), 1));
        }
    }

    // Returns a SimdInt4 vector with sign bits of x, y and z components set to 1.
    #[inline]
    pub fn mask_sign_xyz() -> SimdInt4 {
        unsafe {
            let zero = _mm_setzero_si128();
            return SimdInt4::new(_mm_srli_si128(_mm_slli_epi32(_mm_cmpeq_epi32(zero, zero), 31), 4));
        }
    }

    // Returns a SimdInt4 vector with sign bits of w component set to 1.
    #[inline]
    pub fn mask_sign_w() -> SimdInt4 {
        unsafe {
            let zero = _mm_setzero_si128();
            return SimdInt4::new(_mm_slli_si128(_mm_slli_epi32(_mm_cmpeq_epi32(zero, zero), 31), 12));
        }
    }

    // Returns a SimdInt4 vector with all bits set to 1.
    #[inline]
    pub fn mask_ffff() -> SimdInt4 {
        unsafe {
            let zero = _mm_setzero_si128();
            return SimdInt4::new(_mm_cmpeq_epi32(zero, zero));
        }
    }

    // Returns a SimdInt4 vector with all bits set to 0.
    #[inline]
    pub fn mask_0000() -> SimdInt4 {
        unsafe {
            return SimdInt4::new(_mm_setzero_si128());
        }
    }

    // Returns a SimdInt4 vector with all the bits of the x, y, z components set to
    // 1, while z is set to 0.
    #[inline]
    pub fn mask_fff0() -> SimdInt4 {
        unsafe {
            let zero = _mm_setzero_si128();
            return SimdInt4::new(_mm_srli_si128(_mm_cmpeq_epi32(zero, zero), 4));
        }
    }

    // Returns a SimdInt4 vector with all the bits of the x component set to 1,
    // while the others are set to 0.
    pub fn mask_f000() -> SimdInt4 {
        unsafe {
            let zero = _mm_setzero_si128();
            return SimdInt4::new(_mm_srli_si128(_mm_cmpeq_epi32(zero, zero), 12));
        }
    }

    // Returns a SimdInt4 vector with all the bits of the y component set to 1,
    // while the others are set to 0.
    #[inline]
    pub fn mask_0f00() -> SimdInt4 {
        unsafe {
            let zero = _mm_setzero_si128();
            return SimdInt4::new(_mm_srli_si128(_mm_slli_si128(_mm_cmpeq_epi32(zero, zero), 12), 8));
        }
    }

    // Returns a SimdInt4 vector with all the bits of the z component set to 1,
    // while the others are set to 0.
    #[inline]
    pub fn mask_00f0() -> SimdInt4 {
        unsafe {
            let zero = _mm_setzero_si128();
            return SimdInt4::new(_mm_srli_si128(_mm_slli_si128(_mm_cmpeq_epi32(zero, zero), 12), 4));
        }
    }

    // Returns a SimdInt4 vector with all the bits of the w component set to 1,
    // while the others are set to 0.
    #[inline]
    pub fn mask_000f() -> SimdInt4 {
        unsafe {
            let zero = _mm_setzero_si128();
            return SimdInt4::new(_mm_slli_si128(_mm_cmpeq_epi32(zero, zero), 12));
        }
    }

    // Loads _x, _y, _z, _w to the returned vector.
    // r.x = _x
    // r.y = _y
    // r.z = _z
    // r.w = _w
    #[inline]
    pub fn load_i32(_x: i32, _y: i32, _z: i32, _w: i32) -> SimdInt4 {
        unsafe {
            return SimdInt4::new(_mm_set_epi32(_w, _z, _y, _x));
        }
    }

    #[inline]
    pub fn load_x_i32(_x: i32) -> SimdInt4 {
        unsafe {
            return SimdInt4::new(_mm_set_epi32(0, 0, 0, _x));
        }
    }

    #[inline]
    pub fn load1_i32(_x: i32) -> SimdInt4 {
        unsafe {
            return SimdInt4::new(_mm_set1_epi32(_x));
        }
    }

    // Loads _x, _y, _z, _w to the returned vector using the following conversion
    // rule.
    // r.x = _x ? 0xffffffff:0
    // r.y = _y ? 0xffffffff:0
    // r.z = _z ? 0xffffffff:0
    // r.w = _w ? 0xffffffff:0
    #[inline]
    pub fn load_bool(_x: bool, _y: bool, _z: bool, _w: bool) -> SimdInt4 {
        unsafe {
            return SimdInt4::new(_mm_sub_epi32(_mm_setzero_si128(), _mm_set_epi32(i32::from(_w),
                                                                                  i32::from(_z),
                                                                                  i32::from(_y),
                                                                                  i32::from(_x))));
        }
    }

    // Loads _x to the x component of the returned vector using the following
    // conversion rule, and sets y, z and w to 0.
    // r.x = _x ? 0xffffffff:0
    // r.y = 0
    // r.z = 0
    // r.w = 0
    #[inline]
    pub fn load_x_bool(_x: bool) -> SimdInt4 {
        unsafe {
            return SimdInt4::new(_mm_sub_epi32(_mm_setzero_si128(), _mm_set_epi32(0, 0, 0, i32::from(_x))));
        }
    }

    // Loads _x to the all the components of the returned vector using the following
    // conversion rule.
    // r.x = _x ? 0xffffffff:0
    // r.y = _x ? 0xffffffff:0
    // r.z = _x ? 0xffffffff:0
    // r.w = _x ? 0xffffffff:0
    #[inline]
    pub fn load1_bool(_x: bool) -> SimdInt4 {
        unsafe {
            return SimdInt4::new(_mm_sub_epi32(_mm_setzero_si128(), _mm_set1_epi32(i32::from(_x))));
        }
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
        unsafe {
            return SimdInt4::new(_mm_cvtsi32_si128(_i[0]));
        }
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
        unsafe {
            return SimdInt4::new(_mm_set_epi32(0, _i[2], _i[1], _i[0]));
        }
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
        unsafe {
            return SimdInt4::new(_mm_cvtsi32_si128(_i[0]));
        }
    }

    // Loads the 4 values of _i to the returned vector.
    // _i must be aligned to 4 bytes.
    // r.x = _i[0]
    // r.y = _i[0]
    // r.z = _i[0]
    // r.w = _i[0]
    #[inline]
    pub fn load1ptr_u(_i: [i32; 4]) -> SimdInt4 {
        unsafe {
            return SimdInt4::new(_mm_set1_epi32(_i[0]));
        }
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
        unsafe {
            return SimdInt4::new(_mm_set_epi32(0, 0, _i[1], _i[0]));
        }
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
        unsafe {
            return SimdInt4::new(_mm_set_epi32(0, _i[2], _i[1], _i[0]));
        }
    }

    // Convert from float to integer by rounding the nearest value.
    #[inline]
    pub fn from_float_round(_f: SimdFloat4) -> SimdInt4 {
        unsafe {
            return SimdInt4::new(_mm_cvtps_epi32(_f.data));
        }
    }

    // Convert from float to integer by truncating.
    #[inline]
    pub fn from_float_trunc(_f: SimdFloat4) -> SimdInt4 {
        unsafe {
            return SimdInt4::new(_mm_cvttps_epi32(_f.data));
        }
    }

    //----------------------------------------------------------------------------------------------
    // Returns the x component of _v as an integer.
    #[inline]
    pub fn get_x(&self) -> i32 {
        unsafe {
            return _mm_cvtsi128_si32(self.data);
        }
    }

    // Returns the y component of _v as a integer.
    #[inline]
    pub fn get_y(&self) -> i32 {
        unsafe {
            return _mm_cvtsi128_si32(ozz_sse_splat_i!(self.data, 1));
        }
    }

    // Returns the z component of _v as a integer.
    #[inline]
    pub fn get_z(&self) -> i32 {
        unsafe {
            return _mm_cvtsi128_si32(_mm_unpackhi_epi32(self.data, self.data));
        }
    }

    // Returns the w component of _v as a integer.
    #[inline]
    pub fn get_w(&self) -> i32 {
        unsafe {
            return _mm_cvtsi128_si32(ozz_sse_splat_i!(self.data, 3));
        }
    }

    // Returns _v with the x component set to x component of _i.
    #[inline]
    pub fn set_x(&mut self, _i: SimdInt4) {
        unsafe {
            self.data = _mm_castps_si128(
                _mm_move_ss(_mm_castsi128_ps(self.data), _mm_castsi128_ps(_i.data)));
        }
    }

    // Returns _v with the y component set to x component of _i.
    #[inline]
    pub fn set_y(&mut self, _i: SimdInt4) {
        unsafe {
            let xfnn = _mm_castsi128_ps(_mm_unpacklo_epi32(self.data, _i.data));
            self.data = _mm_castps_si128(
                _mm_shuffle_ps(xfnn, _mm_castsi128_ps(self.data), _mm_shuffle!(3, 2, 1, 0)));
        }
    }

    // Returns _v with the z component set to x component of _i.
    #[inline]
    pub fn set_z(&mut self, _i: SimdInt4) {
        unsafe {
            let ffww = _mm_shuffle_ps(_mm_castsi128_ps(_i.data), _mm_castsi128_ps(self.data),
                                      _mm_shuffle!(3, 3, 0, 0));
            self.data = _mm_castps_si128(
                _mm_shuffle_ps(_mm_castsi128_ps(self.data), ffww, _mm_shuffle!(2, 0, 1, 0)));
        }
    }

    // Returns _v with the w component set to x component of _i.
    #[inline]
    pub fn set_w(&mut self, _i: SimdInt4) {
        unsafe {
            let ffzz = _mm_shuffle_ps(_mm_castsi128_ps(_i.data), _mm_castsi128_ps(self.data),
                                      _mm_shuffle!(2, 2, 0, 0));
            self.data = _mm_castps_si128(
                _mm_shuffle_ps(_mm_castsi128_ps(self.data), ffzz, _mm_shuffle!(0, 2, 1, 0)));
        }
    }

    // Returns _v with the _ith component set to _i.
    // _i must be in range [0,3]
    #[inline]
    pub fn set_i(&mut self, _i: SimdInt4, _ith: usize) {
        unsafe {
            let mut u = SimdInt4Union {
                ret: self.data,
            };

            u.af[_ith] = _i.get_x();
            self.data = u.ret;
        }
    }

    // Stores the 4 components of _v to the four first integers of _i.
    // _i must be aligned to 16 bytes.
    // _i[0] = _v.x
    // _i[1] = _v.y
    // _i[2] = _v.z
    // _i[3] = _v.w
    #[inline]
    pub fn store_ptr(&self, _i: &mut [i32; 4]) {
        todo!()
    }

    // Stores the x component of _v to the first integers of _i.
    // _i must be aligned to 16 bytes.
    // _i[0] = _v.x
    #[inline]
    pub fn store1ptr(&self, _i: &mut [i32; 4]) {
        unsafe {
            _i[0] = _mm_cvtsi128_si32(self.data);
        }
    }

    // Stores x and y components of _v to the two first integers of _i.
    // _i must be aligned to 16 bytes.
    // _i[0] = _v.x
    // _i[1] = _v.y
    #[inline]
    pub fn store2ptr(&self, _i: &mut [i32; 4]) {
        unsafe {
            _i[0] = _mm_cvtsi128_si32(self.data);
            _i[1] = _mm_cvtsi128_si32(ozz_sse_splat_i!(self.data, 1));
        }
    }

    // Stores x, y and z components of _v to the three first integers of _i.
    // _i must be aligned to 16 bytes.
    // _i[0] = _v.x
    // _i[1] = _v.y
    // _i[2] = _v.z
    #[inline]
    pub fn store3ptr(&self, _i: &mut [i32; 4]) {
        unsafe {
            _i[0] = _mm_cvtsi128_si32(self.data);
            _i[1] = _mm_cvtsi128_si32(ozz_sse_splat_i!(self.data, 1));
            _i[2] = _mm_cvtsi128_si32(_mm_unpackhi_epi32(self.data, self.data));
        }
    }

    // Stores the 4 components of _v to the four first integers of _i.
    // _i must be aligned to 4 bytes.
    // _i[0] = _v.x
    // _i[1] = _v.y
    // _i[2] = _v.z
    // _i[3] = _v.w
    #[inline]
    pub fn store_ptr_u(&self, _i: &mut [i32; 4]) {
        todo!()
    }

    // Stores the x component of _v to the first float of _i.
    // _i must be aligned to 4 bytes.
    // _i[0] = _v.x
    #[inline]
    pub fn store1ptr_u(&self, _i: &mut [i32; 4]) {
        unsafe {
            _i[0] = _mm_cvtsi128_si32(self.data);
        }
    }

    // Stores x and y components of _v to the two first integers of _i.
    // _i must be aligned to 4 bytes.
    // _i[0] = _v.x
    // _i[1] = _v.y
    #[inline]
    pub fn store2ptr_u(&self, _i: &mut [i32; 4]) {
        unsafe {
            _i[0] = _mm_cvtsi128_si32(self.data);
            _i[1] = _mm_cvtsi128_si32(ozz_sse_splat_i!(self.data, 1));
        }
    }

    // Stores x, y and z components of _v to the three first integers of _i.
    // _i must be aligned to 4 bytes.
    // _i[0] = _v.x
    // _i[1] = _v.y
    // _i[2] = _v.z
    #[inline]
    pub fn store3ptr_u(&self, _i: &mut [i32; 4]) {
        unsafe {
            _i[0] = _mm_cvtsi128_si32(self.data);
            _i[1] = _mm_cvtsi128_si32(ozz_sse_splat_i!(self.data, 1));
            _i[2] = _mm_cvtsi128_si32(_mm_unpackhi_epi32(self.data, self.data));
        }
    }

    // Replicates x of _a to all the components of the returned vector.
    #[inline]
    pub fn splat_x(&self) -> SimdInt4 {
        unsafe {
            return SimdInt4::new(ozz_sse_splat_i!(self.data, 0));
        }
    }

    // Replicates y of _a to all the components of the returned vector.
    #[inline]
    pub fn splat_y(&self) -> SimdInt4 {
        unsafe {
            return SimdInt4::new(ozz_sse_splat_i!(self.data, 1));
        }
    }

    // Replicates z of _a to all the components of the returned vector.
    #[inline]
    pub fn splat_z(&self) -> SimdInt4 {
        unsafe {
            return SimdInt4::new(ozz_sse_splat_i!(self.data, 2));
        }
    }

    // Replicates w of _a to all the components of the returned vector.
    #[inline]
    pub fn splat_w(&self) -> SimdInt4 {
        unsafe {
            return SimdInt4::new(ozz_sse_splat_i!(self.data, 3));
        }
    }

    // Swizzle x, y, z and w components based on compile time arguments _X, _Y, _Z
    // and _W. Arguments can vary from 0 (x), to 3 (w).
    #[inline]
    pub fn swizzle0123(&self) -> SimdInt4 {
        return SimdInt4::new(self.data);
    }

    // Creates a 4-bit mask from the most significant bits of each component of _v.
    // i := sign(a3)<<3 | sign(a2)<<2 | sign(a1)<<1 | sign(a0)
    #[inline]
    pub fn move_mask(&self) -> i32 {
        unsafe {
            return _mm_movemask_ps(_mm_castsi128_ps(self.data));
        }
    }

    // Returns true if all the components of _v are not 0.
    #[inline]
    pub fn are_all_true(&self) -> bool {
        unsafe {
            return _mm_movemask_ps(_mm_castsi128_ps(self.data)) == 0xf;
        }
    }

    // Returns true if x, y and z components of _v are not 0.
    #[inline]
    pub fn are_all_true3(&self) -> bool {
        unsafe {
            return (_mm_movemask_ps(_mm_castsi128_ps(self.data)) & 0x7) == 0x7;
        }
    }

    // Returns true if x and y components of _v are not 0.
    #[inline]
    pub fn are_all_true2(&self) -> bool {
        unsafe {
            return (_mm_movemask_ps(_mm_castsi128_ps(self.data)) & 0x3) == 0x3;
        }
    }

    // Returns true if x component of _v is not 0.
    #[inline]
    pub fn are_all_true1(&self) -> bool {
        unsafe {
            return (_mm_movemask_ps(_mm_castsi128_ps(self.data)) & 0x1) == 0x1;
        }
    }

    // Returns true if all the components of _v are 0.
    #[inline]
    pub fn are_all_false(&self) -> bool {
        unsafe {
            return _mm_movemask_ps(_mm_castsi128_ps(self.data)) == 0;
        }
    }

    // Returns true if x, y and z components of _v are 0.
    #[inline]
    pub fn are_all_false3(&self) -> bool {
        unsafe {
            return (_mm_movemask_ps(_mm_castsi128_ps(self.data)) & 0x7) == 0;
        }
    }

    // Returns true if x and y components of _v are 0.
    #[inline]
    pub fn are_all_false2(&self) -> bool {
        unsafe {
            return (_mm_movemask_ps(_mm_castsi128_ps(self.data)) & 0x3) == 0;
        }
    }

    // Returns true if x component of _v is 0.
    #[inline]
    pub fn are_all_false1(&self) -> bool {
        unsafe {
            return (_mm_movemask_ps(_mm_castsi128_ps(self.data)) & 0x1) == 0;
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
    pub fn hadd2(&self) -> SimdInt4 {
        unsafe {
            let hadd = _mm_add_epi32(self.data, ozz_sse_splat_i!(self.data, 1));
            return SimdInt4::new(_mm_castps_si128(
                _mm_move_ss(_mm_castsi128_ps(self.data), _mm_castsi128_ps(hadd))));
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
    pub fn hadd3(&self) -> SimdInt4 {
        unsafe {
            let hadd = _mm_add_epi32(_mm_add_epi32(self.data, ozz_sse_splat_i!(self.data, 1)),
                                     _mm_unpackhi_epi32(self.data, self.data));
            return SimdInt4::new(_mm_castps_si128(
                _mm_move_ss(_mm_castsi128_ps(self.data), _mm_castsi128_ps(hadd))));
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
    pub fn hadd4(&self) -> SimdInt4 {
        unsafe {
            let v = _mm_castsi128_ps(self.data);
            let haddxyzw =
                _mm_add_epi32(self.data, _mm_castps_si128(_mm_movehl_ps(v, v)));
            return SimdInt4::new(_mm_castps_si128(_mm_move_ss(
                v,
                _mm_castsi128_ps(_mm_add_epi32(haddxyzw, ozz_sse_splat_i!(haddxyzw, 1))))));
        }
    }

    // Returns the per element absolute value of _v.
    #[inline]
    pub fn abs(&self) -> SimdInt4 {
        unsafe {
            return SimdInt4::new(_mm_abs_epi32(self.data));
        }
    }

    // Returns the sign bit of _v.
    #[inline]
    pub fn sign(&self) -> SimdInt4 {
        unsafe {
            return SimdInt4::new(_mm_slli_epi32(_mm_srli_epi32(self.data, 31), 31));
        }
    }

    // Returns the per component minimum of _a and _b.
    #[inline]
    pub fn min(&self, _b: SimdInt4) -> SimdInt4 {
        unsafe {
            return SimdInt4::new(_mm_min_epi32(self.data, _b.data));
        }
    }

    // Returns the per component maximum of _a and _b.
    #[inline]
    pub fn max(&self, _b: SimdInt4) -> SimdInt4 {
        unsafe {
            return SimdInt4::new(_mm_max_epi32(self.data, _b.data));
        }
    }

    // Returns the per component minimum of _v and 0.
    #[inline]
    pub fn min0(&self) -> SimdInt4 {
        unsafe {
            let zero = _mm_setzero_si128();
            return SimdInt4::new(_mm_min_epi32(zero, self.data));
        }
    }

    // Returns the per component maximum of _v and 0.
    #[inline]
    pub fn max0(&self) -> SimdInt4 {
        unsafe {
            let zero = _mm_setzero_si128();
            return SimdInt4::new(_mm_max_epi32(zero, self.data));
        }
    }

    // Clamps each element of _x between _a and _b.
    // Result is unknown if _a is not less or equal to _b.
    #[inline]
    pub fn clamp(&self, _a: SimdInt4, _b: SimdInt4) -> SimdInt4 {
        unsafe {
            return SimdInt4::new(_mm_min_epi32(_mm_max_epi32(_a.data, self.data), _b.data));
        }
    }

    // Returns boolean selection of vectors _true and _false according to consition
    // _b. All bits a each component of _b must have the same value (O or
    // 0xffffffff) to ensure portability.
    #[inline]
    pub fn select(_b: SimdInt4, _true: SimdInt4, _false: SimdInt4) -> SimdInt4 {
        unsafe {
            return SimdInt4::new(ozz_sse_select_i!(_b.data, _true.data, _false.data));
        }
    }

    // Returns per element binary and operation of _a and _b.
    // _v[0...127] = _a[0...127] & _b[0...127]
    #[inline]
    pub fn and(&self, _b: SimdInt4) -> SimdInt4 {
        unsafe {
            return SimdInt4::new(_mm_and_si128(self.data, _b.data));
        }
    }

    // Returns per element binary and operation of _a and ~_b.
    // _v[0...127] = _a[0...127] & ~_b[0...127]
    #[inline]
    pub fn and_not(&self, _b: SimdInt4) -> SimdInt4 {
        unsafe {
            return SimdInt4::new(_mm_andnot_si128(_b.data, self.data));
        }
    }

    // Returns per element binary or operation of _a and _b.
    // _v[0...127] = _a[0...127] | _b[0...127]
    #[inline]
    pub fn or(&self, _b: SimdInt4) -> SimdInt4 {
        unsafe {
            return SimdInt4::new(_mm_or_si128(self.data, _b.data));
        }
    }

    // Returns per element binary logical xor operation of _a and _b.
    // _v[0...127] = _a[0...127] ^ _b[0...127]
    #[inline]
    pub fn xor(&self, _b: SimdInt4) -> SimdInt4 {
        unsafe {
            return SimdInt4::new(_mm_xor_si128(self.data, _b.data));
        }
    }

    // Returns per element binary complement of _v.
    // _v[0...127] = ~_b[0...127]
    #[inline]
    pub fn not(&self) -> SimdInt4 {
        unsafe {
            return SimdInt4::new(_mm_xor_si128(self.data, _mm_cmpeq_epi32(self.data, self.data)));
        }
    }

    // Shifts the 4 signed or unsigned 32-bit integers in a left by count _bits
    // while shifting in zeros.
    #[inline]
    pub fn shift_l<const BITS: i32>(&self) -> SimdInt4 {
        unsafe {
            return SimdInt4::new(_mm_slli_epi32(self.data, BITS));
        }
    }

    // Shifts the 4 signed 32-bit integers in a right by count bits while shifting
    // in the sign bit.
    #[inline]
    pub fn shift_r<const BITS: i32>(&self) -> SimdInt4 {
        unsafe {
            return SimdInt4::new(_mm_srai_epi32(self.data, BITS));
        }
    }

    // Shifts the 4 signed or unsigned 32-bit integers in a right by count bits
    // while shifting in zeros.
    #[inline]
    pub fn shift_ru<const BITS: i32>(&self) -> SimdInt4 {
        unsafe {
            return SimdInt4::new(_mm_srli_epi32(self.data, BITS));
        }
    }

    // Per element "equal" comparison of _a and _b.
    #[inline]
    pub fn cmp_eq(&self, _b: SimdInt4) -> SimdInt4 {
        unsafe {
            return SimdInt4::new(_mm_cmpeq_epi32(self.data, _b.data));
        }
    }

    // Per element "not equal" comparison of _a and _b.
    #[inline]
    pub fn cmp_ne(&self, _b: SimdInt4) -> SimdInt4 {
        unsafe {
            let eq = _mm_cmpeq_epi32(self.data, _b.data);
            return SimdInt4::new(_mm_xor_si128(eq, _mm_cmpeq_epi32(self.data, self.data)));
        }
    }

    // Per element "less than" comparison of _a and _b.
    #[inline]
    pub fn cmp_lt(&self, _b: SimdInt4) -> SimdInt4 {
        unsafe {
            return SimdInt4::new(_mm_cmpgt_epi32(_b.data, self.data));
        }
    }

    // Per element "less than or equal" comparison of _a and _b.
    #[inline]
    pub fn cmp_le(&self, _b: SimdInt4) -> SimdInt4 {
        unsafe {
            let gt = _mm_cmpgt_epi32(self.data, _b.data);
            return SimdInt4::new(_mm_xor_si128(gt, _mm_cmpeq_epi32(self.data, self.data)));
        }
    }

    // Per element "greater than" comparison of _a and _b.
    #[inline]
    pub fn cmp_gt(&self, _b: SimdInt4) -> SimdInt4 {
        unsafe {
            return SimdInt4::new(_mm_cmpgt_epi32(self.data, _b.data));
        }
    }

    // Per element "greater than or equal" comparison of _a and _b.
    #[inline]
    pub fn cmp_ge(&self, _b: SimdInt4) -> SimdInt4 {
        unsafe {
            let lt = _mm_cmpgt_epi32(_b.data, self.data);
            return SimdInt4::new(_mm_xor_si128(lt, _mm_cmpeq_epi32(self.data, self.data)));
        }
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
        unsafe {
            let zero = _mm_setzero_si128();
            let ffff = _mm_cmpeq_epi32(zero, zero);
            let one = _mm_srli_epi32(_mm_slli_epi32(ffff, 25), 2);
            let x = _mm_srli_si128(one, 12);
            return Float4x4 {
                cols: [SimdFloat4::new(_mm_castsi128_ps(x)),
                    SimdFloat4::new(_mm_castsi128_ps(_mm_slli_si128(x, 4))),
                    SimdFloat4::new(_mm_castsi128_ps(_mm_slli_si128(x, 8))),
                    SimdFloat4::new(_mm_castsi128_ps(_mm_slli_si128(one, 12)))]
            };
        }
    }

    // Returns a translation matrix.
    // _v.w is ignored.
    #[inline]
    pub fn translation(_v: SimdFloat4) -> Float4x4 {
        unsafe {
            let zero = _mm_setzero_si128();
            let ffff = _mm_cmpeq_epi32(zero, zero);
            let mask000f = _mm_slli_si128(ffff, 12);
            let one = _mm_srli_epi32(_mm_slli_epi32(ffff, 25), 2);
            let x = _mm_srli_si128(one, 12);
            return Float4x4 {
                cols:
                [SimdFloat4::new(_mm_castsi128_ps(x)), SimdFloat4::new(_mm_castsi128_ps(_mm_slli_si128(x, 4))),
                    SimdFloat4::new(_mm_castsi128_ps(_mm_slli_si128(x, 8))),
                    SimdFloat4::new(ozz_sse_select_f!(mask000f, _mm_castsi128_ps(one), _v.data))]
            };
        }
    }

    // Returns a scaling matrix that scales along _v.
    // _v.w is ignored.
    #[inline]
    pub fn scaling(_v: SimdFloat4) -> Float4x4 {
        unsafe {
            let zero = _mm_setzero_si128();
            let ffff = _mm_cmpeq_epi32(zero, zero);
            let if000 = _mm_srli_si128(ffff, 12);
            let ione = _mm_srli_epi32(_mm_slli_epi32(ffff, 25), 2);
            return Float4x4 {
                cols:
                [SimdFloat4::new(_mm_and_ps(_v.data, _mm_castsi128_ps(if000))),
                    SimdFloat4::new(_mm_and_ps(_v.data, _mm_castsi128_ps(_mm_slli_si128(if000, 4)))),
                    SimdFloat4::new(_mm_and_ps(_v.data, _mm_castsi128_ps(_mm_slli_si128(if000, 8)))),
                    SimdFloat4::new(_mm_castsi128_ps(_mm_slli_si128(ione, 12)))]
            };
        }
    }

    // Returns the rotation matrix built from Euler angles defined by x, y and z
    // components of _v. Euler angles are ordered Heading, Elevation and Bank, or
    // Yaw, Pitch and Roll. _v.w is ignored.
    #[inline]
    pub fn from_euler(_v: SimdFloat4) -> Float4x4 {
        let cos = _v.cos();
        let sin = _v.sin();

        let cx = cos.get_x();
        let sx = sin.get_x();
        let cy = cos.get_y();
        let sy = sin.get_y();
        let cz = cos.get_z();
        let sz = sin.get_z();

        let sycz = sy * cz;
        let sysz = sy * sz;

        return Float4x4 {
            cols: [SimdFloat4::load(cx * cy, sx * sz - cx * sycz,
                                    cx * sysz + sx * cz, 0.0),
                SimdFloat4::load(sy, cy * cz, -cy * sz, 0.0),
                SimdFloat4::load(-sx * cy, sx * sycz + cx * sz,
                                 -sx * sysz + cx * cz, 0.0),
                SimdFloat4::w_axis()]
        };
    }

    // Returns the rotation matrix built from axis defined by _axis.xyz and
    // _angle.x
    #[inline]
    pub fn from_axis_angle(_axis: SimdFloat4, _angle: SimdFloat4) -> Float4x4 {
        unsafe {
            debug_assert!(SimdInt4::are_all_true1(&_axis.is_normalized_est3()));

            let zero = _mm_setzero_si128();
            let ffff = _mm_cmpeq_epi32(zero, zero);
            let ione = _mm_srli_epi32(_mm_slli_epi32(ffff, 25), 2);
            let fff0 = _mm_castsi128_ps(_mm_srli_si128(ffff, 4));
            let one = _mm_castsi128_ps(ione);
            let w_axis = _mm_castsi128_ps(_mm_slli_si128(ione, 12));

            let sin = SimdFloat4::splat_x(&SimdFloat4::sin_x(&_angle));
            let cos = SimdFloat4::splat_x(&SimdFloat4::cos_x(&_angle));
            let one_minus_cos = _mm_sub_ps(one, cos.data);

            let v0 =
                _mm_mul_ps(_mm_mul_ps(one_minus_cos,
                                      ozz_shuffle_ps1!(_axis.data, _mm_shuffle!(3, 0, 2, 1))),
                           ozz_shuffle_ps1!(_axis.data, _mm_shuffle!(3, 1, 0, 2)));
            let r0 =
                _mm_add_ps(_mm_mul_ps(_mm_mul_ps(one_minus_cos, _axis.data), _axis.data), cos.data);
            let r1 = _mm_add_ps(_mm_mul_ps(sin.data, _axis.data), v0);
            let r2 = _mm_sub_ps(v0, _mm_mul_ps(sin.data, _axis.data));
            let r0fff0 = _mm_and_ps(r0, fff0);
            let r1r22120 = _mm_shuffle_ps(r1, r2, _mm_shuffle!(2, 1, 2, 0));
            let v1 = ozz_shuffle_ps1!(r1r22120, _mm_shuffle!(0, 3, 2, 1));
            let r1r20011 = _mm_shuffle_ps(r1, r2, _mm_shuffle!(0, 0, 1, 1));
            let v2 = ozz_shuffle_ps1!(r1r20011, _mm_shuffle!(2, 0, 2, 0));

            let t0 = _mm_shuffle_ps(r0fff0, v1, _mm_shuffle!(1, 0, 3, 0));
            let t1 = _mm_shuffle_ps(r0fff0, v1, _mm_shuffle!(3, 2, 3, 1));
            return Float4x4 {
                cols: [SimdFloat4::new(ozz_shuffle_ps1!(t0, _mm_shuffle!(1, 3, 2, 0))),
                    SimdFloat4::new(ozz_shuffle_ps1!(t1, _mm_shuffle!(1, 3, 0, 2))),
                    SimdFloat4::new(_mm_shuffle_ps(v2, r0fff0, _mm_shuffle!(3, 2, 1, 0))),
                    SimdFloat4::new(w_axis)]
            };
        }
    }

    // Returns the rotation matrix built from quaternion defined by x, y, z and w
    // components of _v.
    #[inline]
    pub fn from_quaternion(_quaternion: SimdFloat4) -> Float4x4 {
        unsafe {
            debug_assert!(SimdInt4::are_all_true1(&_quaternion.is_normalized_est4()));

            let zero = _mm_setzero_si128();
            let ffff = _mm_cmpeq_epi32(zero, zero);
            let ione = _mm_srli_epi32(_mm_slli_epi32(ffff, 25), 2);
            let fff0 = _mm_castsi128_ps(_mm_srli_si128(ffff, 4));
            let c1110 = _mm_castsi128_ps(_mm_srli_si128(ione, 4));
            let w_axis = _mm_castsi128_ps(_mm_slli_si128(ione, 12));

            let vsum = _mm_add_ps(_quaternion.data, _quaternion.data);
            let vms = _mm_mul_ps(_quaternion.data, vsum);

            let r0 = _mm_sub_ps(
                _mm_sub_ps(
                    c1110,
                    _mm_and_ps(ozz_shuffle_ps1!(vms, _mm_shuffle!(3, 0, 0, 1)), fff0)),
                _mm_and_ps(ozz_shuffle_ps1!(vms, _mm_shuffle!(3, 1, 2, 2)), fff0));
            let v0 =
                _mm_mul_ps(ozz_shuffle_ps1!(_quaternion.data, _mm_shuffle!(3, 1, 0, 0)),
                           ozz_shuffle_ps1!(vsum, _mm_shuffle!(3, 2, 1, 2)));
            let v1 =
                _mm_mul_ps(ozz_shuffle_ps1!(_quaternion.data, _mm_shuffle!(3, 3, 3, 3)),
                           ozz_shuffle_ps1!(vsum, _mm_shuffle!(3, 0, 2, 1)));

            let r1 = _mm_add_ps(v0, v1);
            let r2 = _mm_sub_ps(v0, v1);

            let r1r21021 = _mm_shuffle_ps(r1, r2, _mm_shuffle!(1, 0, 2, 1));
            let v2 = ozz_shuffle_ps1!(r1r21021, _mm_shuffle!(1, 3, 2, 0));
            let r1r22200 = _mm_shuffle_ps(r1, r2, _mm_shuffle!(2, 2, 0, 0));
            let v3 = ozz_shuffle_ps1!(r1r22200, _mm_shuffle!(2, 0, 2, 0));

            let q0 = _mm_shuffle_ps(r0, v2, _mm_shuffle!(1, 0, 3, 0));
            let q1 = _mm_shuffle_ps(r0, v2, _mm_shuffle!(3, 2, 3, 1));
            return Float4x4 {
                cols: [SimdFloat4::new(ozz_shuffle_ps1!(q0, _mm_shuffle!(1, 3, 2, 0))),
                    SimdFloat4::new(ozz_shuffle_ps1!(q1, _mm_shuffle!(1, 3, 0, 2))),
                    SimdFloat4::new(_mm_shuffle_ps(v3, r0, _mm_shuffle!(3, 2, 1, 0))),
                    SimdFloat4::new(w_axis)]
            };
        }
    }

    // Returns the affine transformation matrix built from split translation,
    // rotation (quaternion) and scale.
    #[inline]
    pub fn from_affine(_translation: SimdFloat4,
                       _quaternion: SimdFloat4,
                       _scale: SimdFloat4) -> Float4x4 {
        unsafe {
            debug_assert!(SimdInt4::are_all_true1(&_quaternion.is_normalized_est4()));

            let zero = _mm_setzero_si128();
            let ffff = _mm_cmpeq_epi32(zero, zero);
            let ione = _mm_srli_epi32(_mm_slli_epi32(ffff, 25), 2);
            let fff0 = _mm_castsi128_ps(_mm_srli_si128(ffff, 4));
            let c1110 = _mm_castsi128_ps(_mm_srli_si128(ione, 4));

            let vsum = _mm_add_ps(_quaternion.data, _quaternion.data);
            let vms = _mm_mul_ps(_quaternion.data, vsum);

            let r0 = _mm_sub_ps(
                _mm_sub_ps(
                    c1110,
                    _mm_and_ps(ozz_shuffle_ps1!(vms, _mm_shuffle!(3, 0, 0, 1)), fff0)),
                _mm_and_ps(ozz_shuffle_ps1!(vms, _mm_shuffle!(3, 1, 2, 2)), fff0));
            let v0 =
                _mm_mul_ps(ozz_shuffle_ps1!(_quaternion.data, _mm_shuffle!(3, 1, 0, 0)),
                           ozz_shuffle_ps1!(vsum, _mm_shuffle!(3, 2, 1, 2)));
            let v1 =
                _mm_mul_ps(ozz_shuffle_ps1!(_quaternion.data, _mm_shuffle!(3, 3, 3, 3)),
                           ozz_shuffle_ps1!(vsum, _mm_shuffle!(3, 0, 2, 1)));

            let r1 = _mm_add_ps(v0, v1);
            let r2 = _mm_sub_ps(v0, v1);

            let r1r21021 = _mm_shuffle_ps(r1, r2, _mm_shuffle!(1, 0, 2, 1));
            let v2 = ozz_shuffle_ps1!(r1r21021, _mm_shuffle!(1, 3, 2, 0));
            let r1r22200 = _mm_shuffle_ps(r1, r2, _mm_shuffle!(2, 2, 0, 0));
            let v3 = ozz_shuffle_ps1!(r1r22200, _mm_shuffle!(2, 0, 2, 0));

            let q0 = _mm_shuffle_ps(r0, v2, _mm_shuffle!(1, 0, 3, 0));
            let q1 = _mm_shuffle_ps(r0, v2, _mm_shuffle!(3, 2, 3, 1));

            return Float4x4 {
                cols:
                [SimdFloat4::new(_mm_mul_ps(ozz_shuffle_ps1!(q0, _mm_shuffle!(1, 3, 2, 0)),
                                            ozz_sse_splat_f!(_scale.data, 0))),
                    SimdFloat4::new(_mm_mul_ps(ozz_shuffle_ps1!(q1, _mm_shuffle!(1, 3, 0, 2)),
                                               ozz_sse_splat_f!(_scale.data, 1))),
                    SimdFloat4::new(_mm_mul_ps(_mm_shuffle_ps(v3, r0, _mm_shuffle!(3, 2, 1, 0)),
                                               ozz_sse_splat_f!(_scale.data, 2))),
                    SimdFloat4::new(_mm_movelh_ps(_translation.data, _mm_unpackhi_ps(_translation.data, c1110)))]
            };
        }
    }

    // Returns the transpose of matrix _m.
    #[inline]
    pub fn transpose(_m: &Float4x4) -> Float4x4 {
        unsafe {
            let tmp0 = _mm_unpacklo_ps(_m.cols[0].data, _m.cols[2].data);
            let tmp1 = _mm_unpacklo_ps(_m.cols[1].data, _m.cols[3].data);
            let tmp2 = _mm_unpackhi_ps(_m.cols[0].data, _m.cols[2].data);
            let tmp3 = _mm_unpackhi_ps(_m.cols[1].data, _m.cols[3].data);
            return Float4x4 {
                cols:
                [SimdFloat4::new(_mm_unpacklo_ps(tmp0, tmp1)), SimdFloat4::new(_mm_unpackhi_ps(tmp0, tmp1)),
                    SimdFloat4::new(_mm_unpacklo_ps(tmp2, tmp3)), SimdFloat4::new(_mm_unpackhi_ps(tmp2, tmp3))]
            };
        }
    }

// // Returns the inverse of matrix _m.
// // If _invertible is not nullptr, its x component will be set to true if matrix is
// // invertible. If _invertible is nullptr, then an assert is triggered in case the
// // matrix isn't invertible.
// #[inline]
// pub fn invert(_m: &Float4x4, mut _invertible: Option<SimdInt4>) -> Float4x4 {
//     unsafe {
//         let _t0 =
//             _mm_shuffle_ps(_m.cols[0], _m.cols[1], _mm_shuffle!(1, 0, 1, 0));
//         let _t1 =
//             _mm_shuffle_ps(_m.cols[2], _m.cols[3], _mm_shuffle!(1, 0, 1, 0));
//         let _t2 =
//             _mm_shuffle_ps(_m.cols[0], _m.cols[1], _mm_shuffle!(3, 2, 3, 2));
//         let _t3 =
//             _mm_shuffle_ps(_m.cols[2], _m.cols[3], _mm_shuffle!(3, 2, 3, 2));
//         let c0 = _mm_shuffle_ps(_t0, _t1, _mm_shuffle!(2, 0, 2, 0));
//         let c1 = _mm_shuffle_ps(_t1, _t0, _mm_shuffle!(3, 1, 3, 1));
//         let c2 = _mm_shuffle_ps(_t2, _t3, _mm_shuffle!(2, 0, 2, 0));
//         let c3 = _mm_shuffle_ps(_t3, _t2, _mm_shuffle!(3, 1, 3, 1));
//
//         let mut tmp1 = _mm_mul_ps(c2, c3);
//         tmp1 = ozz_shuffle_ps1!(tmp1, 0xB1);
//         let mut minor0 = _mm_mul_ps(c1, tmp1);
//         let mut minor1 = _mm_mul_ps(c0, tmp1);
//         tmp1 = ozz_shuffle_ps1!(tmp1, 0x4E);
//         minor0 = ozz_msub!(c1, tmp1, minor0);
//         minor1 = ozz_msub!(c0, tmp1, minor1);
//         minor1 = ozz_shuffle_ps1!(minor1, 0x4E);
//
//         tmp1 = _mm_mul_ps(c1, c2);
//         tmp1 = ozz_shuffle_ps1!(tmp1, 0xB1);
//         minor0 = ozz_madd!(c3, tmp1, minor0);
//         let mut minor3 = _mm_mul_ps(c0, tmp1);
//         tmp1 = ozz_shuffle_ps1!(tmp1, 0x4E);
//         minor0 = ozz_nmadd!(c3, tmp1, minor0);
//         minor3 = ozz_msub!(c0, tmp1, minor3);
//         minor3 = ozz_shuffle_ps1!(minor3, 0x4E);
//
//         tmp1 = _mm_mul_ps(ozz_shuffle_ps1!(c1, 0x4E), c3);
//         tmp1 = ozz_shuffle_ps1!(tmp1, 0xB1);
//         let tmp2 = ozz_shuffle_ps1!(c2, 0x4E);
//         minor0 = ozz_madd!(tmp2, tmp1, minor0);
//         let mut minor2 = _mm_mul_ps(c0, tmp1);
//         tmp1 = ozz_shuffle_ps1!(tmp1, 0x4E);
//         minor0 = ozz_nmadd!(tmp2, tmp1, minor0);
//         minor2 = ozz_msub!(c0, tmp1, minor2);
//         minor2 = ozz_shuffle_ps1!(minor2, 0x4E);
//
//         tmp1 = _mm_mul_ps(c0, c1);
//         tmp1 = ozz_shuffle_ps1!(tmp1, 0xB1);
//         minor2 = ozz_madd!(c3, tmp1, minor2);
//         minor3 = ozz_msub!(tmp2, tmp1, minor3);
//         tmp1 = ozz_shuffle_ps1!(tmp1, 0x4E);
//         minor2 = ozz_msub!(c3, tmp1, minor2);
//         minor3 = ozz_nmadd!(tmp2, tmp1, minor3);
//
//         tmp1 = _mm_mul_ps(c0, c3);
//         tmp1 = ozz_shuffle_ps1!(tmp1, 0xB1);
//         minor1 = ozz_nmadd!(tmp2, tmp1, minor1);
//         minor2 = ozz_madd!(c1, tmp1, minor2);
//         tmp1 = ozz_shuffle_ps1!(tmp1, 0x4E);
//         minor1 = ozz_madd!(tmp2, tmp1, minor1);
//         minor2 = ozz_nmadd!(c1, tmp1, minor2);
//
//         tmp1 = _mm_mul_ps(c0, tmp2);
//         tmp1 = ozz_shuffle_ps1!(tmp1, 0xB1);
//         minor1 = ozz_madd!(c3, tmp1, minor1);
//         minor3 = ozz_nmadd!(c1, tmp1, minor3);
//         tmp1 = ozz_shuffle_ps1!(tmp1, 0x4E);
//         minor1 = ozz_nmadd!(c3, tmp1, minor1);
//         minor3 = ozz_madd!(c1, tmp1, minor3);
//
//         let mut det = _mm_mul_ps(c0, minor0);
//         det = _mm_add_ps(ozz_shuffle_ps1!(det, 0x4E), det);
//         det = _mm_add_ss(ozz_shuffle_ps1!(det, 0xB1), det);
//         let invertible = simd_float4::cmp_ne(det, simd_float4::zero());
//         debug_assert!((_invertible.is_none() || simd_int4::are_all_true1(invertible)) &&
//             "Matrix is not invertible".parse().unwrap());
//         if _invertible.is_some() {
//             _invertible = Some(invertible);
//         }
//         tmp1 = ozz_sse_select_f!(invertible, simd_float4::rcp_est_nr(det), simd_float4::zero());
//         det = ozz_nmaddx!(det, _mm_mul_ss(tmp1, tmp1), _mm_add_ss(tmp1, tmp1));
//         det = ozz_shuffle_ps1!(det, 0x00);
//
//         // Copy the final columns
//         return Float4x4 {
//             cols: [_mm_mul_ps(det, minor0), _mm_mul_ps(det, minor1),
//                 _mm_mul_ps(det, minor2), _mm_mul_ps(det, minor3)]
//         };
//     }
// }
//
// // Translates matrix _m along the axis defined by _v components.
// // _v.w is ignored.
// #[inline]
// pub fn translate(_m: &Float4x4, _v: SimdFloat4) -> Float4x4 {
//     unsafe {
//         let a01 = ozz_madd!(_m.cols[0], ozz_sse_splat_f!(_v, 0),
//                                     _mm_mul_ps(_m.cols[1], ozz_sse_splat_f!(_v, 1)));
//         let m3 = ozz_madd!(_m.cols[2], ozz_sse_splat_f!(_v, 2), _m.cols[3]);
//         return Float4x4 {
//             cols:
//             [_m.cols[0], _m.cols[1], _m.cols[2], _mm_add_ps(a01, m3)]
//         };
//     }
// }
//
// // Scales matrix _m along each axis with x, y, z components of _v.
// // _v.w is ignored.
// #[inline]
// pub fn scale(_m: &Float4x4, _v: SimdFloat4) -> Float4x4 {
//     unsafe {
//         return Float4x4 {
//             cols: [_mm_mul_ps(_m.cols[0], ozz_sse_splat_f!(_v, 0)),
//                 _mm_mul_ps(_m.cols[1], ozz_sse_splat_f!(_v, 1)),
//                 _mm_mul_ps(_m.cols[2], ozz_sse_splat_f!(_v, 2)),
//                 _m.cols[3]]
//         };
//     }
// }
//
// // Multiply each column of matrix _m with vector _v.
// #[inline]
// pub fn column_multiply(_m: &Float4x4, _v: SimdFloat4) -> Float4x4 {
//     unsafe {
//         return Float4x4 {
//             cols: [_mm_mul_ps(_m.cols[0], _v), _mm_mul_ps(_m.cols[1], _v),
//                 _mm_mul_ps(_m.cols[2], _v),
//                 _mm_mul_ps(_m.cols[3], _v)]
//         };
//     }
// }
//
// // Tests if each 3 column of upper 3x3 matrix of _m is a normal matrix.
// // Returns the result in the x, y and z component of the returned vector. w is
// // set to 0.
// #[inline]
// pub fn is_normalized(_m: &Float4x4) -> SimdInt4 {
//     unsafe {
//         let max = _mm_set_ps1(1.0 + crate::math_constant::K_NORMALIZATION_TOLERANCE_SQ);
//         let min = _mm_set_ps1(1.0 - crate::math_constant::K_NORMALIZATION_TOLERANCE_SQ);
//
//         let tmp0 = _mm_unpacklo_ps(_m.cols[0], _m.cols[2]);
//         let tmp1 = _mm_unpacklo_ps(_m.cols[1], _m.cols[3]);
//         let tmp2 = _mm_unpackhi_ps(_m.cols[0], _m.cols[2]);
//         let tmp3 = _mm_unpackhi_ps(_m.cols[1], _m.cols[3]);
//         let row0 = _mm_unpacklo_ps(tmp0, tmp1);
//         let row1 = _mm_unpackhi_ps(tmp0, tmp1);
//         let row2 = _mm_unpacklo_ps(tmp2, tmp3);
//
//         let dot =
//             ozz_madd!(row0, row0, ozz_madd!(row1, row1, _mm_mul_ps(row2, row2)));
//         let normalized =
//             _mm_and_ps(_mm_cmplt_ps(dot, max), _mm_cmpgt_ps(dot, min));
//         return Float4x4::new(_mm_castps_si128(
//             _mm_and_ps(normalized, _mm_castsi128_ps(simd_int4::mask_fff0())));
//     }
// }
//
// // Tests if each 3 column of upper 3x3 matrix of _m is a normal matrix.
// // Uses the estimated tolerance
// // Returns the result in the x, y and z component of the returned vector. w is
// // set to 0.
// #[inline]
// pub fn is_normalized_est(_m: &Float4x4) -> SimdInt4 {
//     unsafe {
//         let max = _mm_set_ps1(1.0 + crate::math_constant::K_NORMALIZATION_TOLERANCE_EST_SQ);
//         let min = _mm_set_ps1(1.0 - crate::math_constant::K_NORMALIZATION_TOLERANCE_EST_SQ);
//
//         let tmp0 = _mm_unpacklo_ps(_m.cols[0], _m.cols[2]);
//         let tmp1 = _mm_unpacklo_ps(_m.cols[1], _m.cols[3]);
//         let tmp2 = _mm_unpackhi_ps(_m.cols[0], _m.cols[2]);
//         let tmp3 = _mm_unpackhi_ps(_m.cols[1], _m.cols[3]);
//         let row0 = _mm_unpacklo_ps(tmp0, tmp1);
//         let row1 = _mm_unpackhi_ps(tmp0, tmp1);
//         let row2 = _mm_unpacklo_ps(tmp2, tmp3);
//
//         let dot =
//             ozz_madd!(row0, row0, ozz_madd!(row1, row1, _mm_mul_ps(row2, row2)));
//
//         let normalized =
//             _mm_and_ps(_mm_cmplt_ps(dot, max), _mm_cmpgt_ps(dot, min));
//
//         return Float4x4::new(_mm_castps_si128(
//             _mm_and_ps(normalized, _mm_castsi128_ps(simd_int4::mask_fff0())));
//     }
// }
//
// // Tests if the upper 3x3 matrix of _m is an orthogonal matrix.
// // A matrix that contains a reflexion cannot be considered orthogonal.
// // Returns the result in the x component of the returned vector. y, z and w are
// // set to 0.
// #[inline]
// pub fn is_orthogonal(_m: &Float4x4) -> SimdInt4 {
//     unsafe {
//         let max = _mm_set_ss(1.0 + crate::math_constant::K_NORMALIZATION_TOLERANCE_SQ);
//         let min = _mm_set_ss(1.0 - crate::math_constant::K_NORMALIZATION_TOLERANCE_SQ);
//         let zero = _mm_setzero_ps();
//
//         // Use simd_float4::zero() if one of the normalization fails. _m will then be
//         // considered not orthogonal.
//         let cross = simd_float4::normalize_safe3(simd_float4::cross3(_m.cols[0], _m.cols[1]), zero);
//         let at = simd_float4::normalize_safe3(_m.cols[2], zero);
//
//         let dot;
//         ozz_sse_dot3_f!(cross, at, dot);
//         let dotx000 = _mm_move_ss(zero, dot);
//         return Float4x4::new(_mm_castps_si128(
//             _mm_and_ps(_mm_cmplt_ss(dotx000, max), _mm_cmpgt_ss(dotx000, min)));
//     }
// }
//
// // Returns the quaternion that represent the rotation of matrix _m.
// // _m must be normalized and orthogonal.
// // the return SimdFloat4::new(quaternion is normalized.
// #[inline]
// pub fn to_quaternion(_m: &Float4x4) -> SimdFloat4 {
//     unsafe {
//         debug_assert!(simd_int4::are_all_true3(is_normalized_est(_m)));
//         debug_assert!(simd_int4::are_all_true1(is_orthogonal(_m)));
//
//         // Prepares constants.
//         let zero = _mm_setzero_si128();
//         let ffff = _mm_cmpeq_epi32(zero, zero);
//         let half = _mm_set1_ps(0.50);
//         let mask_f000 = _mm_srli_si128(ffff, 12);
//         let mask_000f = _mm_slli_si128(ffff, 12);
//         let one =
//             _mm_castsi128_ps(_mm_srli_epi32(_mm_slli_epi32(ffff, 25), 2));
//         let mask_0f00 = _mm_slli_si128(mask_f000, 4);
//         let mask_00f0 = _mm_slli_si128(mask_f000, 8);
//
//         let xx_yy = ozz_sse_select_f!(mask_0f00, _m.cols[1], _m.cols[0]);
//         let xx_yy_0010 = ozz_shuffle_ps1!(xx_yy, _mm_shuffle!(0, 0, 1, 0));
//         let xx_yy_zz_xx =
//             ozz_sse_select_f!(mask_00f0, _m.cols[2], xx_yy_0010);
//         let yy_zz_xx_yy =
//             ozz_shuffle_ps1!(xx_yy_zz_xx, _mm_shuffle!(1, 0, 2, 1));
//         let zz_xx_yy_zz =
//             ozz_shuffle_ps1!(xx_yy_zz_xx, _mm_shuffle!(2, 1, 0, 2));
//
//         let diag_sum =
//             _mm_add_ps(_mm_add_ps(xx_yy_zz_xx, yy_zz_xx_yy), zz_xx_yy_zz);
//         let diag_diff =
//             _mm_sub_ps(_mm_sub_ps(xx_yy_zz_xx, yy_zz_xx_yy), zz_xx_yy_zz);
//         let radicand =
//             _mm_add_ps(ozz_sse_select_f!(mask_000f, diag_sum, diag_diff), one);
//         let inv_sqrt = _mm_div_ps(one, _mm_sqrt_ps(radicand));
//
//         let mut zy_xz_yx = ozz_sse_select_f!(mask_00f0, _m.cols[1], _m.cols[0]);
//         zy_xz_yx = ozz_shuffle_ps1!(zy_xz_yx, _mm_shuffle!(0, 1, 2, 2));
//         zy_xz_yx =
//             ozz_sse_select_f!(mask_0f00, ozz_sse_splat_f!(_m.cols[2], 0), zy_xz_yx);
//         let mut yz_zx_xy = ozz_sse_select_f!(mask_f000, _m.cols[1], _m.cols[0]);
//         yz_zx_xy = ozz_shuffle_ps1!(yz_zx_xy, _mm_shuffle!(0, 0, 2, 0));
//         yz_zx_xy =
//             ozz_sse_select_f!(mask_f000, ozz_sse_splat_f!(_m.cols[2], 1), yz_zx_xy);
//         let sum = _mm_add_ps(zy_xz_yx, yz_zx_xy);
//         let diff = _mm_sub_ps(zy_xz_yx, yz_zx_xy);
//         let scale = _mm_mul_ps(inv_sqrt, half);
//
//         let sum0 = ozz_shuffle_ps1!(sum, _mm_shuffle!(0, 1, 2, 0));
//         let sum1 = ozz_shuffle_ps1!(sum, _mm_shuffle!(0, 0, 0, 2));
//         let sum2 = ozz_shuffle_ps1!(sum, _mm_shuffle!(0, 0, 0, 1));
//         let mut res0 = ozz_sse_select_f!(mask_000f, ozz_sse_splat_f!(diff, 0), sum0);
//         let mut res1 = ozz_sse_select_f!(mask_000f, ozz_sse_splat_f!(diff, 1), sum1);
//         let mut res2 = ozz_sse_select_f!(mask_000f, ozz_sse_splat_f!(diff, 2), sum2);
//         res0 = _mm_mul_ps(ozz_sse_select_f!(mask_f000, radicand, res0),
//                           ozz_sse_splat_f!(scale, 0));
//         res1 = _mm_mul_ps(ozz_sse_select_f!(mask_0f00, radicand, res1),
//                           ozz_sse_splat_f!(scale, 1));
//         res2 = _mm_mul_ps(ozz_sse_select_f!(mask_00f0, radicand, res2),
//                           ozz_sse_splat_f!(scale, 2));
//         let res3 = _mm_mul_ps(ozz_sse_select_f!(mask_000f, radicand, diff),
//                               ozz_sse_splat_f!(scale, 3));
//
//         let xx = ozz_sse_splat_f!(_m.cols[0], 0);
//         let yy = ozz_sse_splat_f!(_m.cols[1], 1);
//         let zz = ozz_sse_splat_f!(_m.cols[2], 2);
//         let cond0 = _mm_castps_si128(_mm_cmpgt_ps(yy, xx));
//         let cond1 =
//             _mm_castps_si128(_mm_and_ps(_mm_cmpgt_ps(zz, xx), _mm_cmpgt_ps(zz, yy)));
//         let cond2 = _mm_castps_si128(
//             _mm_cmpgt_ps(ozz_sse_splat_f!(diag_sum, 0), _mm_castsi128_ps(zero)));
//         let mut res = ozz_sse_select_f!(cond0, res1, res0);
//         res = ozz_sse_select_f!(cond1, res2, res);
//         res = ozz_sse_select_f!(cond2, res3, res);
//
//         debug_assert!(simd_int4::are_all_true1(is_normalized_est4(res)));
//         return res;
//     }
// }
//
// // Decompose a general 3D transformation matrix _m into its scalar, rotational
// // and translational components.
// // Returns false if it was not possible to decompose the matrix. This would be
// // because more than 1 of the 3 first column of _m are scaled to 0.
// #[inline]
// pub fn to_affine(_m: &Float4x4, _translation: &mut SimdFloat4, _quaternion: &mut SimdFloat4, _scale: &mut SimdFloat4) -> bool {
//     unsafe {
//         let zero = _mm_setzero_ps();
//         let one = simd_float4::one();
//         let fff0 = simd_int4::mask_fff0();
//         let max = _mm_set_ps1(crate::math_constant::K_ORTHOGONALISATION_TOLERANCE_SQ);
//         let min = _mm_set_ps1(-crate::math_constant::K_ORTHOGONALISATION_TOLERANCE_SQ);
//
//         // Extracts translation.
//         *_translation = ozz_sse_select_f!(fff0, _m.cols[3], one);
//
//         // Extracts scale.
//         let m_tmp0 = _mm_unpacklo_ps(_m.cols[0], _m.cols[2]);
//         let m_tmp1 = _mm_unpacklo_ps(_m.cols[1], _m.cols[3]);
//         let m_tmp2 = _mm_unpackhi_ps(_m.cols[0], _m.cols[2]);
//         let m_tmp3 = _mm_unpackhi_ps(_m.cols[1], _m.cols[3]);
//         let m_row0 = _mm_unpacklo_ps(m_tmp0, m_tmp1);
//         let m_row1 = _mm_unpackhi_ps(m_tmp0, m_tmp1);
//         let m_row2 = _mm_unpacklo_ps(m_tmp2, m_tmp3);
//
//         let dot = ozz_madd!(
//             m_row0, m_row0, ozz_madd!(m_row1, m_row1, _mm_mul_ps(m_row2, m_row2)));
//         let abs_scale = _mm_sqrt_ps(dot);
//
//         let zero_axis =
//             _mm_and_ps(_mm_cmplt_ps(dot, max), _mm_cmpgt_ps(dot, min));
//
//         // Builds an orthonormal matrix in order to support quaternion extraction.
//         let mut orthonormal = Float4x4::identity();
//         let mask = _mm_movemask_ps(zero_axis);
//         if mask & 1 != 0 {
//             if mask & 6 != 0 {
//                 return SimdFloat4::new(false;
//             }
//             orthonormal.cols[1] = _mm_div_ps(_m.cols[1], ozz_sse_splat_f!(abs_scale, 1));
//             orthonormal.cols[0] = simd_float4::normalize3(simd_float4::cross3(orthonormal.cols[1], _m.cols[2]));
//             orthonormal.cols[2] =
//                 simd_float4::normalize3(simd_float4::cross3(orthonormal.cols[0], orthonormal.cols[1]));
//         } else if mask & 4 != 0 {
//             if mask & 3 != 0 {
//                 return SimdFloat4::new(false;
//             }
//             orthonormal.cols[0] = _mm_div_ps(_m.cols[0], ozz_sse_splat_f!(abs_scale, 0));
//             orthonormal.cols[2] = simd_float4::normalize3(simd_float4::cross3(orthonormal.cols[0], _m.cols[1]));
//             orthonormal.cols[1] =
//                 simd_float4::normalize3(simd_float4::cross3(orthonormal.cols[2], orthonormal.cols[0]));
//         } else {  // Favor z axis in the default case
//             if mask & 5 != 0 {
//                 return SimdFloat4::new(false;
//             }
//             orthonormal.cols[2] = _mm_div_ps(_m.cols[2], ozz_sse_splat_f!(abs_scale, 2));
//             orthonormal.cols[1] = simd_float4::normalize3(simd_float4::cross3(orthonormal.cols[2], _m.cols[0]));
//             orthonormal.cols[0] =
//                 simd_float4::normalize3(simd_float4::cross3(orthonormal.cols[1], orthonormal.cols[2]));
//         }
//         orthonormal.cols[3] = simd_float4::w_axis();
//
//         // Get back scale signs in case of reflexions
//         let o_tmp0 =
//             _mm_unpacklo_ps(orthonormal.cols[0], orthonormal.cols[2]);
//         let o_tmp1 =
//             _mm_unpacklo_ps(orthonormal.cols[1], orthonormal.cols[3]);
//         let o_tmp2 =
//             _mm_unpackhi_ps(orthonormal.cols[0], orthonormal.cols[2]);
//         let o_tmp3 =
//             _mm_unpackhi_ps(orthonormal.cols[1], orthonormal.cols[3]);
//         let o_row0 = _mm_unpacklo_ps(o_tmp0, o_tmp1);
//         let o_row1 = _mm_unpackhi_ps(o_tmp0, o_tmp1);
//         let o_row2 = _mm_unpacklo_ps(o_tmp2, o_tmp3);
//
//         let scale_dot = ozz_madd!(
//             o_row0, m_row0, ozz_madd!(o_row1, m_row1, _mm_mul_ps(o_row2, m_row2)));
//
//         let cond = _mm_castps_si128(_mm_cmpgt_ps(scale_dot, zero));
//         let cfalse = _mm_sub_ps(zero, abs_scale);
//         let scale = ozz_sse_select_f!(cond, abs_scale, cfalse);
//         *_scale = ozz_sse_select_f!(fff0, scale, one);
//
//         // Extracts quaternion.
//         *_quaternion = to_quaternion(&orthonormal);
//         return true;
//     }
// }
//
// // Computes the transformation of a Float4x4 matrix and a point _p.
// // This is equivalent to multiplying a matrix by a SimdFloat4 with a w component
// // of 1.
// #[inline]
// pub fn transform_point(_m: &Float4x4, _v: SimdFloat4) -> SimdFloat4 {
//     unsafe {
//         let xxxx = _mm_mul_ps(ozz_sse_splat_f!(_v, 0), _m.cols[0]);
//         let a23 = ozz_madd!(ozz_sse_splat_f!(_v, 2), _m.cols[2], _m.cols[3]);
//         let a01 = ozz_madd!(ozz_sse_splat_f!(_v, 1), _m.cols[1], xxxx);
//         return SimdFloat4::new(_mm_add_ps(a01, a23));
//     }
// }
//
// // Computes the transformation of a Float4x4 matrix and a vector _v.
// // This is equivalent to multiplying a matrix by a SimdFloat4 with a w component
// // of 0.
// #[inline]
// pub fn transform_vector(_m: &Float4x4, _v: SimdFloat4) -> SimdFloat4 {
//     unsafe {
//         let xxxx = _mm_mul_ps(_m.cols[0], ozz_sse_splat_f!(_v, 0));
//         let zzzz = _mm_mul_ps(_m.cols[1], ozz_sse_splat_f!(_v, 1));
//         let a21 = ozz_madd!(_m.cols[2], ozz_sse_splat_f!(_v, 2), xxxx);
//         return SimdFloat4::new(_mm_add_ps(zzzz, a21));
//     }
// }
//
// // Computes the multiplication of matrix Float4x4 and vector _v.
// impl Mul<SimdFloat4> for Float4x4 {
//     type Output = SimdFloat4;
//     #[inline]
//     fn mul(self, rhs: SimdFloat4) -> Self::Output {
//         unsafe {
//             let xxxx = _mm_mul_ps(ozz_sse_splat_f!(rhs, 0), self.cols[0]);
//             let zzzz = _mm_mul_ps(ozz_sse_splat_f!(rhs, 2), self.cols[2]);
//             let a01 = ozz_madd!(ozz_sse_splat_f!(rhs, 1), self.cols[1], xxxx);
//             let a23 = ozz_madd!(ozz_sse_splat_f!(rhs, 3), self.cols[3], zzzz);
//             return SimdFloat4::new(_mm_add_ps(a01, a23));
//         }
//     }
// }
//
// // Computes the multiplication of two matrices _a and _b.
// impl Mul for Float4x4 {
//     type Output = Float4x4;
//     #[inline]
//     fn mul(self, rhs: Self) -> Self::Output {
//         unsafe {
//             let mut ret = Float4x4::identity();
//             {
//                 let xxxx = _mm_mul_ps(ozz_sse_splat_f!(rhs.cols[0], 0), self.cols[0]);
//                 let zzzz = _mm_mul_ps(ozz_sse_splat_f!(rhs.cols[0], 2), self.cols[2]);
//                 let a01 =
//                     ozz_madd!(ozz_sse_splat_f!(rhs.cols[0], 1), self.cols[1], xxxx);
//                 let a23 =
//                     ozz_madd!(ozz_sse_splat_f!(rhs.cols[0], 3), self.cols[3], zzzz);
//                 ret.cols[0] = _mm_add_ps(a01, a23);
//             }
//             {
//                 let xxxx = _mm_mul_ps(ozz_sse_splat_f!(rhs.cols[1], 0), self.cols[0]);
//                 let zzzz = _mm_mul_ps(ozz_sse_splat_f!(rhs.cols[1], 2), self.cols[2]);
//                 let a01 =
//                     ozz_madd!(ozz_sse_splat_f!(rhs.cols[1], 1), self.cols[1], xxxx);
//                 let a23 =
//                     ozz_madd!(ozz_sse_splat_f!(rhs.cols[1], 3), self.cols[3], zzzz);
//                 ret.cols[1] = _mm_add_ps(a01, a23);
//             }
//             {
//                 let xxxx = _mm_mul_ps(ozz_sse_splat_f!(rhs.cols[2], 0), self.cols[0]);
//                 let zzzz = _mm_mul_ps(ozz_sse_splat_f!(rhs.cols[2], 2), self.cols[2]);
//                 let a01 =
//                     ozz_madd!(ozz_sse_splat_f!(rhs.cols[2], 1), self.cols[1], xxxx);
//                 let a23 =
//                     ozz_madd!(ozz_sse_splat_f!(rhs.cols[2], 3), self.cols[3], zzzz);
//                 ret.cols[2] = _mm_add_ps(a01, a23);
//             }
//             {
//                 let xxxx = _mm_mul_ps(ozz_sse_splat_f!(rhs.cols[3], 0), self.cols[0]);
//                 let zzzz = _mm_mul_ps(ozz_sse_splat_f!(rhs.cols[3], 2), self.cols[2]);
//                 let a01 =
//                     ozz_madd!(ozz_sse_splat_f!(rhs.cols[3], 1), self.cols[1], xxxx);
//                 let a23 =
//                     ozz_madd!(ozz_sse_splat_f!(rhs.cols[3], 3), self.cols[3], zzzz);
//                 ret.cols[3] = _mm_add_ps(a01, a23);
//             }
//             return ret;
//         }
//     }
// }
//
// // Computes the per element addition of two matrices _a and _b.
// impl Add for Float4x4 {
//     type Output = Float4x4;
//     #[inline]
//     fn add(self, rhs: Self) -> Self::Output {
//         unsafe {
//             return Float4x4 {
//                 cols:
//                 [_mm_add_ps(self.cols[0], rhs.cols[0]), _mm_add_ps(self.cols[1], rhs.cols[1]),
//                     _mm_add_ps(self.cols[2], rhs.cols[2]), _mm_add_ps(self.cols[3], rhs.cols[3])]
//             };
//         }
//     }
// }
//
// // Computes the per element subtraction of two matrices _a and _b.
// impl Sub for Float4x4 {
//     type Output = Float4x4;
//     #[inline]
//     fn sub(self, rhs: Self) -> Self::Output {
//         unsafe {
//             return Float4x4 {
//                 cols:
//                 [_mm_sub_ps(self.cols[0], rhs.cols[0]), _mm_sub_ps(self.cols[1], rhs.cols[1]),
//                     _mm_sub_ps(self.cols[2], rhs.cols[2]), _mm_sub_ps(self.cols[3], rhs.cols[3])]
//             };
//         }
//     }
// }
//
// // Converts from a float to a half.
// #[inline]
// pub fn float_to_half(_f: f32) -> u16 {
//     unsafe {
//         let h = _mm_cvtsi128_si32(float_to_half_simd(_mm_set1_ps(_f)));
//         return h as u16;
//     }
// }
//
// // Converts from a half to a float.
// #[inline]
// pub fn half_to_float(_h: u16) -> f32 {
//     unsafe {
//         return _mm_cvtss_f32(half_to_float_simd(_mm_set1_epi32(_h as i32)));
//     }
// }
//
// // Converts from a float to a half.
// #[inline]
// #[allow(overflowing_literals)]
// pub fn float_to_half_simd(_f: SimdFloat4) -> SimdInt4 {
//     unsafe {
//         let mask_sign = _mm_set1_epi32(0x80000000);
//         let mask_round = _mm_set1_epi32(!0xfff);
//         let f32infty = _mm_set1_epi32(255 << 23);
//         let magic = _mm_castsi128_ps(_mm_set1_epi32(15 << 23));
//         let nanbit = _mm_set1_epi32(0x200);
//         let infty_as_fp16 = _mm_set1_epi32(0x7c00);
//         let clamp = _mm_castsi128_ps(_mm_set1_epi32((31 << 23) - 0x1000));
//
//         let msign = _mm_castsi128_ps(mask_sign);
//         let justsign = _mm_and_ps(msign, _f);
//         let absf = _mm_xor_ps(_f, justsign);
//         let mround = _mm_castsi128_ps(mask_round);
//         let absf_int = _mm_castps_si128(absf);
//         let b_isnan = _mm_cmpgt_epi32(absf_int, f32infty);
//         let b_isnormal = _mm_cmpgt_epi32(f32infty, _mm_castps_si128(absf));
//         let inf_or_nan =
//             _mm_or_si128(_mm_and_si128(b_isnan, nanbit), infty_as_fp16);
//         let fnosticky = _mm_and_ps(absf, mround);
//         let scaled = _mm_mul_ps(fnosticky, magic);
//         // Logically, we want PMINSD on "biased", but this should gen better code
//         let clamped = _mm_min_ps(scaled, clamp);
//         let biased =
//             _mm_sub_epi32(_mm_castps_si128(clamped), _mm_castps_si128(mround));
//         let shifted = _mm_srli_epi32(biased, 13);
//         let normal = _mm_and_si128(shifted, b_isnormal);
//         let not_normal = _mm_andnot_si128(b_isnormal, inf_or_nan);
//         let joined = _mm_or_si128(normal, not_normal);
//
//         let sign_shift = _mm_srli_epi32(_mm_castps_si128(justsign), 16);
//         return SimdInt4::new(_mm_or_si128(joined, sign_shift));
//     }
// }
//
// // Converts from a half to a float.
// #[inline]
// pub fn half_to_float_simd(_h: SimdInt4) -> SimdFloat4 {
//     unsafe {
//         let mask_nosign = _mm_set1_epi32(0x7fff);
//         let magic = _mm_castsi128_ps(_mm_set1_epi32((254 - 15) << 23));
//         let was_infnan = _mm_set1_epi32(0x7bff);
//         let exp_infnan = _mm_castsi128_ps(_mm_set1_epi32(255 << 23));
//
//         let expmant = _mm_and_si128(mask_nosign, _h);
//         let shifted = _mm_slli_epi32(expmant, 13);
//         let scaled = _mm_mul_ps(_mm_castsi128_ps(shifted), magic);
//         let b_wasinfnan = _mm_cmpgt_epi32(expmant, was_infnan);
//         let sign = _mm_slli_epi32(_mm_xor_si128(_h, expmant), 16);
//         let infnanexp =
//             _mm_and_ps(_mm_castsi128_ps(b_wasinfnan), exp_infnan);
//         let sign_inf = _mm_or_ps(_mm_castsi128_ps(sign), infnanexp);
//         return SimdFloat4::new(_mm_or_ps(scaled, sign_inf));
//     }
}