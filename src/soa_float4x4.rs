/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::simd_math::*;
use crate::soa_float::{SoaFloat4, SoaFloat3};
use crate::soa_quaternion::*;
use std::ops::{Mul, Add, Sub};

// Declare the 4x4 soa matrix type. Uses the column major convention where the
// matrix-times-vector is written v'=Mv:
// [ m.cols[0].x m.cols[1].x m.cols[2].x m.cols[3].x ]   {v.x}
// | m.cols[0].y m.cols[1].y m.cols[2].y m.cols[3].y | * {v.y}
// | m.cols[0].z m.cols[1].y m.cols[2].y m.cols[3].y |   {v.z}
// [ m.cols[0].w m.cols[1].w m.cols[2].w m.cols[3].w ]   {v.1}
#[derive(Clone)]
pub struct SoaFloat4x4 {
    // Soa matrix columns.
    pub cols: [SoaFloat4; 4],
}

impl SoaFloat4x4 {
    // Returns the identity matrix.
    #[inline]
    pub fn identity() -> SoaFloat4x4 {
        let zero = SimdFloat4::load(0.0, 0.0, 0.0, 0.0);
        let one = SimdFloat4::load(1.0, 1.0, 1.0, 1.0);
        return SoaFloat4x4 {
            cols: [SoaFloat4::load(one, zero, zero, zero),
                SoaFloat4::load(zero, one, zero, zero),
                SoaFloat4::load(zero, zero, one, zero),
                SoaFloat4::load(zero, zero, zero, one)]
        };
    }

    // Returns a scaling matrix that scales along _v.
    // _v.w is ignored.
    #[inline]
    pub fn scaling(_v: &SoaFloat4) -> SoaFloat4x4 {
        let zero = SimdFloat4::load(0.0, 0.0, 0.0, 0.0);
        let one = SimdFloat4::load(1.0, 1.0, 1.0, 1.0);
        return SoaFloat4x4 {
            cols: [SoaFloat4::load(_v.x, zero, zero, zero),
                SoaFloat4::load(zero, _v.y, zero, zero),
                SoaFloat4::load(zero, zero, _v.z, zero),
                SoaFloat4::load(zero, zero, zero, one)]
        };
    }

    // Returns the rotation matrix built from quaternion defined by x, y, z and w
    // components of _v.
    #[inline]
    pub fn from_quaternion(_q: &SoaQuaternion) -> SoaFloat4x4 {
        debug_assert!(is_normalized_est(_q).are_all_true());

        let zero = SimdFloat4::load(0.0, 0.0, 0.0, 0.0);
        let one = SimdFloat4::load(1.0, 1.0, 1.0, 1.0);
        let two = one + one;

        let xx = _q.x * _q.x;
        let xy = _q.x * _q.y;
        let xz = _q.x * _q.z;
        let xw = _q.x * _q.w;
        let yy = _q.y * _q.y;
        let yz = _q.y * _q.z;
        let yw = _q.y * _q.w;
        let zz = _q.z * _q.z;
        let zw = _q.z * _q.w;

        return SoaFloat4x4 {
            cols: [SoaFloat4::load(one - two * (yy + zz), two * (xy + zw), two * (xz - yw), zero),
                SoaFloat4::load(two * (xy - zw), one - two * (xx + zz), two * (yz + xw), zero),
                SoaFloat4::load(two * (xz + yw), two * (yz - xw), one - two * (xx + yy), zero),
                SoaFloat4::load(zero, zero, zero, one)]
        };
    }

    // Returns the affine transformation matrix built from split translation,
    // rotation (quaternion) and scale.
    #[inline]
    pub fn from_affine(_translation: &SoaFloat3,
                       _quaternion: &SoaQuaternion,
                       _scale: &SoaFloat3) -> SoaFloat4x4 {
        debug_assert!(is_normalized_est(_quaternion).are_all_true());

        let zero = SimdFloat4::load(0.0, 0.0, 0.0, 0.0);
        let one = SimdFloat4::load(1.0, 1.0, 1.0, 1.0);
        let two = one + one;

        let xx = _quaternion.x * _quaternion.x;
        let xy = _quaternion.x * _quaternion.y;
        let xz = _quaternion.x * _quaternion.z;
        let xw = _quaternion.x * _quaternion.w;
        let yy = _quaternion.y * _quaternion.y;
        let yz = _quaternion.y * _quaternion.z;
        let yw = _quaternion.y * _quaternion.w;
        let zz = _quaternion.z * _quaternion.z;
        let zw = _quaternion.z * _quaternion.w;

        return SoaFloat4x4 {
            cols:
            [SoaFloat4::load(_scale.x * (one - two * (yy + zz)), _scale.x * two * (xy + zw),
                             _scale.x * two * (xz - yw), zero),
                SoaFloat4::load(_scale.y * two * (xy - zw), _scale.y * (one - two * (xx + zz)),
                                _scale.y * two * (yz + xw), zero),
                SoaFloat4::load(_scale.z * two * (xz + yw), _scale.z * two * (yz - xw),
                                _scale.z * (one - two * (xx + yy)), zero),
                SoaFloat4::load(_translation.x, _translation.y, _translation.z, one)]
        };
    }
}

// Returns the transpose of matrix _m.
#[inline]
pub fn transpose(_m: &SoaFloat4x4) -> SoaFloat4x4 {
    return SoaFloat4x4 {
        cols:
        [SoaFloat4::load(_m.cols[0].x, _m.cols[1].x, _m.cols[2].x, _m.cols[3].x),
            SoaFloat4::load(_m.cols[0].y, _m.cols[1].y, _m.cols[2].y, _m.cols[3].y),
            SoaFloat4::load(_m.cols[0].z, _m.cols[1].z, _m.cols[2].z, _m.cols[3].z),
            SoaFloat4::load(_m.cols[0].w, _m.cols[1].w, _m.cols[2].w, _m.cols[3].w)]
    };
}

// Returns the inverse of matrix _m.
// If _invertible is not nullptr, each component will be set to true if its
// respective matrix is invertible. If _invertible is nullptr, then an assert is
// triggered in case any of the 4 matrices isn't invertible.
#[inline]
pub fn invert(_m: &SoaFloat4x4, _invertible: &mut Option<SimdInt4>) -> SoaFloat4x4 {
    let cols = &_m.cols;
    let a00 = cols[2].z * cols[3].w - cols[3].z * cols[2].w;
    let a01 = cols[2].y * cols[3].w - cols[3].y * cols[2].w;
    let a02 = cols[2].y * cols[3].z - cols[3].y * cols[2].z;
    let a03 = cols[2].x * cols[3].w - cols[3].x * cols[2].w;
    let a04 = cols[2].x * cols[3].z - cols[3].x * cols[2].z;
    let a05 = cols[2].x * cols[3].y - cols[3].x * cols[2].y;
    let a06 = cols[1].z * cols[3].w - cols[3].z * cols[1].w;
    let a07 = cols[1].y * cols[3].w - cols[3].y * cols[1].w;
    let a08 = cols[1].y * cols[3].z - cols[3].y * cols[1].z;
    let a09 = cols[1].x * cols[3].w - cols[3].x * cols[1].w;
    let a10 = cols[1].x * cols[3].z - cols[3].x * cols[1].z;
    let a11 = cols[1].y * cols[3].w - cols[3].y * cols[1].w;
    let a12 = cols[1].x * cols[3].y - cols[3].x * cols[1].y;
    let a13 = cols[1].z * cols[2].w - cols[2].z * cols[1].w;
    let a14 = cols[1].y * cols[2].w - cols[2].y * cols[1].w;
    let a15 = cols[1].y * cols[2].z - cols[2].y * cols[1].z;
    let a16 = cols[1].x * cols[2].w - cols[2].x * cols[1].w;
    let a17 = cols[1].x * cols[2].z - cols[2].x * cols[1].z;
    let a18 = cols[1].x * cols[2].y - cols[2].x * cols[1].y;

    let b0x = cols[1].y * a00 - cols[1].z * a01 + cols[1].w * a02;
    let b1x = -cols[1].x * a00 + cols[1].z * a03 - cols[1].w * a04;
    let b2x = cols[1].x * a01 - cols[1].y * a03 + cols[1].w * a05;
    let b3x = -cols[1].x * a02 + cols[1].y * a04 - cols[1].z * a05;

    let b0y = -cols[0].y * a00 + cols[0].z * a01 - cols[0].w * a02;
    let b1y = cols[0].x * a00 - cols[0].z * a03 + cols[0].w * a04;
    let b2y = -cols[0].x * a01 + cols[0].y * a03 - cols[0].w * a05;
    let b3y = cols[0].x * a02 - cols[0].y * a04 + cols[0].z * a05;

    let b0z = cols[0].y * a06 - cols[0].z * a07 + cols[0].w * a08;
    let b1z = -cols[0].x * a06 + cols[0].z * a09 - cols[0].w * a10;
    let b2z = cols[0].x * a11 - cols[0].y * a09 + cols[0].w * a12;
    let b3z = -cols[0].x * a08 + cols[0].y * a10 - cols[0].z * a12;

    let b0w = -cols[0].y * a13 + cols[0].z * a14 - cols[0].w * a15;
    let b1w = cols[0].x * a13 - cols[0].z * a16 + cols[0].w * a17;
    let b2w = -cols[0].x * a14 + cols[0].y * a16 - cols[0].w * a18;
    let b3w = cols[0].x * a15 - cols[0].y * a17 + cols[0].z * a18;

    let det =
        cols[0].x * b0x + cols[0].y * b1x + cols[0].z * b2x + cols[0].w * b3x;
    let invertible = det.cmp_ne(SimdFloat4::load(0.0, 0.0, 0.0, 0.0));
    debug_assert!((_invertible.is_none() || invertible.are_all_true()) && "Matrix is not invertible".parse().unwrap());
    if _invertible.is_some() {
        *_invertible = Some(invertible);
    }
    let inv_det = SimdFloat4::select(invertible, det.rcp_est_nr(),
                                     SimdFloat4::load(0.0, 0.0, 0.0, 0.0));

    return SoaFloat4x4 {
        cols:
        [SoaFloat4::load(b0x * inv_det, b0y * inv_det, b0z * inv_det, b0w * inv_det),
            SoaFloat4::load(b1x * inv_det, b1y * inv_det, b1z * inv_det, b1w * inv_det),
            SoaFloat4::load(b2x * inv_det, b2y * inv_det, b2z * inv_det, b2w * inv_det),
            SoaFloat4::load(b3x * inv_det, b3y * inv_det, b3z * inv_det, b3w * inv_det)]
    };
}

// Scales matrix _m along the axis defined by _v components.
// _v.w is ignored.
#[inline]
pub fn scale(_m: &SoaFloat4x4, _v: &SoaFloat4) -> SoaFloat4x4 {
    return SoaFloat4x4 {
        cols: [SoaFloat4::load(_m.cols[0].x * _v.x, _m.cols[0].y * _v.x,
                               _m.cols[0].z * _v.x, _m.cols[0].w * _v.x),
            SoaFloat4::load(_m.cols[1].x * _v.y, _m.cols[1].y * _v.y,
                            _m.cols[1].z * _v.y, _m.cols[1].w * _v.y),
            SoaFloat4::load(_m.cols[2].x * _v.z, _m.cols[2].y * _v.z,
                            _m.cols[2].z * _v.z, _m.cols[2].w * _v.z),
            _m.cols[3].clone()]
    };
}

//--------------------------------------------------------------------------------------------------
// Computes the multiplication of matrix Float4x4 and vector  _v.
impl Mul<SoaFloat4> for SoaFloat4x4 {
    type Output = SoaFloat4;
    #[inline]
    fn mul(self, rhs: SoaFloat4) -> Self::Output {
        return SoaFloat4::load(
            self.cols[0].x * rhs.x + self.cols[1].x * rhs.y + self.cols[2].x * rhs.z +
                self.cols[3].x * rhs.w,
            self.cols[0].y * rhs.x + self.cols[1].y * rhs.y + self.cols[2].y * rhs.z +
                self.cols[3].y * rhs.w,
            self.cols[0].z * rhs.x + self.cols[1].z * rhs.y + self.cols[2].z * rhs.z +
                self.cols[3].z * rhs.w,
            self.cols[0].w * rhs.x + self.cols[1].w * rhs.y + self.cols[2].w * rhs.z +
                self.cols[3].w * rhs.w);
    }
}

impl Mul for SoaFloat4x4 {
    type Output = SoaFloat4x4;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        return SoaFloat4x4 {
            cols: [self.clone() * rhs.cols[0].clone(),
                self.clone() * rhs.cols[1].clone(),
                self.clone() * rhs.cols[2].clone(),
                self.clone() * rhs.cols[3].clone()]
        };
    }
}

impl Mul for &SoaFloat4x4 {
    type Output = SoaFloat4x4;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        return SoaFloat4x4 {
            cols: [self.clone() * rhs.cols[0].clone(),
                self.clone() * rhs.cols[1].clone(),
                self.clone() * rhs.cols[2].clone(),
                self.clone() * rhs.cols[3].clone()]
        };
    }
}

impl Add for SoaFloat4x4 {
    type Output = SoaFloat4x4;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        return SoaFloat4x4 {
            cols:
            [SoaFloat4::load(self.cols[0].x + rhs.cols[0].x, self.cols[0].y + rhs.cols[0].y,
                             self.cols[0].z + rhs.cols[0].z, self.cols[0].w + rhs.cols[0].w),
                SoaFloat4::load(self.cols[1].x + rhs.cols[1].x, self.cols[1].y + rhs.cols[1].y,
                                self.cols[1].z + rhs.cols[1].z, self.cols[1].w + rhs.cols[1].w),
                SoaFloat4::load(self.cols[2].x + rhs.cols[2].x, self.cols[2].y + rhs.cols[2].y,
                                self.cols[2].z + rhs.cols[2].z, self.cols[2].w + rhs.cols[2].w),
                SoaFloat4::load(self.cols[3].x + rhs.cols[3].x, self.cols[3].y + rhs.cols[3].y,
                                self.cols[3].z + rhs.cols[3].z, self.cols[3].w + rhs.cols[3].w)]
        };
    }
}

impl Add for &SoaFloat4x4 {
    type Output = SoaFloat4x4;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        return SoaFloat4x4 {
            cols:
            [SoaFloat4::load(self.cols[0].x + rhs.cols[0].x, self.cols[0].y + rhs.cols[0].y,
                             self.cols[0].z + rhs.cols[0].z, self.cols[0].w + rhs.cols[0].w),
                SoaFloat4::load(self.cols[1].x + rhs.cols[1].x, self.cols[1].y + rhs.cols[1].y,
                                self.cols[1].z + rhs.cols[1].z, self.cols[1].w + rhs.cols[1].w),
                SoaFloat4::load(self.cols[2].x + rhs.cols[2].x, self.cols[2].y + rhs.cols[2].y,
                                self.cols[2].z + rhs.cols[2].z, self.cols[2].w + rhs.cols[2].w),
                SoaFloat4::load(self.cols[3].x + rhs.cols[3].x, self.cols[3].y + rhs.cols[3].y,
                                self.cols[3].z + rhs.cols[3].z, self.cols[3].w + rhs.cols[3].w)]
        };
    }
}

impl Sub for SoaFloat4x4 {
    type Output = SoaFloat4x4;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        return SoaFloat4x4 {
            cols:
            [SoaFloat4::load(self.cols[0].x - rhs.cols[0].x, self.cols[0].y - rhs.cols[0].y,
                             self.cols[0].z - rhs.cols[0].z, self.cols[0].w - rhs.cols[0].w),
                SoaFloat4::load(self.cols[1].x - rhs.cols[1].x, self.cols[1].y - rhs.cols[1].y,
                                self.cols[1].z - rhs.cols[1].z, self.cols[1].w - rhs.cols[1].w),
                SoaFloat4::load(self.cols[2].x - rhs.cols[2].x, self.cols[2].y - rhs.cols[2].y,
                                self.cols[2].z - rhs.cols[2].z, self.cols[2].w - rhs.cols[2].w),
                SoaFloat4::load(self.cols[3].x - rhs.cols[3].x, self.cols[3].y - rhs.cols[3].y,
                                self.cols[3].z - rhs.cols[3].z, self.cols[3].w - rhs.cols[3].w)]
        };
    }
}

impl Sub for &SoaFloat4x4 {
    type Output = SoaFloat4x4;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        return SoaFloat4x4 {
            cols:
            [SoaFloat4::load(self.cols[0].x - rhs.cols[0].x, self.cols[0].y - rhs.cols[0].y,
                             self.cols[0].z - rhs.cols[0].z, self.cols[0].w - rhs.cols[0].w),
                SoaFloat4::load(self.cols[1].x - rhs.cols[1].x, self.cols[1].y - rhs.cols[1].y,
                                self.cols[1].z - rhs.cols[1].z, self.cols[1].w - rhs.cols[1].w),
                SoaFloat4::load(self.cols[2].x - rhs.cols[2].x, self.cols[2].y - rhs.cols[2].y,
                                self.cols[2].z - rhs.cols[2].z, self.cols[2].w - rhs.cols[2].w),
                SoaFloat4::load(self.cols[3].x - rhs.cols[3].x, self.cols[3].y - rhs.cols[3].y,
                                self.cols[3].z - rhs.cols[3].z, self.cols[3].w - rhs.cols[3].w)]
        };
    }
}