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
#[derive(Clone, Copy)]
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

impl SoaFloat4x4 {
    // Returns the transpose of matrix self.
    #[inline]
    pub fn transpose(&self) -> SoaFloat4x4 {
        return SoaFloat4x4 {
            cols:
            [SoaFloat4::load(self.cols[0].x, self.cols[1].x, self.cols[2].x, self.cols[3].x),
                SoaFloat4::load(self.cols[0].y, self.cols[1].y, self.cols[2].y, self.cols[3].y),
                SoaFloat4::load(self.cols[0].z, self.cols[1].z, self.cols[2].z, self.cols[3].z),
                SoaFloat4::load(self.cols[0].w, self.cols[1].w, self.cols[2].w, self.cols[3].w)]
        };
    }

    // Returns the inverse of matrix self.
    // If _invertible is not nullptr, each component will be set to true if its
    // respective matrix is invertible. If _invertible is nullptr, then an assert is
    // triggered in case any of the 4 matrices isn't invertible.
    #[inline]
    pub fn invert(&self, _invertible: &mut Option<SimdInt4>) -> SoaFloat4x4 {
        let cols = &self.cols;
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
        debug_assert!((_invertible.is_some() || invertible.are_all_true()) && "Matrix is not invertible".parse().unwrap_or(true));
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
    pub fn scale(&self, _v: &SoaFloat4) -> SoaFloat4x4 {
        return SoaFloat4x4 {
            cols: [SoaFloat4::load(self.cols[0].x * _v.x, self.cols[0].y * _v.x,
                                   self.cols[0].z * _v.x, self.cols[0].w * _v.x),
                SoaFloat4::load(self.cols[1].x * _v.y, self.cols[1].y * _v.y,
                                self.cols[1].z * _v.y, self.cols[1].w * _v.y),
                SoaFloat4::load(self.cols[2].x * _v.z, self.cols[2].y * _v.z,
                                self.cols[2].z * _v.z, self.cols[2].w * _v.z),
                self.cols[3].clone()]
        };
    }
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

//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
#[cfg(test)]
mod ozz_soa_math {
    use crate::soa_float4x4::SoaFloat4x4;
    use crate::simd_math::*;
    use crate::math_test_helper::*;
    use crate::*;
    use crate::soa_float::{SoaFloat4, SoaFloat3};
    use crate::soa_quaternion::SoaQuaternion;

    #[test]
    fn soa_float4x4constant() {
        let identity = SoaFloat4x4::identity();
        expect_soa_float4x4_eq!(identity, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
                              1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0);
    }

    #[test]
    #[allow(overflowing_literals)]
    fn soa_float4x4arithmetic() {
        let m0 = SoaFloat4x4 {
            cols: [
                SoaFloat4 {
                    x: SimdFloat4::load(0.0, 1.0, 0.0, 0.0),
                    y: SimdFloat4::load(1.0, 0.0, 0.0, 0.0),
                    z: SimdFloat4::load(2.0, 0.0, 0.0, -1.0),
                    w: SimdFloat4::load(3.0, 0.0, 0.0, 0.0),
                },
                SoaFloat4 {
                    x: SimdFloat4::load(4.0, 0.0, 0.0, 0.0),
                    y: SimdFloat4::load(5.0, 1.0, 0.0, 1.0),
                    z: SimdFloat4::load(6.0, 0.0, 0.0, 0.0),
                    w: SimdFloat4::load(7.0, 0.0, 0.0, 0.0),
                },
                SoaFloat4 {
                    x: SimdFloat4::load(8.0, 0.0, 0.0, 1.0),
                    y: SimdFloat4::load(9.0, 0.0, 0.0, 0.0),
                    z: SimdFloat4::load(10.0, 1.0, 0.0, 0.0),
                    w: SimdFloat4::load(11.0, 0.0, 0.0, 0.0),
                },
                SoaFloat4 {
                    x: SimdFloat4::load(12.0, 0.0, 0.0, 0.0),
                    y: SimdFloat4::load(13.0, 0.0, 0.0, 0.0),
                    z: SimdFloat4::load(14.0, 0.0, 0.0, 0.0),
                    w: SimdFloat4::load(15.0, 1.0, 0.0, 1.0),
                }]
        };
        let m1 = SoaFloat4x4 {
            cols: [
                SoaFloat4 {
                    x: SimdFloat4::load(-0.0, 0.0, 0.0, 1.0),
                    y: SimdFloat4::load(-1.0, -1.0, 0.0, 0.0),
                    z: SimdFloat4::load(-2.0, 2.0, -1.0, 0.0),
                    w: SimdFloat4::load(-3.0, 3.0, 0.0, 0.0),
                },
                SoaFloat4 {
                    x: SimdFloat4::load(-4.0, -4.0, 0.0, 0.0),
                    y: SimdFloat4::load(-5.0, 5.0, 1.0, 1.0),
                    z: SimdFloat4::load(-6.0, 6.0, 0.0, 0.0),
                    w: SimdFloat4::load(-7.0, -7.0, 0.0, 0.0),
                },
                SoaFloat4 {
                    x: SimdFloat4::load(-8.0, 8.0, 1.0, 0.0),
                    y: SimdFloat4::load(-9.0, -9.0, 0.0, 0.0),
                    z: SimdFloat4::load(-10.0, -10.0, 0.0, 1.0),
                    w: SimdFloat4::load(-11.0, 11.0, 0.0, 0.0),
                },
                SoaFloat4 {
                    x: SimdFloat4::load(-12.0, -12.0, 0.0, 0.0),
                    y: SimdFloat4::load(-13.0, 13.0, 0.0, 0.0),
                    z: SimdFloat4::load(-14.0, -14.0, 0.0, 0.0),
                    w: SimdFloat4::load(-15.0, 15.0, 1.0, 1.0),
                }]
        };
        let m2 = SoaFloat4x4 {
            cols: [
                SoaFloat4 {
                    x: SimdFloat4::load(2.0, 0.0, 0.0, 1.0),
                    y: SimdFloat4::load(0.0, -1.0, 0.0, 0.0),
                    z: SimdFloat4::load(0.0, 2.0, -1.0, 0.0),
                    w: SimdFloat4::load(0.0, 3.0, 0.0, 0.0),
                },
                SoaFloat4 {
                    x: SimdFloat4::load(0.0, -4.0, 0.0, 0.0),
                    y: SimdFloat4::load(0.0, 5.0, 1.0, 1.0),
                    z: SimdFloat4::load(-2.0, 6.0, 0.0, 0.0),
                    w: SimdFloat4::load(0.0, -7.0, 0.0, 0.0),
                },
                SoaFloat4 {
                    x: SimdFloat4::load(0.0, 8.0, 1.0, 0.0),
                    y: SimdFloat4::load(3.0, -9.0, 0.0, 0.0),
                    z: SimdFloat4::load(0.0, -10.0, 0.0, 1.0),
                    w: SimdFloat4::load(0.0, 11.0, 0.0, 0.0),
                },
                SoaFloat4 {
                    x: SimdFloat4::load(0.0, -12.0, 0.0, 0.0),
                    y: SimdFloat4::load(46.0, 13.0, 0.0, 0.0),
                    z: SimdFloat4::load(12.0, -14.0, 0.0, 0.0),
                    w: SimdFloat4::load(1.0, 15.0, 1.0, 1.0),
                }]
        };
        let v = SoaFloat4 {
            x: SimdFloat4::load(0.0, 1.0, -2.0, 3.0),
            y: SimdFloat4::load(-1.0, 2.0, 5.0, 46.0),
            z: SimdFloat4::load(-2.0, 3.0, 7.0, -1.0),
            w: SimdFloat4::load(-3.0, 4.0, 0.0, 1.0),
        };
        let mul_vector = m0 * v;
        expect_soa_float4_eq!(mul_vector, -56.0, 1.0, 0.0, -1.0, -62.0, 2.0, 0.0, 46.0,
                            -68.0, 3.0, 0.0, -3.0, -74.0, 4.0, 0.0, 1.0);


        let mul_mat = m0 * m1;
        expect_soa_float4x4_eq!(
            mul_mat, -56.0, 0.0, 0.0, 0.0, -62.0, -1.0, 0.0, 0.0, -68.0, 2.0, 0.0,
            -1.0, -74.0, 3.0, 0.0, 0.0, -152.0, -4.0, 0.0, 0.0, -174.0, 5.0, 0.0, 1.0,
            -196.0, 6.0, 0.0, 0.0, -218.0, -7.0, 0.0, 0.0, -248.0, 8.0, 0.0, 1.0,
            -286.0, -9.0, 0.0, 0.0, -324.0, -10.0, 0.0, 0.0, -362.0, 11.0, 0.0, 0.0,
            -344.0, -12.0, 0.0, 0.0, -398.0, 13.0, 0.0, 0.0, -452.0, -14.0, 0.0, 0.0,
            -506.0, 15.0, 0.0, 1.0);

        let add_mat = m0 + m1;
        expect_soa_float4x4_eq!(
            add_mat, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 2.0, -1.0, -1.0,
            0.0, 3.0, 0.0, 0.0, 0.0, -4.0, 0.0, 0.0, 0.0, 6.0, 1.0, 2.0, 0.0, 6.0,
            0.0, 0.0, 0.0, -7.0, 0.0, 0.0, 0.0, 8.0, 1.0, 1.0, 0.0, -9.0, 0.0, 0.0,
            0.0, -9.0, 0.0, 1.0, 0.0, 11.0, 0.0, 0.0, 0.0, -12.0, 0.0, 0.0, 0.0, 13.0,
            0.0, 0.0, 0.0, -14.0, 0.0, 0.0, 0.0, 16.0, 1.0, 2.0);

        let sub_mat = m0 - m1;
        expect_soa_float4x4_eq!(
            sub_mat, 0.0, 1.0, 0.0, -1.0, 2.0, 1.0, 0.0, 0.0, 4.0, -2.0, 1.0, -1.0,
            6.0, -3.0, 0.0, 0.0, 8.0, 4.0, 0.0, 0.0, 10.0, -4.0, -1.0, 0.0, 12.0,
            -6.0, 0.0, 0.0, 14.0, 7.0, 0.0, 0.0, 16.0, -8.0, -1.0, 1.0, 18.0, 9.0,
            0.0, 0.0, 20.0, 11.0, 0.0, -1.0, 22.0, -11.0, 0.0, 0.0, 24.0, 12.0, 0.0,
            0.0, 26.0, -13.0, 0.0, 0.0, 28.0, 14.0, 0.0, 0.0, 30.0, -14.0, -1.0, 0.0);

        let transpose = m0.transpose();
        expect_soa_float4x4_eq!(
            transpose, 0.0, 1.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 1.0,
            12.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 5.0, 1.0, 0.0, 1.0, 9.0, 0.0,
            0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, -1.0, 6.0, 0.0, 0.0, 0.0,
            10.0, 1.0, 0.0, 0.0, 14.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 7.0, 0.0,
            0.0, 0.0, 11.0, 0.0, 0.0, 0.0, 15.0, 1.0, 0.0, 1.0);

        let mut invertible: Option<SimdInt4> = None;
        let invert_ident = SoaFloat4x4::identity().invert(&mut invertible);
        expect_soa_float4x4_eq!(
            invert_ident, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
            1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0);

        let invert = m2.invert(&mut invertible);
        expect_soa_float4x4_eq!(invert, 0.5, 0.216667, 0.0, 1.0, 0.0, 2.75, 0.0, 0.0,
                              0.0, 1.6, 1.0, 0.0, 0.0, 0.066666, 0.0, 0.0, 0.0, 0.2,
                              0.0, 0.0, 0.0, 2.5, 1.0, 1.0, 0.333333, 1.4, 0.0, 0.0,
                              0.0, 0.1, 0.0, 0.0, 0.0, 0.25, -1.0, 0.0, -0.5, 0.5,
                              0.0, 0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.233333, 0.0, 0.0, 6.0, 0.5, 0.0, 0.0, -15.33333, 0.3,
                              0.0, 0.0, 1.0, 0.03333, 1.0, 1.0);

        let invert_mul = m2 * invert;
        expect_soa_float4x4_eq!(invert_mul, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
                              1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0);

        // EXPECT_ASSERTION(Invert(m0), "Matrix is not invertible");

        // Invertible
        let mut invertible: Option<SimdInt4> = Some(SimdInt4::zero());
        expect_soa_float4x4_eq!(m2.invert(&mut invertible), 0.5, 0.216667, 0.0, 1.0, 0.0, 2.75, 0.0, 0.0,
            0.0, 1.6, 1.0, 0.0, 0.0, 0.066666, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0,
            2.5, 1.0, 1.0, 0.333333, 1.4, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.25,
            -1.0, 0.0, -0.5, 0.5, 0.0, 0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.233333, 0.0, 0.0, 6.0, 0.5, 0.0, 0.0, -15.33333, 0.3, 0.0, 0.0,
            1.0, 0.03333, 1.0, 1.0);
        expect_simd_int_eq!(invertible.unwrap(), 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff);

        // Non invertible
        // EXPECT_ASSERTION(Invert(m0), "Matrix is not invertible");

        let mut not_invertible: Option<SimdInt4> = Some(SimdInt4::zero());
        expect_soa_float4x4_eq!(m0.invert(&mut not_invertible), 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0);
        expect_simd_int_eq!(not_invertible.unwrap(), 0, 0xffffffff, 0, 0xffffffff);
    }

    #[test]
    fn soa_float4x4scale() {
        let m0 = SoaFloat4x4 {
            cols:
            [SoaFloat4 {
                x: SimdFloat4::load(0.0, 1.0, 0.0, 0.0),
                y: SimdFloat4::load(1.0, 0.0, -1.0, 0.0),
                z: SimdFloat4::load(2.0, 0.0, 2.0, -1.0),
                w: SimdFloat4::load(3.0, 0.0, 3.0, 0.0),
            },
                SoaFloat4 {
                    x: SimdFloat4::load(4.0, 0.0, -4.0, 0.0),
                    y: SimdFloat4::load(5.0, 1.0, 5.0, 1.0),
                    z: SimdFloat4::load(6.0, 0.0, 6.0, 0.0),
                    w: SimdFloat4::load(7.0, 0.0, -7.0, 0.0),
                },
                SoaFloat4 {
                    x: SimdFloat4::load(8.0, 0.0, 8.0, 1.0),
                    y: SimdFloat4::load(9.0, 0.0, -9.0, 0.0),
                    z: SimdFloat4::load(10.0, 1.0, -10.0, 0.0),
                    w: SimdFloat4::load(11.0, 0.0, 11.0, 0.0),
                },
                SoaFloat4 {
                    x: SimdFloat4::load(12.0, 0.0, -12.0, 0.0),
                    y: SimdFloat4::load(13.0, 0.0, 13.0, 0.0),
                    z: SimdFloat4::load(14.0, 0.0, -14.0, 0.0),
                    w: SimdFloat4::load(15.0, 1.0, 15.0, 1.0),
                }]
        };
        let v = SoaFloat4 {
            x: SimdFloat4::load(0.0, 1.0, -2.0, 3.0),
            y: SimdFloat4::load(-1.0, 2.0, 5.0, 46.0),
            z: SimdFloat4::load(-2.0, 3.0, 7.0, -1.0),
            w: SimdFloat4::load(-3.0, 4.0, 0.0, 1.0),
        };

        let scaling = SoaFloat4x4::scaling(&v);
        expect_soa_float4x4_eq!(scaling, 0.0, 1.0, -2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              -1.0, 2.0, 5.0, 46.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 3.0,
                              7.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0);

        let scale_mul = m0 * scaling;
        expect_soa_float4x4_eq!(scale_mul, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0,
                              0.0, -4.0, -3.0, 0.0, 0.0, -6.0, 0.0, -4.0, 0.0, -20.0,
                              0.0, -5.0, 2.0, 25.0, 46.0, -6.0, 0.0, 30.0, 0.0, -7.0,
                              0.0, -35.0, 0.0, -16.0, 0.0, 56.0, -1.0, -18.0, 0.0,
                              -63.0, 0.0, -20.0, 3.0, -70.0, 0.0, -22.0, 0.0, 77.0,
                              0.0, 12.0, 0.0, -12.0, 0.0, 13.0, 0.0, 13.0, 0.0, 14.0,
                              0.0, -14.0, 0.0, 15.0, 1.0, 15.0, 1.0);

        let scale = m0.scale(&v);
        expect_soa_float4x4_eq!(scale, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0,
                              -4.0, -3.0, 0.0, 0.0, -6.0, 0.0, -4.0, 0.0, -20.0, 0.0,
                              -5.0, 2.0, 25.0, 46.0, -6.0, 0.0, 30.0, 0.0, -7.0, 0.0,
                              -35.0, 0.0, -16.0, 0.0, 56.0, -1.0, -18.0, 0.0, -63.0,
                              0.0, -20.0, 3.0, -70.0, 0.0, -22.0, 0.0, 77.0, 0.0,
                              12.0, 0.0, -12.0, 0.0, 13.0, 0.0, 13.0, 0.0, 14.0, 0.0,
                              -14.0, 0.0, 15.0, 1.0, 15.0, 1.0);
    }

    #[test]
    fn soa_float4x4rotate() {
        // let unormalized =
        //     SoaQuaternion::Load(SimdFloat4::load(0.0, 0.0, 0.0, 0.0),
        //                         SimdFloat4::load(0.0, 0.0, 1.0, 0.0),
        //                         SimdFloat4::load(0.0, 0.0, 0.0, 0.0),
        //                         SimdFloat4::load(1.0, 1.0, 1.0, 1.0));
        //
        // EXPECT_ASSERTION(SoaFloat4x4::from_quaternion(unormalized), "IsNormalized");

        let identity = SoaFloat4x4::from_quaternion(&SoaQuaternion::identity());
        expect_soa_float4x4_eq!(identity, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
                              1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0);
        let quaternion = SoaQuaternion::load(
            SimdFloat4::load(0.70710677, 0.0, 0.0, -0.382683432),
            SimdFloat4::load(0.0, 0.70710677, 0.0, 0.0),
            SimdFloat4::load(0.70710677, 0.0, 0.0, 0.0),
            SimdFloat4::load(0.0, 0.70710677, 1.0, 0.9238795));
        let matrix = SoaFloat4x4::from_quaternion(&quaternion);
        expect_soa_float4x4_eq!(
            matrix, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 1.0, 0.707106, 0.0, 0.0,
            0.0, -0.707106, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.707106, 0.0, 0.0, 1.0, 0.707106, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0);
    }

    #[test]
    fn soa_float4x4affine() {
        let identity = SoaFloat4x4::from_affine(
            &SoaFloat3::zero(), &SoaQuaternion::identity(), &SoaFloat3::one());
        expect_soa_float4x4_eq!(identity, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
                              1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0);
        let translation =
            SoaFloat3::load(SimdFloat4::load(0.0, 46.0, 7.0, -12.0),
                            SimdFloat4::load(0.0, 12.0, 7.0, -46.0),
                            SimdFloat4::load(0.0, 0.0, 7.0, 46.0));
        let scale =
            SoaFloat3::load(SimdFloat4::load(1.0, 1.0, -1.0, 0.1),
                            SimdFloat4::load(1.0, 2.0, -1.0, 0.1),
                            SimdFloat4::load(1.0, 3.0, -1.0, 0.1));
        let quaternion = SoaQuaternion::load(
            SimdFloat4::load(0.70710677, 0.0, 0.0, -0.382683432),
            SimdFloat4::load(0.0, 0.70710677, 0.0, 0.0),
            SimdFloat4::load(0.70710677, 0.0, 0.0, 0.0),
            SimdFloat4::load(0.0, 0.70710677, 1.0, 0.9238795));
        let matrix =
            SoaFloat4x4::from_affine(&translation, &quaternion, &scale);
        expect_soa_float4x4_eq!(
            matrix, 0.0, 0.0, -1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0707106, 0.0, 0.0,
            0.0, -0.0707106, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0707106, 0.0, 0.0, -1.0, 0.0707106, 0.0, 0.0, 0.0, 0.0, 0.0, 46.0, 7.0,
            -12.0, 0.0, 12.0, 7.0, -46.0, 0.0, 0.0, 7.0, 46.0, 1.0, 1.0, 1.0, 1.0);
    }
}