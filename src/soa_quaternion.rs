/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::math_constant::*;
use crate::simd_math::*;
use crate::soa_float::SoaFloat4;
use std::ops::{Add, Neg, Mul};

#[derive(Copy, Clone)]
pub struct SoaQuaternion {
    pub x: SimdFloat4,
    pub y: SimdFloat4,
    pub z: SimdFloat4,
    pub w: SimdFloat4,
}

impl SoaQuaternion {
    // Loads a quaternion from 4 SimdFloat4 values.
    #[inline]
    pub fn load(_x: SimdFloat4, _y: SimdFloat4,
                _z: SimdFloat4, _w: SimdFloat4) -> SoaQuaternion {
        return SoaQuaternion { x: _x, y: _y, z: _z, w: _w };
    }

    // Returns the identity SoaQuaternion.
    #[inline]
    pub fn identity() -> SoaQuaternion {
        let zero = SimdFloat4::load(0.0, 0.0, 0.0, 0.0);
        return SoaQuaternion {
            x: zero,
            y: zero,
            z: zero,
            w: SimdFloat4::load(1.0, 1.0, 1.0, 1.0),
        };
    }
}

//--------------------------------------------------------------------------------------------------
impl Neg for SoaQuaternion {
    type Output = SoaQuaternion;
    #[inline]
    fn neg(self) -> Self::Output {
        return SoaQuaternion { x: -self.x, y: -self.y, z: -self.z, w: -self.w };
    }
}

impl SoaQuaternion {
    // Returns the conjugate of self. This is the same as the inverse if self is
    // normalized. Otherwise the magnitude of the inverse is 1.0/|self|.
    #[inline]
    pub fn conjugate(&self) -> SoaQuaternion {
        return SoaQuaternion { x: -self.x, y: -self.y, z: -self.z, w: self.w };
    }

    // Returns the 4D dot product of quaternion _a and _b.
    #[inline]
    pub fn dot(&self, _b: &SoaQuaternion) -> SimdFloat4 {
        return self.x * _b.x + self.y * _b.y + self.z * _b.z + self.w * _b.w;
    }

    // Returns the normalized SoaQuaternion self.
    #[inline]
    pub fn normalize(&self) -> SoaQuaternion {
        let len2 = self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w;
        let inv_len = SimdFloat4::load(1.0, 1.0, 1.0, 1.0) / (len2).sqrt();
        return SoaQuaternion {
            x: self.x * inv_len,
            y: self.y * inv_len,
            z: self.z * inv_len,
            w: self.w * inv_len,
        };
    }

    // Returns the estimated normalized SoaQuaternion self.
    #[inline]
    pub fn normalize_est(&self) -> SoaQuaternion {
        let len2 = self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w;
        // Uses RSqrtEstNR (with one more Newton-Raphson step) as quaternions loose
        // much precision due to normalization.
        let inv_len = len2.rsqrt_est_nr();
        return SoaQuaternion {
            x: self.x * inv_len,
            y: self.y * inv_len,
            z: self.z * inv_len,
            w: self.w * inv_len,
        };
    }

    // Test if each quaternion of self is normalized.
    #[inline]
    pub fn is_normalized(&self) -> SimdInt4 {
        let len2 = self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w;
        return (len2 - SimdFloat4::load(1.0, 1.0, 1.0, 1.0)).abs().cmp_lt(SimdFloat4::load(K_NORMALIZATION_TOLERANCE_SQ,
                                                                                           K_NORMALIZATION_TOLERANCE_SQ,
                                                                                           K_NORMALIZATION_TOLERANCE_SQ,
                                                                                           K_NORMALIZATION_TOLERANCE_SQ));
    }


    // Test if each quaternion of self is normalized. using estimated tolerance.
    #[inline]
    pub fn is_normalized_est(&self) -> SimdInt4 {
        let len2 = self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w;
        return (len2 - SimdFloat4::load(1.0, 1.0, 1.0, 1.0)).abs().cmp_lt(SimdFloat4::load(K_NORMALIZATION_TOLERANCE_EST_SQ,
                                                                                           K_NORMALIZATION_TOLERANCE_EST_SQ,
                                                                                           K_NORMALIZATION_TOLERANCE_EST_SQ,
                                                                                           K_NORMALIZATION_TOLERANCE_EST_SQ));
    }

    // Returns the linear interpolation of SoaQuaternion self and _b with coefficient
    // _f.
    #[inline]
    pub fn lerp(&self, _b: &SoaQuaternion,
                _f: SimdFloat4) -> SoaQuaternion {
        return SoaQuaternion {
            x: (_b.x - self.x) * _f + self.x,
            y: (_b.y - self.y) * _f + self.y,
            z: (_b.z - self.z) * _f + self.z,
            w: (_b.w - self.w) * _f + self.w,
        };
    }

    // Returns the linear interpolation of SoaQuaternion self and _b with coefficient
    // _f.
    #[inline]
    pub fn nlerp(&self, _b: &SoaQuaternion, _f: SimdFloat4) -> SoaQuaternion {
        let lerp = SoaFloat4::load((_b.x - self.x) * _f + self.x, (_b.y - self.y) * _f + self.y,
                                   (_b.z - self.z) * _f + self.z, (_b.w - self.w) * _f + self.w);
        let len2 = lerp.x * lerp.x + lerp.y * lerp.y + lerp.z * lerp.z + lerp.w * lerp.w;
        let inv_len = SimdFloat4::load(1.0, 1.0, 1.0, 1.0) / (len2).sqrt();
        return SoaQuaternion {
            x: lerp.x * inv_len,
            y: lerp.y * inv_len,
            z: lerp.z * inv_len,
            w: lerp.w * inv_len,
        };
    }

    // Returns the estimated linear interpolation of SoaQuaternion self and _b with
// coefficient _f.
    #[inline]
    pub fn nlerp_est(&self, _b: &SoaQuaternion, _f: SimdFloat4) -> SoaQuaternion {
        let lerp = SoaFloat4::load((_b.x - self.x) * _f + self.x, (_b.y - self.y) * _f + self.y,
                                   (_b.z - self.z) * _f + self.z, (_b.w - self.w) * _f + self.w);
        let len2 = lerp.x * lerp.x + lerp.y * lerp.y + lerp.z * lerp.z + lerp.w * lerp.w;
        // Uses RSqrtEstNR (with one more Newton-Raphson step) as quaternions loose
        // much precision due to normalization.
        let inv_len = len2.rsqrt_est_nr();
        return SoaQuaternion {
            x: lerp.x * inv_len,
            y: lerp.y * inv_len,
            z: lerp.z * inv_len,
            w: lerp.w * inv_len,
        };
    }
}

//--------------------------------------------------------------------------------------------------
// Returns the addition of _a and _b.
impl Add for SoaQuaternion {
    type Output = SoaQuaternion;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        return SoaQuaternion {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
            w: self.w + rhs.w,
        };
    }
}

impl Add for &SoaQuaternion {
    type Output = SoaQuaternion;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        return SoaQuaternion {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
            w: self.w + rhs.w,
        };
    }
}

impl Mul<SimdFloat4> for SoaQuaternion {
    type Output = SoaQuaternion;
    #[inline]
    fn mul(self, rhs: SimdFloat4) -> Self::Output {
        return SoaQuaternion {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
            w: self.w * rhs,
        };
    }
}

impl Mul for SoaQuaternion {
    type Output = SoaQuaternion;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        return SoaQuaternion {
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y + self.y * rhs.w + self.z * rhs.x - self.x * rhs.z,
            z: self.w * rhs.z + self.z * rhs.w + self.x * rhs.y - self.y * rhs.x,
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
        };
    }
}

impl Mul for &SoaQuaternion {
    type Output = SoaQuaternion;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        return SoaQuaternion {
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y + self.y * rhs.w + self.z * rhs.x - self.x * rhs.z,
            z: self.w * rhs.z + self.z * rhs.w + self.x * rhs.y - self.y * rhs.x,
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
        };
    }
}

impl SoaQuaternion {
    #[inline]
    pub fn eq(&self, other: &Self) -> SimdInt4 {
        let x = self.x.cmp_eq(other.x);
        let y = self.y.cmp_eq(other.y);
        let z = self.z.cmp_eq(other.z);
        let w = self.w.cmp_eq(other.w);
        return x.and(y).and(z).and(w);
    }
}

//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
#[cfg(test)]
mod ozz_soa_math {
    use crate::soa_float::*;
    use crate::math_test_helper::*;
    use crate::*;
    use crate::soa_quaternion::*;

    #[test]
    fn soa_quaternion_constant() {
        expect_soa_quaternion_eq!(SoaQuaternion::identity(), 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                                1.0);
    }

    #[test]
    #[allow(overflowing_literals)]
    fn soa_quaternion_arithmetic() {
        let a = SoaQuaternion::load(
            SimdFloat4::load(0.70710677, 0.0, 0.0, 0.382683432),
            SimdFloat4::load(0.0, 0.0, 0.70710677, 0.0),
            SimdFloat4::load(0.0, 0.0, 0.0, 0.0),
            SimdFloat4::load(0.70710677, 1.0, 0.70710677, 0.9238795));
        let b = SoaQuaternion::load(
            SimdFloat4::load(0.0, 0.70710677, 0.0, -0.382683432),
            SimdFloat4::load(0.0, 0.0, 0.70710677, 0.0),
            SimdFloat4::load(0.0, 0.0, 0.0, 0.0),
            SimdFloat4::load(1.0, 0.70710677, 0.70710677, 0.9238795));
        let denorm =
            SoaQuaternion::load(SimdFloat4::load(0.5, 0.0, 2.0, 3.0),
                                SimdFloat4::load(4.0, 0.0, 6.0, 7.0),
                                SimdFloat4::load(8.0, 0.0, 10.0, 11.0),
                                SimdFloat4::load(12.0, 1.0, 14.0, 15.0));

        assert_eq!(a.is_normalized().are_all_true(), true);
        assert_eq!(b.is_normalized().are_all_true(), true);
        expect_simd_int_eq!(denorm.is_normalized(), 0, 0xffffffff, 0, 0);


        let conjugate = a.conjugate();
        expect_soa_quaternion_eq!(conjugate, -0.70710677, -0.0, -0.0, -0.382683432,
                                -0.0, -0.0, -0.70710677, -0.0, -0.0, -0.0, -0.0, -0.0,
                                0.70710677, 1.0, 0.70710677, 0.9238795);
        assert_eq!(conjugate.is_normalized().are_all_true(), true);

        let negate = -a;
        expect_soa_quaternion_eq!(negate, -0.70710677, 0.0, 0.0, -0.382683432, 0.0, 0.0,
                                -0.70710677, 0.0, 0.0, 0.0, 0.0, 0.0, -0.70710677,
                                -1.0, -0.70710677, -0.9238795);

        let add = a + b;
        expect_soa_quaternion_eq!(add, 0.70710677, 0.70710677, 0.0, 0.0, 0.0, 0.0,
                                1.41421354, 0.0, 0.0, 0.0, 0.0, 0.0, 1.70710677,
                                1.70710677, 1.41421354, 1.847759);

        let muls = a * SimdFloat4::load1(2.0);
        expect_soa_quaternion_eq!(muls, 1.41421354, 0.0, 0.0, 0.765366864, 0.0, 0.0,
                                1.41421354, 0.0, 0.0, 0.0, 0.0, 0.0, 1.41421354,
                                2.0, 1.41421354, 1.847759);

        let mul0 = a * conjugate;
        expect_soa_quaternion_eq!(mul0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0);
        assert_eq!(mul0.is_normalized().are_all_true(), true);

        let mul1 = conjugate * a;
        expect_soa_quaternion_eq!(mul1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0);
        assert_eq!(mul1.is_normalized().are_all_true(), true);

        let dot = a.dot(&b);
        expect_soa_float1_eq!(dot, 0.70710677, 0.70710677, 1.0, 0.70710677);

        let normalize = denorm.normalize();
        assert_eq!(normalize.is_normalized().are_all_true(), true);
        expect_soa_quaternion_eq!(normalize, 0.033389, 0.0, 0.1091089, 0.1492555,
                                0.267112, 0.0, 0.3273268, 0.348263, 0.53422445, 0.0,
                                0.545544, 0.547270, 0.80133667, 1.0, 0.763762,
                                0.74627789);

        let normalize_est = denorm.normalize_est();
        expect_soa_quaternion_eq!(normalize_est, 0.033389, 0.0, 0.1091089,
                                    0.1492555, 0.267112, 0.0, 0.3273268, 0.348263,
                                    0.53422445, 0.0, 0.545544, 0.547270, 0.80133667,
                                    1.0, 0.763762, 0.74627789);
        assert_eq!(normalize_est.is_normalized_est().are_all_true(), true);

        let lerp_0 = a.lerp(&b, SimdFloat4::zero());
        expect_soa_quaternion_eq!(lerp_0, 0.70710677, 0.0, 0.0, 0.382683432, 0.0, 0.0,
                                0.70710677, 0.0, 0.0, 0.0, 0.0, 0.0, 0.70710677, 1.0,
                                0.70710677, 0.9238795);

        let lerp_1 = a.lerp(&b, SimdFloat4::one());
        expect_soa_quaternion_eq!(lerp_1, 0.0, 0.70710677, 0.0, -0.382683432, 0.0, 0.0,
                                0.70710677, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.70710677,
                                0.70710677, 0.9238795);

        let lerp_0_2 = a.lerp(&b, SimdFloat4::load1(0.2));
        expect_soa_quaternion_eq!(lerp_0_2, 0.565685416, 0.14142136, 0.0, 0.22961006,
                                0.0, 0.0, 0.70710677, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.76568544, 0.94142133, 0.70710677, 0.92387950);

        let lerp_m = a.lerp(&b, SimdFloat4::load(0.0, 1.0, 1.0, 0.2));
        expect_soa_quaternion_eq!(lerp_m, 0.70710677, 0.70710677, 0.0, 0.22961006, 0.0,
                                0.0, 0.70710677, 0.0, 0.0, 0.0, 0.0, 0.0, 0.70710677,
                                0.70710677, 0.70710677, 0.92387950);

        let nlerp_0 = a.nlerp(&b, SimdFloat4::zero());
        expect_soa_quaternion_eq!(nlerp_0, 0.70710677, 0.0, 0.0, 0.382683432, 0.0, 0.0,
                                0.70710677, 0.0, 0.0, 0.0, 0.0, 0.0, 0.70710677, 1.0,
                                0.70710677, 0.9238795);
        assert_eq!(nlerp_0.is_normalized().are_all_true(), true);

        let nlerp_1 = a.nlerp(&b, SimdFloat4::one());
        expect_soa_quaternion_eq!(nlerp_1, 0.0, 0.70710677, 0.0, -0.382683432, 0.0, 0.0,
                                0.70710677, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.70710677,
                                0.70710677, 0.9238795);
        assert_eq!(nlerp_1.is_normalized().are_all_true(), true);

        let nlerp_0_2 = a.nlerp(&b, SimdFloat4::load1(0.2));
        assert_eq!(nlerp_0_2.is_normalized().are_all_true(), true);
        expect_soa_quaternion_eq!(nlerp_0_2, 0.59421712, 0.14855431, 0.0, 0.24119100,
                                0.0, 0.0, 0.70710683, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.80430466, 0.98890430, 0.70710683, 0.97047764);
        assert_eq!(nlerp_0_2.is_normalized().are_all_true(), true);

        let nlerp_m = a.nlerp(&b, SimdFloat4::load(0.0, 1.0, 1.0, 0.2));
        expect_soa_quaternion_eq!(nlerp_m, 0.70710677, 0.70710677, 0.0, 0.24119100, 0.0,
                                0.0, 0.70710677, 0.0, 0.0, 0.0, 0.0, 0.0, 0.70710677,
                                0.70710677, 0.70710677, 0.97047764);
        assert_eq!(nlerp_m.is_normalized().are_all_true(), true);

        let nlerp_est_m = a.nlerp_est(&b, SimdFloat4::load(0.0, 1.0, 1.0, 0.2));
        expect_soa_quaternion_eq!(nlerp_est_m, 0.70710677, 0.70710677, 0.0,
                                    0.24119100, 0.0, 0.0, 0.70710677, 0.0, 0.0, 0.0,
                                    0.0, 0.0, 0.70710677, 0.70710677, 0.70710677,
                                    0.97047764);
        assert_eq!(nlerp_est_m.is_normalized_est().are_all_true(), true);
    }
}