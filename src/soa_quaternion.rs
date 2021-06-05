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
// Returns the conjugate of _q. This is the same as the inverse if _q is
// normalized. Otherwise the magnitude of the inverse is 1.f/|_q|.
#[inline]
pub fn conjugate(_q: &SoaQuaternion) -> SoaQuaternion {
    return SoaQuaternion { x: -_q.x, y: -_q.y, z: -_q.z, w: _q.w };
}

impl Neg for SoaQuaternion {
    type Output = SoaQuaternion;
    #[inline]
    fn neg(self) -> Self::Output {
        return SoaQuaternion { x: -self.x, y: -self.y, z: -self.z, w: -self.w };
    }
}

// Returns the 4D dot product of quaternion _a and _b.
#[inline]
pub fn dot(_a: &SoaQuaternion, _b: &SoaQuaternion) -> SimdFloat4 {
    return _a.x * _b.x + _a.y * _b.y + _a.z * _b.z + _a.w * _b.w;
}

// Returns the normalized SoaQuaternion _q.
#[inline]
pub fn normalize(_q: &SoaQuaternion) -> SoaQuaternion {
    let len2 = _q.x * _q.x + _q.y * _q.y + _q.z * _q.z + _q.w * _q.w;
    let inv_len = SimdFloat4::load(1.0, 1.0, 1.0, 1.0) / (len2).sqrt();
    return SoaQuaternion {
        x: _q.x * inv_len,
        y: _q.y * inv_len,
        z: _q.z * inv_len,
        w: _q.w * inv_len,
    };
}

// Returns the estimated normalized SoaQuaternion _q.
#[inline]
pub fn normalize_est(_q: &SoaQuaternion) -> SoaQuaternion {
    let len2 = _q.x * _q.x + _q.y * _q.y + _q.z * _q.z + _q.w * _q.w;
    // Uses RSqrtEstNR (with one more Newton-Raphson step) as quaternions loose
    // much precision due to normalization.
    let inv_len = len2.rsqrt_est_nr();
    return SoaQuaternion {
        x: _q.x * inv_len,
        y: _q.y * inv_len,
        z: _q.z * inv_len,
        w: _q.w * inv_len,
    };
}

// Test if each quaternion of _q is normalized.
#[inline]
pub fn is_normalized(_q: &SoaQuaternion) -> SimdInt4 {
    let len2 = _q.x * _q.x + _q.y * _q.y + _q.z * _q.z + _q.w * _q.w;
    return (len2 - SimdFloat4::load(1.0, 1.0, 1.0, 1.0)).abs().cmp_lt(SimdFloat4::load(K_NORMALIZATION_TOLERANCE_SQ,
                                                                                       K_NORMALIZATION_TOLERANCE_SQ,
                                                                                       K_NORMALIZATION_TOLERANCE_SQ,
                                                                                       K_NORMALIZATION_TOLERANCE_SQ));
}


// Test if each quaternion of _q is normalized. using estimated tolerance.
#[inline]
pub fn is_normalized_est(_q: &SoaQuaternion) -> SimdInt4 {
    let len2 = _q.x * _q.x + _q.y * _q.y + _q.z * _q.z + _q.w * _q.w;
    return (len2 - SimdFloat4::load(1.0, 1.0, 1.0, 1.0)).abs().cmp_lt(SimdFloat4::load(K_NORMALIZATION_TOLERANCE_EST_SQ,
                                                                                       K_NORMALIZATION_TOLERANCE_EST_SQ,
                                                                                       K_NORMALIZATION_TOLERANCE_EST_SQ,
                                                                                       K_NORMALIZATION_TOLERANCE_EST_SQ));
}

// Returns the linear interpolation of SoaQuaternion _a and _b with coefficient
// _f.
#[inline]
pub fn lerp(_a: &SoaQuaternion, _b: &SoaQuaternion,
            _f: SimdFloat4) -> SoaQuaternion {
    return SoaQuaternion {
        x: (_b.x - _a.x) * _f + _a.x,
        y: (_b.y - _a.y) * _f + _a.y,
        z: (_b.z - _a.z) * _f + _a.z,
        w: (_b.w - _a.w) * _f + _a.w,
    };
}

// Returns the linear interpolation of SoaQuaternion _a and _b with coefficient
// _f.
#[inline]
pub fn nlerp(_a: &SoaQuaternion, _b: &SoaQuaternion, _f: SimdFloat4) -> SoaQuaternion {
    let lerp = SoaFloat4::load((_b.x - _a.x) * _f + _a.x, (_b.y - _a.y) * _f + _a.y,
                               (_b.z - _a.z) * _f + _a.z, (_b.w - _a.w) * _f + _a.w);
    let len2 = lerp.x * lerp.x + lerp.y * lerp.y + lerp.z * lerp.z + lerp.w * lerp.w;
    let inv_len = SimdFloat4::load(1.0, 1.0, 1.0, 1.0) / (len2).sqrt();
    return SoaQuaternion {
        x: lerp.x * inv_len,
        y: lerp.y * inv_len,
        z: lerp.z * inv_len,
        w: lerp.w * inv_len,
    };
}

// Returns the estimated linear interpolation of SoaQuaternion _a and _b with
// coefficient _f.
#[inline]
pub fn nlerp_est(_a: &SoaQuaternion, _b: &SoaQuaternion, _f: SimdFloat4) -> SoaQuaternion {
    let lerp = SoaFloat4::load((_b.x - _a.x) * _f + _a.x, (_b.y - _a.y) * _f + _a.y,
                               (_b.z - _a.z) * _f + _a.z, (_b.w - _a.w) * _f + _a.w);
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