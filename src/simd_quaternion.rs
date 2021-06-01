/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use packed_simd_2::*;
use std::ops::{Mul, BitXor, Neg};

// Declare the Quaternion type.
pub struct SimdQuaternion {
    pub xyzw: f32x4,
}

impl SimdQuaternion {
    // Returns the identity quaternion.
    #[inline]
    pub fn identity() -> SimdQuaternion {
        return SimdQuaternion { xyzw: f32x4::new(0.0, 0.0, 0.0, 1.0) };
    }

    // the angle in radian.
    #[inline]
    pub fn from_axis_angle(_axis: f32x4,
                           _angle: f32x4) -> SimdQuaternion {
        todo!()
    }

    // Returns a normalized quaternion initialized from an axis and angle cosine
    // representation.
    // Assumes the axis part (x, y, z) of _axis_angle is normalized.
    // _angle.x is the angle cosine in radian, it must be within [-1,1] range.
    #[inline]
    pub fn from_axis_cos_angle(_axis: f32x4,
                               _cos: f32x4) -> SimdQuaternion {
        todo!()
    }

    // Returns the quaternion that will rotate vector _from into vector _to,
    // around their plan perpendicular axis.The input vectors don't need to be
    // normalized, they can be null also.
    #[inline]
    pub fn from_vectors(_from: f32x4,
                        _to: f32x4) -> SimdQuaternion {
        todo!()
    }

    // Returns the quaternion that will rotate vector _from into vector _to,
    // around their plan perpendicular axis. The input vectors must be normalized.
    #[inline]
    pub fn from_unit_vectors(_from: f32x4,
                             _to: f32x4) -> SimdQuaternion {
        todo!()
    }
}

impl Mul for SimdQuaternion {
    type Output = SimdQuaternion;

    fn mul(self, rhs: Self) -> Self::Output {
        todo!("Check it!");
        // Original quaternion multiplication can be swizzled in a simd friendly way
        // if w is negated, and some w multiplications parts (1st/last) are swaped.
        //
        //        p1            p2            p3            p4
        //    _a.w * _b.x + _a.x * _b.w + _a.y * _b.z - _a.z * _b.y
        //    _a.w * _b.y + _a.y * _b.w + _a.z * _b.x - _a.x * _b.z
        //    _a.w * _b.z + _a.z * _b.w + _a.x * _b.y - _a.y * _b.x
        //    _a.w * _b.w - _a.x * _b.x - _a.y * _b.y - _a.z * _b.z
        // ... becomes ->
        //    _a.w * _b.x + _a.x * _b.w + _a.y * _b.z - _a.z * _b.y
        //    _a.w * _b.y + _a.y * _b.w + _a.z * _b.x - _a.x * _b.z
        //    _a.w * _b.z + _a.z * _b.w + _a.x * _b.y - _a.y * _b.x
        // - (_a.z * _b.z + _a.x * _b.x + _a.y * _b.y - _a.w * _b.w)
        let p1 = self.xyzw.shuffle1_dyn(u32x4::new(3, 3, 3, 2))
            * rhs.xyzw.shuffle1_dyn(u32x4::new(0, 1, 2, 2));
        let p2 = self.xyzw.shuffle1_dyn(u32x4::new(0, 1, 2, 0))
            * rhs.xyzw.shuffle1_dyn(u32x4::new(3, 3, 3, 0));
        let p13 = self.xyzw.shuffle1_dyn(u32x4::new(1, 2, 0, 1))
            .mul_add(rhs.xyzw.shuffle1_dyn(u32x4::new(2, 0, 1, 1)), p1);
        let p24 = self.xyzw.shuffle1_dyn(u32x4::new(2, 0, 1, 3))
            .mul_add(rhs.xyzw.shuffle1_dyn(u32x4::new(1, 2, 0, 3)), p2);
        return SimdQuaternion {
            xyzw: f32x4::from_cast(u32x4::from_cast(p13 + p24)
                .bitxor(u32x4::new(0, 0, 0, 1)))
        };
    }
}

// Returns the conjugate of _q. This is the same as the inverse if _q is
// normalized. Otherwise the magnitude of the inverse is 1.f/|_q|.
pub fn conjugate(_q: &SimdQuaternion) -> SimdQuaternion {
    return SimdQuaternion {
        xyzw: f32x4::from_cast(u32x4::from_cast(_q.xyzw)
            .bitxor(u32x4::new(1, 1, 1, 0)))
    };
}

impl Neg for SimdQuaternion {
    type Output = SimdQuaternion;

    fn neg(self) -> Self::Output {
        return SimdQuaternion {
            xyzw: f32x4::from_cast(u32x4::from_cast(self.xyzw)
                .bitxor(u32x4::new(1, 1, 1, 1)))
        };
    }
}

// Returns the normalized quaternion _q.
pub fn normalize(_q: &SimdQuaternion) -> SimdQuaternion {
    todo!()
}

// Returns the normalized quaternion _q if the norm of _q is not 0.
// Otherwise returns _safer.
pub fn normalize_safe(_q: &SimdQuaternion,
                      _safer: &SimdQuaternion) -> SimdQuaternion {
    todo!()
}

// Returns the estimated normalized quaternion _q.
pub fn normalize_est(_q: &SimdQuaternion) -> SimdQuaternion {
    todo!()
}

// Returns the estimated normalized quaternion _q if the norm of _q is not 0.
// Otherwise returns _safer.
pub fn normalize_safe_est(_q: &SimdQuaternion,
                          _safer: &SimdQuaternion) -> SimdQuaternion {
    todo!()
}