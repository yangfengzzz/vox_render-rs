/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::vec_float::*;
use std::ops::{Add, Mul, Neg};
use cgmath::{Zero, InnerSpace, VectorSpace, Rotation3, Rad};

#[derive(Clone)]
pub struct Quaternion {
    pub value: cgmath::Quaternion<f32>,
}

impl Quaternion {
    // Constructs an uninitialized quaternion.
    #[inline]
    pub fn new_default() -> Quaternion {
        return Quaternion {
            value: cgmath::Quaternion::zero()
        };
    }

    // Constructs a quaternion from 4 floating point values.
    #[inline]
    pub fn new(_x: f32, _y: f32, _z: f32, _w: f32) -> Quaternion {
        return Quaternion {
            value: cgmath::Quaternion::new(_w, _x, _y, _z)
        };
    }

    // Returns a normalized quaternion initialized from an axis angle
    // representation.
    // Assumes the axis part (x, y, z) of _axis_angle is normalized.
    // _angle.x is the angle in radian.
    #[inline]
    pub fn from_axis_angle(_axis: &Float3, _angle: f32) -> Quaternion {
        return Quaternion {
            value: cgmath::Quaternion::from_axis_angle(_axis.value, Rad(_angle))
        };
    }

    // Returns a normalized quaternion initialized from an axis and angle cosine
    // representation.
    // Assumes the axis part (x, y, z) of _axis_angle is normalized.
    // _angle.x is the angle cosine in radian, it must be within [-1,1] range.
    #[inline]
    pub fn from_axis_cos_angle(_axis: &Float3, _cos: f32) -> Quaternion {
        debug_assert!(is_normalized3(_axis) && "axis is not normalized.".parse().unwrap());
        debug_assert!(_cos >= -1.0 && _cos <= 1.0 && "cos is not in [-1,1] range.".parse().unwrap());

        let half_cos2 = (1.0 + _cos) * 0.5;
        let half_sin = f32::sqrt(1.0 - half_cos2);
        return Quaternion::new(_axis.value.x * half_sin, _axis.value.y * half_sin, _axis.value.z * half_sin,
                               f32::sqrt(half_cos2));
    }

    // Returns a normalized quaternion initialized from an Euler representation.
    // Euler angles are ordered Heading, Elevation and Bank, or Yaw, Pitch and
    // Roll.
    #[inline]
    pub fn from_euler(_yaw: f32, _pitch: f32, _roll: f32) -> Quaternion {
        let half_yaw = _yaw * 0.5;
        let c1 = f32::cos(half_yaw);
        let s1 = f32::sin(half_yaw);
        let half_pitch = _pitch * 0.5;
        let c2 = f32::cos(half_pitch);
        let s2 = f32::sin(half_pitch);
        let half_roll = _roll * 0.5;
        let c3 = f32::cos(half_roll);
        let s3 = f32::sin(half_roll);
        let c1c2 = c1 * c2;
        let s1s2 = s1 * s2;
        return Quaternion::new(c1c2 * s3 + s1s2 * c3, s1 * c2 * c3 + c1 * s2 * s3,
                               c1 * s2 * c3 - s1 * c2 * s3, c1c2 * c3 - s1s2 * s3);
    }

    // Returns the quaternion that will rotate vector _from into vector _to,
    // around their plan perpendicular axis.The input vectors don't need to be
    // normalized, they can be null as well.
    #[inline]
    pub fn from_vectors(_from: &Float3, _to: &Float3) -> Quaternion {
        // http://lolengine.net/blog/2014/02/24/quaternion-from-two-vectors-final

        let norm_from_norm_to = f32::sqrt(length_sqr3(_from) * length_sqr3(_to));
        if norm_from_norm_to < 1.0e-5 {
            return Quaternion::identity();
        }
        let real_part = norm_from_norm_to + dot3(_from, _to);
        let quat;
        if real_part < 1.0e-6 * norm_from_norm_to {
            // If _from and _to are exactly opposite, rotate 180 degrees around an
            // arbitrary orthogonal axis. Axis normalization can happen later, when we
            // normalize the quaternion.
            quat = match f32::abs(_from.value.x) > f32::abs(_from.value.z) {
                true => Quaternion::new(-_from.value.y, _from.value.x, 0.0, 0.0),
                false => Quaternion::new(0.0, -_from.value.z, _from.value.y, 0.0)
            };
        } else {
            let cross = cross(_from, _to);
            quat = Quaternion::new(cross.value.x, cross.value.y, cross.value.z, real_part);
        }
        return normalize(&quat);
    }

    // Returns the quaternion that will rotate vector _from into vector _to,
    // around their plan perpendicular axis. The input vectors must be normalized.
    #[inline]
    pub fn from_unit_vectors(_from: &Float3, _to: &Float3) -> Quaternion {
        debug_assert!(is_normalized3(_from) && is_normalized3(_to) &&
            "Input vectors must be normalized.".parse().unwrap());

        // http://lolengine.net/blog/2014/02/24/quaternion-from-two-vectors-final
        let real_part = 1.0 + dot3(_from, _to);
        if real_part < 1.0e-6 {
            // If _from and _to are exactly opposite, rotate 180 degrees around an
            // arbitrary orthogonal axis.
            // Normalisation isn't needed, as from is already.
            return match f32::abs(_from.value.x) > f32::abs(_from.value.z) {
                true => Quaternion::new(-_from.value.y, _from.value.x, 0.0, 0.0),
                false => Quaternion::new(0.0, -_from.value.z, _from.value.y, 0.0)
            };
        } else {
            let cross = cross(_from, _to);
            return normalize(&Quaternion::new(cross.value.x, cross.value.y, cross.value.z, real_part));
        }
    }

    // Returns the identity quaternion.
    #[inline]
    pub fn identity() -> Quaternion {
        return Quaternion::new(0.0, 0.0, 0.0, 1.0);
    }
}

impl PartialEq for Quaternion {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        return self.value.eq(&other.value);
    }
}

// Returns the conjugate of _q. This is the same as the inverse if _q is
// normalized. Otherwise the magnitude of the inverse is 1.f/|_q|.
#[inline]
pub fn conjugate(_q: &Quaternion) -> Quaternion {
    return Quaternion {
        value: _q.value.conjugate()
    };
}

impl Add for Quaternion {
    type Output = Quaternion;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        return Quaternion {
            value: self.value + rhs.value
        };
    }
}

impl Mul<f32> for Quaternion {
    type Output = Quaternion;
    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        return Quaternion {
            value: self.value * rhs
        };
    }
}

impl Mul for Quaternion {
    type Output = Quaternion;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        return Quaternion {
            value: self.value * rhs.value
        };
    }
}

impl Neg for Quaternion {
    type Output = Quaternion;
    #[inline]
    fn neg(self) -> Self::Output {
        return Quaternion {
            value: -self.value
        };
    }
}

// Returns true if the angle between _a and _b is less than _tolerance.
#[inline]
pub fn compare(_a: &Quaternion, _b: &Quaternion,
               _cos_half_tolerance: f32) -> bool {
    // Computes w component of a-1 * b.
    let cos_half_angle = _a.value.v.x * _b.value.v.x + _a.value.v.y * _b.value.v.y + _a.value.v.z * _b.value.v.z + _a.value.s * _b.value.s;
    return f32::abs(cos_half_angle) >= _cos_half_tolerance;
}

// Returns true if _q is a normalized quaternion.
#[inline]
pub fn is_normalized(_q: &Quaternion) -> bool {
    let sq_len = _q.value.v.x * _q.value.v.x + _q.value.v.y * _q.value.v.y + _q.value.v.z * _q.value.v.z + _q.value.s * _q.value.s;
    return f32::abs(sq_len - 1.0) < crate::math_constant::K_NORMALIZATION_TOLERANCE_SQ;
}

// Returns the normalized quaternion _q.
#[inline]
pub fn normalize(_q: &Quaternion) -> Quaternion {
    let sq_len = _q.value.v.x * _q.value.v.x + _q.value.v.y * _q.value.v.y + _q.value.v.z * _q.value.v.z + _q.value.s * _q.value.s;
    debug_assert!(sq_len != 0.0 && "_q is not normalizable".parse().unwrap());

    let inv_len = 1.0 / f32::sqrt(sq_len);
    return Quaternion::new(_q.value.v.x * inv_len, _q.value.v.y * inv_len, _q.value.v.z * inv_len,
                           _q.value.s * inv_len);
}

// Returns the normalized quaternion _q if the norm of _q is not 0.
// Otherwise returns _safer.
#[inline]
pub fn normalize_safe(_q: &Quaternion, _safer: &Quaternion) -> Quaternion {
    debug_assert!(is_normalized(_safer) && "_safer is not normalized".parse().unwrap());
    let sq_len = _q.value.v.x * _q.value.v.x + _q.value.v.y * _q.value.v.y + _q.value.v.z * _q.value.v.z + _q.value.s * _q.value.s;
    if sq_len == 0.0 {
        return _safer.clone();
    }
    let inv_len = 1.0 / f32::sqrt(sq_len);
    return Quaternion::new(_q.value.v.x * inv_len, _q.value.v.y * inv_len, _q.value.v.z * inv_len,
                           _q.value.s * inv_len);
}

// Returns to an axis angle representation of quaternion _q.
// Assumes quaternion _q is normalized.
#[inline]
pub fn to_axis_angle(_q: &Quaternion) -> Float4 {
    debug_assert!(is_normalized(_q));
    let clamped_w = f32::clamp(-1.0, _q.value.s, 1.0);
    let angle = 2.0 * f32::acos(clamped_w);
    let s = f32::sqrt(1.0 - clamped_w * clamped_w);

    // Assuming quaternion normalized then s always positive.
    return if s < 0.001 {  // Tests to avoid divide by zero.
        // If s close to zero then direction of axis is not important.
        Float4::new(1.0, 0.0, 0.0, angle)
    } else {
        // normalize axis
        let inv_s = 1.0 / s;
        Float4::new(_q.value.v.x * inv_s, _q.value.v.y * inv_s, _q.value.v.z * inv_s, angle)
    };
}

// Returns to an Euler representation of quaternion _q.
// Quaternion _q does not require to be normalized.
#[inline]
pub fn to_euler(_q: &Quaternion) -> Float3 {
    let sqw = _q.value.s * _q.value.s;
    let sqx = _q.value.v.x * _q.value.v.x;
    let sqy = _q.value.v.y * _q.value.v.y;
    let sqz = _q.value.v.z * _q.value.v.z;
    // If normalized is one, otherwise is correction factor.
    let unit = sqx + sqy + sqz + sqw;
    let test = _q.value.v.x * _q.value.v.y + _q.value.v.z * _q.value.s;
    let mut euler = Float3::new_default();
    if test > 0.499 * unit {  // Singularity at north pole
        euler.value.x = 2.0 * f32::atan2(_q.value.v.x, _q.value.s);
        euler.value.y = crate::math_constant::K_PI_2;
        euler.value.z = 0.0;
    } else if test < -0.499 * unit {  // Singularity at south pole
        euler.value.x = -2.0 * f32::atan2(_q.value.v.x, _q.value.s);
        euler.value.y = -crate::math_constant::K_PI_2;
        euler.value.z = 0.0;
    } else {
        euler.value.x = f32::atan2(2.0 * _q.value.v.y * _q.value.s - 2.0 * _q.value.v.x * _q.value.v.z,
                                   sqx - sqy - sqz + sqw);
        euler.value.y = f32::asin(2.0 * test / unit);
        euler.value.z = f32::atan2(2.0 * _q.value.v.x * _q.value.s - 2.0 * _q.value.v.y * _q.value.v.z,
                                   -sqx + sqy - sqz + sqw);
    }
    return euler;
}


// Returns the dot product of _a and _b.
#[inline]
pub fn dot(_a: &Quaternion, _b: &Quaternion) -> f32 {
    return _a.value.dot(_b.value);
}


// Returns the linear interpolation of quaternion _a and _b with coefficient
// _f.
#[inline]
pub fn lerp(_a: &Quaternion, _b: &Quaternion, _f: f32) -> Quaternion {
    return Quaternion {
        value: _a.value.lerp(_b.value, _f)
    };
}

// Returns the linear interpolation of quaternion _a and _b with coefficient
// _f. _a and _n must be from the same hemisphere (aka dot(_a, _b) >= 0).
#[inline]
pub fn nlerp(_a: &Quaternion, _b: &Quaternion, _f: f32) -> Quaternion {
    return Quaternion {
        value: _a.value.nlerp(_b.value, _f)
    };
}

// Returns the spherical interpolation of quaternion _a and _b with
// coefficient _f.
#[inline]
pub fn slerp(_a: &Quaternion, _b: &Quaternion, _f: f32) -> Quaternion {
    return Quaternion {
        value: _a.value.slerp(_b.value, _f)
    };
}

// Computes the transformation of a Quaternion and a vector _v.
// This is equivalent to carrying out the quaternion multiplications:
// _q.conjugate() * (*this) * _q
#[inline]
pub fn transform_vector(_q: &Quaternion, _v: &Float3) -> Float3 {
    // http://www.neil.dantam.name/note/dantam-quaternion.pdf
    // _v + 2.f * cross(_q.xyz, cross(_q.xyz, _v) + _q.w * _v);
    let a = Float3::new(_q.value.v.y * _v.value.z - _q.value.v.z * _v.value.y + _v.value.x * _q.value.s,
                        _q.value.v.z * _v.value.x - _q.value.v.x * _v.value.z + _v.value.y * _q.value.s,
                        _q.value.v.x * _v.value.y - _q.value.v.y * _v.value.x + _v.value.z * _q.value.s);
    let b = Float3::new(_q.value.v.y * a.value.z - _q.value.v.z * a.value.y,
                        _q.value.v.z * a.value.x - _q.value.v.x * a.value.z,
                        _q.value.v.x * a.value.y - _q.value.v.y * a.value.x);

    return Float3::new(_v.value.x + b.value.x + b.value.x,
                       _v.value.y + b.value.y + b.value.y,
                       _v.value.z + b.value.z + b.value.z);
}

