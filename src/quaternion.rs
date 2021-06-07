/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::vec_float::*;
use std::ops::{Add, Mul, Neg};

#[derive(Clone)]
pub struct Quaternion {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Quaternion {
    // Constructs an uninitialized quaternion.
    #[inline]
    pub fn new_default() -> Quaternion {
        return Quaternion {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 0.0,
        };
    }

    // Constructs a quaternion from 4 floating point values.
    #[inline]
    pub fn new(_x: f32, _y: f32, _z: f32, _w: f32) -> Quaternion {
        return Quaternion {
            x: _x,
            y: _y,
            z: _z,
            w: _w,
        };
    }

    // Returns a normalized quaternion initialized from an axis angle
    // representation.
    // Assumes the axis part (x, y, z) of _axis_angle is normalized.
    // _angle.x is the angle in radian.
    #[inline]
    pub fn from_axis_angle(_axis: &Float3, _angle: f32) -> Quaternion {
        debug_assert!(is_normalized3(_axis) && "axis is not normalized.".parse().unwrap_or(true));
        let half_angle = _angle * 0.5;
        let half_sin = f32::sin(half_angle);
        let half_cos = f32::cos(half_angle);
        return Quaternion::new(_axis.x * half_sin, _axis.y * half_sin, _axis.z * half_sin,
                               half_cos);
    }

    // Returns a normalized quaternion initialized from an axis and angle cosine
    // representation.
    // Assumes the axis part (x, y, z) of _axis_angle is normalized.
    // _angle.x is the angle cosine in radian, it must be within [-1,1] range.
    #[inline]
    pub fn from_axis_cos_angle(_axis: &Float3, _cos: f32) -> Quaternion {
        debug_assert!(is_normalized3(_axis) && "axis is not normalized.".parse().unwrap_or(true));
        debug_assert!(_cos >= -1.0 && _cos <= 1.0 && "cos is not in [-1,1] range.".parse().unwrap_or(true));

        let half_cos2 = (1.0 + _cos) * 0.5;
        let half_sin = f32::sqrt(1.0 - half_cos2);
        return Quaternion::new(_axis.x * half_sin, _axis.y * half_sin, _axis.z * half_sin,
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
            quat = match f32::abs(_from.x) > f32::abs(_from.z) {
                true => Quaternion::new(-_from.y, _from.x, 0.0, 0.0),
                false => Quaternion::new(0.0, -_from.z, _from.y, 0.0)
            };
        } else {
            let cross = cross(_from, _to);
            quat = Quaternion::new(cross.x, cross.y, cross.z, real_part);
        }
        return normalize(&quat);
    }

    // Returns the quaternion that will rotate vector _from into vector _to,
    // around their plan perpendicular axis. The input vectors must be normalized.
    #[inline]
    pub fn from_unit_vectors(_from: &Float3, _to: &Float3) -> Quaternion {
        debug_assert!(is_normalized3(_from) && is_normalized3(_to) &&
            "Input vectors must be normalized.".parse().unwrap_or(true));

        // http://lolengine.net/blog/2014/02/24/quaternion-from-two-vectors-final
        let real_part = 1.0 + dot3(_from, _to);
        if real_part < 1.0e-6 {
            // If _from and _to are exactly opposite, rotate 180 degrees around an
            // arbitrary orthogonal axis.
            // Normalisation isn't needed, as from is already.
            return match f32::abs(_from.x) > f32::abs(_from.z) {
                true => Quaternion::new(-_from.y, _from.x, 0.0, 0.0),
                false => Quaternion::new(0.0, -_from.z, _from.y, 0.0)
            };
        } else {
            let cross = cross(_from, _to);
            return normalize(&Quaternion::new(cross.x, cross.y, cross.z, real_part));
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
        return self.x == other.x && self.y == other.y && self.z == other.z && self.w == other.w;
    }
}

// Returns the conjugate of _q. This is the same as the inverse if _q is
// normalized. Otherwise the magnitude of the inverse is 1.0/|_q|.
#[inline]
pub fn conjugate(_q: &Quaternion) -> Quaternion {
    return Quaternion::new(-_q.x, -_q.y, -_q.z, _q.w);
}

impl Add for Quaternion {
    type Output = Quaternion;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        return Quaternion::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z, self.w + rhs.w);
    }
}

impl Mul<f32> for Quaternion {
    type Output = Quaternion;
    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        return Quaternion::new(self.x * rhs, self.y * rhs, self.z * rhs, self.w * rhs);
    }
}

impl Mul for Quaternion {
    type Output = Quaternion;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        return Quaternion::new(self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
                               self.w * rhs.y + self.y * rhs.w + self.z * rhs.x - self.x * rhs.z,
                               self.w * rhs.z + self.z * rhs.w + self.x * rhs.y - self.y * rhs.x,
                               self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z);
    }
}

impl Neg for Quaternion {
    type Output = Quaternion;
    #[inline]
    fn neg(self) -> Self::Output {
        return Quaternion::new(-self.x, -self.y, -self.z, -self.w);
    }
}

// Returns true if the angle between _a and _b is less than _tolerance.
#[inline]
pub fn compare(_a: &Quaternion, _b: &Quaternion,
               _cos_half_tolerance: f32) -> bool {
    // Computes w component of a-1 * b.
    let cos_half_angle = _a.x * _b.x + _a.y * _b.y + _a.z * _b.z + _a.w * _b.w;
    return f32::abs(cos_half_angle) >= _cos_half_tolerance;
}

// Returns true if _q is a normalized quaternion.
#[inline]
pub fn is_normalized(_q: &Quaternion) -> bool {
    let sq_len = _q.x * _q.x + _q.y * _q.y + _q.z * _q.z + _q.w * _q.w;
    return f32::abs(sq_len - 1.0) < crate::math_constant::K_NORMALIZATION_TOLERANCE_SQ;
}

// Returns the normalized quaternion _q.
#[inline]
pub fn normalize(_q: &Quaternion) -> Quaternion {
    let sq_len = _q.x * _q.x + _q.y * _q.y + _q.z * _q.z + _q.w * _q.w;
    debug_assert!(sq_len != 0.0 && "_q is not normalizable".parse().unwrap_or(true));

    let inv_len = 1.0 / f32::sqrt(sq_len);
    return Quaternion::new(_q.x * inv_len, _q.y * inv_len, _q.z * inv_len,
                           _q.w * inv_len);
}

// Returns the normalized quaternion _q if the norm of _q is not 0.
// Otherwise returns _safer.
#[inline]
pub fn normalize_safe(_q: &Quaternion, _safer: &Quaternion) -> Quaternion {
    debug_assert!(is_normalized(_safer) && "_safer is not normalized".parse().unwrap_or(true));
    let sq_len = _q.x * _q.x + _q.y * _q.y + _q.z * _q.z + _q.w * _q.w;
    if sq_len == 0.0 {
        return _safer.clone();
    }
    let inv_len = 1.0 / f32::sqrt(sq_len);
    return Quaternion::new(_q.x * inv_len, _q.y * inv_len, _q.z * inv_len,
                           _q.w * inv_len);
}

// Returns to an axis angle representation of quaternion _q.
// Assumes quaternion _q is normalized.
#[inline]
pub fn to_axis_angle(_q: &Quaternion) -> Float4 {
    debug_assert!(is_normalized(_q));
    let clamped_w = f32::clamp(-1.0, _q.w, 1.0);
    let angle = 2.0 * f32::acos(clamped_w);
    let s = f32::sqrt(1.0 - clamped_w * clamped_w);

    // Assuming quaternion normalized then s always positive.
    return if s < 0.001 {  // Tests to avoid divide by zero.
        // If s close to zero then direction of axis is not important.
        Float4::new(1.0, 0.0, 0.0, angle)
    } else {
        // normalize axis
        let inv_s = 1.0 / s;
        Float4::new(_q.x * inv_s, _q.y * inv_s, _q.z * inv_s, angle)
    };
}

// Returns to an Euler representation of quaternion _q.
// Quaternion _q does not require to be normalized.
#[inline]
pub fn to_euler(_q: &Quaternion) -> Float3 {
    let sqw = _q.w * _q.w;
    let sqx = _q.x * _q.x;
    let sqy = _q.y * _q.y;
    let sqz = _q.z * _q.z;
    // If normalized is one, otherwise is correction factor.
    let unit = sqx + sqy + sqz + sqw;
    let test = _q.x * _q.y + _q.z * _q.w;
    let mut euler = Float3::new_default();
    if test > 0.499 * unit {  // Singularity at north pole
        euler.x = 2.0 * f32::atan2(_q.x, _q.w);
        euler.y = crate::math_constant::K_PI_2;
        euler.z = 0.0;
    } else if test < -0.499 * unit {  // Singularity at south pole
        euler.x = -2.0 * f32::atan2(_q.x, _q.w);
        euler.y = -crate::math_constant::K_PI_2;
        euler.z = 0.0;
    } else {
        euler.x = f32::atan2(2.0 * _q.y * _q.w - 2.0 * _q.x * _q.z,
                             sqx - sqy - sqz + sqw);
        euler.y = f32::asin(2.0 * test / unit);
        euler.z = f32::atan2(2.0 * _q.x * _q.w - 2.0 * _q.y * _q.z,
                             -sqx + sqy - sqz + sqw);
    }
    return euler;
}


// Returns the dot product of _a and _b.
#[inline]
pub fn dot(_a: &Quaternion, _b: &Quaternion) -> f32 {
    return _a.x * _b.x + _a.y * _b.y + _a.z * _b.z + _a.w * _b.w;
}


// Returns the linear interpolation of quaternion _a and _b with coefficient
// _f.
#[inline]
pub fn lerp(_a: &Quaternion, _b: &Quaternion, _f: f32) -> Quaternion {
    return Quaternion::new((_b.x - _a.x) * _f + _a.x, (_b.y - _a.y) * _f + _a.y,
                           (_b.z - _a.z) * _f + _a.z, (_b.w - _a.w) * _f + _a.w);
}

// Returns the linear interpolation of quaternion _a and _b with coefficient
// _f. _a and _n must be from the same hemisphere (aka dot(_a, _b) >= 0).
#[inline]
pub fn nlerp(_a: &Quaternion, _b: &Quaternion, _f: f32) -> Quaternion {
    let lerp = Float4::new((_b.x - _a.x) * _f + _a.x, (_b.y - _a.y) * _f + _a.y,
                           (_b.z - _a.z) * _f + _a.z, (_b.w - _a.w) * _f + _a.w);
    let sq_len =
        lerp.x * lerp.x + lerp.y * lerp.y + lerp.z * lerp.z + lerp.w * lerp.w;
    let inv_len = 1.0 / f32::sqrt(sq_len);
    return Quaternion::new(lerp.x * inv_len, lerp.y * inv_len, lerp.z * inv_len,
                           lerp.w * inv_len);
}

// Returns the spherical interpolation of quaternion _a and _b with
// coefficient _f.
#[inline]
pub fn slerp(_a: &Quaternion, _b: &Quaternion, _f: f32) -> Quaternion {
    debug_assert!(is_normalized(_a));
    debug_assert!(is_normalized(_b));
    // Calculate angle between them.
    let cos_half_theta = _a.x * _b.x + _a.y * _b.y + _a.z * _b.z + _a.w * _b.w;

    // If _a=_b or _a=-_b then theta = 0 and we can return _a.
    if f32::abs(cos_half_theta) >= 0.999 {
        return _a.clone();
    }

    // Calculate temporary values.
    let half_theta = f32::acos(cos_half_theta);
    let sin_half_theta = f32::sqrt(1.0 - cos_half_theta * cos_half_theta);

    // If theta = pi then result is not fully defined, we could rotate around
    // any axis normal to _a or _b.
    if sin_half_theta < 0.001 {
        return Quaternion::new((_a.x + _b.x) * 0.5, (_a.y + _b.y) * 0.5,
                               (_a.z + _b.z) * 0.5, (_a.w + _b.w) * 0.5);
    }

    let ratio_a = f32::sin((1.0 - _f) * half_theta) / sin_half_theta;
    let ratio_b = f32::sin(_f * half_theta) / sin_half_theta;

    // Calculate Quaternion.
    return Quaternion::new(
        ratio_a * _a.x + ratio_b * _b.x, ratio_a * _a.y + ratio_b * _b.y,
        ratio_a * _a.z + ratio_b * _b.z, ratio_a * _a.w + ratio_b * _b.w);
}

// Computes the transformation of a Quaternion and a vector _v.
// This is equivalent to carrying out the quaternion multiplications:
// _q.conjugate() * (*this) * _q
#[inline]
pub fn transform_vector(_q: &Quaternion, _v: &Float3) -> Float3 {
    // http://www.neil.dantam.name/note/dantam-quaternion.pdf
    // _v + 2.0 * cross(_q.xyz, cross(_q.xyz, _v) + _q.w * _v);
    let a = Float3::new(_q.y * _v.z - _q.z * _v.y + _v.x * _q.w,
                        _q.z * _v.x - _q.x * _v.z + _v.y * _q.w,
                        _q.x * _v.y - _q.y * _v.x + _v.z * _q.w);
    let b = Float3::new(_q.y * a.z - _q.z * a.y, _q.z * a.x - _q.x * a.z,
                        _q.x * a.y - _q.y * a.x);
    return Float3::new(_v.x + b.x + b.x, _v.y + b.y + b.y, _v.z + b.z + b.z);
}

//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
#[cfg(test)]
mod ozz_math {
    use crate::quaternion::*;
    use crate::math_test_helper::*;
    use crate::*;
    use crate::vec_float::Float3;

    #[test]
    fn quaternion_constant() {
        expect_quaternion_eq!(Quaternion::identity(), 0.0, 0.0, 0.0, 1.0);
    }

    #[test]
    fn quaternion_axis_angle() {
        // Expect assertions from invalid inputs
        // EXPECT_ASSERTION(Quaternion::FromAxisAngle(Float3::zero(), 0.0),
        //                  "axis is not normalized");
        // EXPECT_ASSERTION(ToAxisAngle(Quaternion(0.0, 0.0, 0.0, 2.0)), "IsNormalized");

        // Identity
        expect_quaternion_eq!(Quaternion::from_axis_angle(&Float3::y_axis(), 0.0), 0.0,
                             0.0, 0.0, 1.0);
        expect_float4_eq!(to_axis_angle(&Quaternion::identity()), 1.0, 0.0, 0.0, 0.0);

        // Other axis angles
        expect_quaternion_eq!(
            Quaternion::from_axis_angle(&Float3::y_axis(), crate::math_constant::K_PI_2), 0.0, 0.70710677, 0.0, 0.70710677);
        expect_float4_eq!(to_axis_angle(&Quaternion::new(0.0, 0.70710677, 0.0, 0.70710677)),
                         0.0, 1.0, 0.0, crate::math_constant::K_PI_2);

        expect_quaternion_eq!(
            Quaternion::from_axis_angle(&Float3::y_axis(), -crate::math_constant::K_PI_2), 0.0,
            -0.70710677, 0.0, 0.70710677);
        expect_quaternion_eq!(
            Quaternion::from_axis_angle(&-Float3::y_axis(), crate::math_constant::K_PI_2), 0.0,
            -0.70710677, 0.0, 0.70710677);
        expect_float4_eq!(to_axis_angle(&Quaternion::new(0.0, -0.70710677, 0.0, 0.70710677)),
                         0.0, -1.0, 0.0, crate::math_constant::K_PI_2);

        expect_quaternion_eq!(
            Quaternion::from_axis_angle(&Float3::y_axis(), 3.0 * crate::math_constant::K_PI_4), 0.0, 0.923879504, 0.0, 0.382683426);
        expect_float4_eq!(
            to_axis_angle(&Quaternion::new(0.0, 0.923879504, 0.0, 0.382683426)), 0.0, 1.0,
            0.0, 3.0 * crate::math_constant::K_PI_4);

        expect_quaternion_eq!(
            Quaternion::from_axis_angle(&Float3::new(0.819865, 0.033034, -0.571604), 1.123),
            0.4365425, 0.017589169, -0.30435428, 0.84645736);
        expect_float4_eq!(
            to_axis_angle(&Quaternion::new(0.4365425, 0.017589169, -0.30435428, 0.84645736)),
            0.819865, 0.033034, -0.571604, 1.123);
    }
}
