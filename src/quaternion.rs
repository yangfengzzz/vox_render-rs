/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::vec_float::*;
use std::ops::{Add, Mul, Neg};

#[derive(Clone, Copy)]
pub struct Quaternion {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl FloatType for Quaternion {
    type ImplType = Quaternion;
    #[inline]
    fn new_default() -> Self::ImplType {
        return Quaternion {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 0.0,
        };
    }
}

impl Quaternion {
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

    #[inline]
    pub fn to_vec(&self) -> [f32; 4] {
        return [self.x, self.y, self.z, self.w];
    }

    // Returns a normalized quaternion initialized from an axis angle
    // representation.
    // Assumes the axis part (x, y, z) of _axis_angle is normalized.
    // _angle.x is the angle in radian.
    #[inline]
    pub fn from_axis_angle(_axis: &Float3, _angle: f32) -> Quaternion {
        debug_assert!(_axis.is_normalized() && "axis is not normalized.".parse().unwrap_or(true));
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
        debug_assert!(_axis.is_normalized() && "axis is not normalized.".parse().unwrap_or(true));
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

        let norm_from_norm_to = f32::sqrt(_from.length_sqr() * _to.length_sqr());
        if norm_from_norm_to < 1.0e-5 {
            return Quaternion::identity();
        }
        let real_part = norm_from_norm_to + _from.dot(_to);
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
            let cross = _from.cross(_to);
            quat = Quaternion::new(cross.x, cross.y, cross.z, real_part);
        }
        return quat.normalize();
    }

    // Returns the quaternion that will rotate vector _from into vector _to,
    // around their plan perpendicular axis. The input vectors must be normalized.
    #[inline]
    pub fn from_unit_vectors(_from: &Float3, _to: &Float3) -> Quaternion {
        debug_assert!(_from.is_normalized() && _to.is_normalized() &&
            "Input vectors must be normalized.".parse().unwrap_or(true));

        // http://lolengine.net/blog/2014/02/24/quaternion-from-two-vectors-final
        let real_part = 1.0 + _from.dot(_to);
        if real_part < 1.0e-6 {
            // If _from and _to are exactly opposite, rotate 180 degrees around an
            // arbitrary orthogonal axis.
            // Normalisation isn't needed, as from is already.
            return match f32::abs(_from.x) > f32::abs(_from.z) {
                true => Quaternion::new(-_from.y, _from.x, 0.0, 0.0),
                false => Quaternion::new(0.0, -_from.z, _from.y, 0.0)
            };
        } else {
            let cross = _from.cross(_to);
            return Quaternion::new(cross.x, cross.y, cross.z, real_part).normalize();
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

macro_rules! impl_neg {
    ($type:ty) => {
        impl Neg for $type {
            type Output = Quaternion;
            #[inline]
            fn neg(self) -> Self::Output {
                return Quaternion::new(-self.x, -self.y, -self.z, -self.w);
            }
        }
    };
}
impl_neg!(Quaternion);
impl_neg!(&Quaternion);
//--------------------------------------------------------------------------------------------------
impl Quaternion {
    // Returns the conjugate of self. This is the same as the inverse if self is
    // normalized. Otherwise the magnitude of the inverse is 1.0/|self|.
    #[inline]
    pub fn conjugate(&self) -> Quaternion {
        return Quaternion::new(-self.x, -self.y, -self.z, self.w);
    }

    // Returns true if the angle between _a and _b is less than _tolerance.
    #[inline]
    pub fn compare(&self, _b: &Quaternion,
                   _cos_half_tolerance: f32) -> bool {
        // Computes w component of a-1 * b.
        let cos_half_angle = self.x * _b.x + self.y * _b.y + self.z * _b.z + self.w * _b.w;
        return f32::abs(cos_half_angle) >= _cos_half_tolerance;
    }

    // Returns true if self is a normalized quaternion.
    #[inline]
    pub fn is_normalized(&self) -> bool {
        let sq_len = self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w;
        return f32::abs(sq_len - 1.0) < crate::math_constant::K_NORMALIZATION_TOLERANCE_SQ;
    }

    // Returns the normalized quaternion self.
    #[inline]
    pub fn normalize(&self) -> Quaternion {
        let sq_len = self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w;
        debug_assert!(sq_len != 0.0 && "self is not normalizable".parse().unwrap_or(true));

        let inv_len = 1.0 / f32::sqrt(sq_len);
        return Quaternion::new(self.x * inv_len, self.y * inv_len, self.z * inv_len,
                               self.w * inv_len);
    }

    // Returns the normalized quaternion self if the norm of self is not 0.
    // Otherwise returns _safer.
    #[inline]
    pub fn normalize_safe(&self, _safer: &Quaternion) -> Quaternion {
        debug_assert!(_safer.is_normalized() && "_safer is not normalized".parse().unwrap_or(true));
        let sq_len = self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w;
        if sq_len == 0.0 {
            return _safer.clone();
        }
        let inv_len = 1.0 / f32::sqrt(sq_len);
        return Quaternion::new(self.x * inv_len, self.y * inv_len, self.z * inv_len,
                               self.w * inv_len);
    }

    // Returns to an axis angle representation of quaternion self.
    // Assumes quaternion self is normalized.
    #[inline]
    pub fn to_axis_angle(&self) -> Float4 {
        debug_assert!(self.is_normalized());
        let clamped_w = f32::clamp(-1.0, self.w, 1.0);
        let angle = 2.0 * f32::acos(clamped_w);
        let s = f32::sqrt(1.0 - clamped_w * clamped_w);

        // Assuming quaternion normalized then s always positive.
        return if s < 0.001 {  // Tests to avoid divide by zero.
            // If s close to zero then direction of axis is not important.
            Float4::new(1.0, 0.0, 0.0, angle)
        } else {
            // normalize axis
            let inv_s = 1.0 / s;
            Float4::new(self.x * inv_s, self.y * inv_s, self.z * inv_s, angle)
        };
    }

    // Returns to an Euler representation of quaternion self.
    // Quaternion self does not require to be normalized.
    #[inline]
    pub fn to_euler(&self) -> Float3 {
        let sqw = self.w * self.w;
        let sqx = self.x * self.x;
        let sqy = self.y * self.y;
        let sqz = self.z * self.z;
        // If normalized is one, otherwise is correction factor.
        let unit = sqx + sqy + sqz + sqw;
        let test = self.x * self.y + self.z * self.w;
        let mut euler = Float3::new_default();
        if test > 0.499 * unit {  // Singularity at north pole
            euler.x = 2.0 * f32::atan2(self.x, self.w);
            euler.y = crate::math_constant::K_PI_2;
            euler.z = 0.0;
        } else if test < -0.499 * unit {  // Singularity at south pole
            euler.x = -2.0 * f32::atan2(self.x, self.w);
            euler.y = -crate::math_constant::K_PI_2;
            euler.z = 0.0;
        } else {
            euler.x = f32::atan2(2.0 * self.y * self.w - 2.0 * self.x * self.z,
                                 sqx - sqy - sqz + sqw);
            euler.y = f32::asin(2.0 * test / unit);
            euler.z = f32::atan2(2.0 * self.x * self.w - 2.0 * self.y * self.z,
                                 -sqx + sqy - sqz + sqw);
        }
        return euler;
    }


    // Returns the dot product of self and _b.
    #[inline]
    pub fn dot(&self, _b: &Quaternion) -> f32 {
        return self.x * _b.x + self.y * _b.y + self.z * _b.z + self.w * _b.w;
    }


    // Returns the linear interpolation of quaternion self and _b with coefficient _f.
    #[inline]
    pub fn lerp(&self, _b: &Quaternion, _f: f32) -> Quaternion {
        return Quaternion::new((_b.x - self.x) * _f + self.x, (_b.y - self.y) * _f + self.y,
                               (_b.z - self.z) * _f + self.z, (_b.w - self.w) * _f + self.w);
    }

    // Returns the linear interpolation of quaternion self and _b with coefficient
    // _f. self and _n must be from the same hemisphere (aka dot(self, _b) >= 0).
    #[inline]
    pub fn nlerp(&self, _b: &Quaternion, _f: f32) -> Quaternion {
        let lerp = Float4::new((_b.x - self.x) * _f + self.x, (_b.y - self.y) * _f + self.y,
                               (_b.z - self.z) * _f + self.z, (_b.w - self.w) * _f + self.w);
        let sq_len =
            lerp.x * lerp.x + lerp.y * lerp.y + lerp.z * lerp.z + lerp.w * lerp.w;
        let inv_len = 1.0 / f32::sqrt(sq_len);
        return Quaternion::new(lerp.x * inv_len, lerp.y * inv_len, lerp.z * inv_len,
                               lerp.w * inv_len);
    }

    // Returns the spherical interpolation of quaternion self and _b with coefficient _f.
    #[inline]
    pub fn slerp(&self, _b: &Quaternion, _f: f32) -> Quaternion {
        debug_assert!(self.is_normalized());
        debug_assert!(_b.is_normalized());
        // Calculate angle between them.
        let cos_half_theta = self.x * _b.x + self.y * _b.y + self.z * _b.z + self.w * _b.w;

        // If self=_b or self=-_b then theta = 0 and we can return self.
        if f32::abs(cos_half_theta) >= 0.999 {
            return self.clone();
        }

        // Calculate temporary values.
        let half_theta = f32::acos(cos_half_theta);
        let sin_half_theta = f32::sqrt(1.0 - cos_half_theta * cos_half_theta);

        // If theta = pi then result is not fully defined, we could rotate around
        // any axis normal to self or _b.
        if sin_half_theta < 0.001 {
            return Quaternion::new((self.x + _b.x) * 0.5, (self.y + _b.y) * 0.5,
                                   (self.z + _b.z) * 0.5, (self.w + _b.w) * 0.5);
        }

        let ratio_a = f32::sin((1.0 - _f) * half_theta) / sin_half_theta;
        let ratio_b = f32::sin(_f * half_theta) / sin_half_theta;

        // Calculate Quaternion.
        return Quaternion::new(
            ratio_a * self.x + ratio_b * _b.x, ratio_a * self.y + ratio_b * _b.y,
            ratio_a * self.z + ratio_b * _b.z, ratio_a * self.w + ratio_b * _b.w);
    }

    // Computes the transformation of a Quaternion and a vector _v.
    // This is equivalent to carrying out the quaternion multiplications:
    // self.conjugate() * (*this) * self
    #[inline]
    pub fn transform_vector(&self, _v: &Float3) -> Float3 {
        // http://www.neil.dantam.name/note/dantam-quaternion.pdf
        // _v + 2.0 * cross(self.xyz, cross(self.xyz, _v) + self.w * _v);
        let a = Float3::new(self.y * _v.z - self.z * _v.y + _v.x * self.w,
                            self.z * _v.x - self.x * _v.z + _v.y * self.w,
                            self.x * _v.y - self.y * _v.x + _v.z * self.w);
        let b = Float3::new(self.y * a.z - self.z * a.y, self.z * a.x - self.x * a.z,
                            self.x * a.y - self.y * a.x);
        return Float3::new(_v.x + b.x + b.x, _v.y + b.y + b.y, _v.z + b.z + b.z);
    }
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
        // EXPECT_ASSERTION(ToAxisAngle(Quaternion(0.0, 0.0, 0.0, 2.0)), "is_normalized");

        // Identity
        expect_quaternion_eq!(Quaternion::from_axis_angle(&Float3::y_axis(), 0.0), 0.0,
                             0.0, 0.0, 1.0);
        expect_float4_eq!(Quaternion::identity().to_axis_angle(), 1.0, 0.0, 0.0, 0.0);

        // Other axis angles
        expect_quaternion_eq!(
            Quaternion::from_axis_angle(&Float3::y_axis(), crate::math_constant::K_PI_2), 0.0, 0.70710677, 0.0, 0.70710677);
        expect_float4_eq!(Quaternion::new(0.0, 0.70710677, 0.0, 0.70710677).to_axis_angle(),
                         0.0, 1.0, 0.0, crate::math_constant::K_PI_2);

        expect_quaternion_eq!(
            Quaternion::from_axis_angle(&Float3::y_axis(), -crate::math_constant::K_PI_2), 0.0,
            -0.70710677, 0.0, 0.70710677);
        expect_quaternion_eq!(
            Quaternion::from_axis_angle(&-Float3::y_axis(), crate::math_constant::K_PI_2), 0.0,
            -0.70710677, 0.0, 0.70710677);
        expect_float4_eq!(Quaternion::new(0.0, -0.70710677, 0.0, 0.70710677).to_axis_angle(),
                         0.0, -1.0, 0.0, crate::math_constant::K_PI_2);

        expect_quaternion_eq!(
            Quaternion::from_axis_angle(&Float3::y_axis(), 3.0 * crate::math_constant::K_PI_4), 0.0, 0.923879504, 0.0, 0.382683426);
        expect_float4_eq!(
            Quaternion::new(0.0, 0.923879504, 0.0, 0.382683426).to_axis_angle(), 0.0, 1.0,
            0.0, 3.0 * crate::math_constant::K_PI_4);

        expect_quaternion_eq!(
            Quaternion::from_axis_angle(&Float3::new(0.819865, 0.033034, -0.571604), 1.123),
            0.4365425, 0.017589169, -0.30435428, 0.84645736);
        expect_float4_eq!(
            Quaternion::new(0.4365425, 0.017589169, -0.30435428, 0.84645736).to_axis_angle(),
            0.819865, 0.033034, -0.571604, 1.123);
    }

    #[test]
    fn quaternion_axis_cos_angle() {
        // Identity
        expect_quaternion_eq!(Quaternion::from_axis_cos_angle(&Float3::y_axis(), 1.0), 0.0,
                             0.0, 0.0, 1.0);

        // Other axis angles
        expect_quaternion_eq!(Quaternion::from_axis_cos_angle(&Float3::y_axis(), f32::cos(crate::math_constant::K_PI_2)),
                             0.0, 0.70710677, 0.0, 0.70710677);
        expect_quaternion_eq!(Quaternion::from_axis_cos_angle(&-Float3::y_axis(), f32::cos(crate::math_constant::K_PI_2)),
                             0.0, -0.70710677, 0.0, 0.70710677);

        expect_quaternion_eq!(Quaternion::from_axis_cos_angle(&Float3::y_axis(), f32::cos(3.0 * crate::math_constant::K_PI_4)),
                             0.0, 0.923879504, 0.0, 0.382683426);

        expect_quaternion_eq!(Quaternion::from_axis_cos_angle(&Float3::new(0.819865, 0.033034, -0.571604), f32::cos(1.123)),
                            0.4365425, 0.017589169, -0.30435428, 0.84645736);
    }

    #[test]
    fn quaternion_quaternion_euler() {
        // Identity
        expect_quaternion_eq!(Quaternion::from_euler(0.0, 0.0, 0.0), 0.0, 0.0, 0.0, 1.0);
        expect_float3_eq!(Quaternion::identity().to_euler(), 0.0, 0.0, 0.0);

        // Heading
        expect_quaternion_eq!(Quaternion::from_euler(crate::math_constant::K_PI_2, 0.0, 0.0), 0.0, 0.70710677, 0.0,0.70710677);
        expect_float3_eq!(Quaternion::new(0.0,0.70710677, 0.0,0.70710677).to_euler(), crate::math_constant::K_PI_2, 0.0, 0.0);

        // Elevation
        expect_quaternion_eq!(Quaternion::from_euler(0.0, crate::math_constant::K_PI_2, 0.0), 0.0, 0.0,0.70710677,0.70710677);
        expect_float3_eq!(Quaternion::new(0.0, 0.0,0.70710677,0.70710677).to_euler(), 0.0, crate::math_constant::K_PI_2, 0.0);

        // Bank
        expect_quaternion_eq!(Quaternion::from_euler(0.0, 0.0, crate::math_constant::K_PI_2), 0.70710677, 0.0, 0.0,0.70710677);
        expect_float3_eq!(Quaternion::new(0.70710677, 0.0, 0.0,0.70710677).to_euler(), 0.0, 0.0, crate::math_constant::K_PI_2);

        // Any rotation
        expect_quaternion_eq!(Quaternion::from_euler(crate::math_constant::K_PI / 4.0, -crate::math_constant::K_PI / 6.0, crate::math_constant::K_PI_2),
                            0.56098551,0.092295974, -0.43045932,0.70105737);
        expect_float3_eq!(Quaternion::new(0.56098551,0.092295974, -0.43045932,0.70105737).to_euler(),
                            crate::math_constant::K_PI / 4.0, -crate::math_constant::K_PI / 6.0, crate::math_constant::K_PI_2);
    }

    #[test]
    fn quaternion_from_vectors() {
        // Returns identity for a 0 length vector
        expect_quaternion_eq!(
            Quaternion::from_vectors(&Float3::zero(), &Float3::x_axis()), 0.0, 0.0, 0.0, 1.0);

        // pi/2 around y
        expect_quaternion_eq!(
            Quaternion::from_vectors(&Float3::z_axis(), &Float3::x_axis()), 0.0, 0.707106769, 0.0, 0.707106769);

        // Non unit pi/2 around y
        expect_quaternion_eq!(
            Quaternion::from_vectors(&(Float3::z_axis() * 7.0), &Float3::x_axis()), 0.0, 0.707106769, 0.0, 0.707106769);

        // Minus pi/2 around y
        expect_quaternion_eq!(
            Quaternion::from_vectors(&Float3::x_axis(), &Float3::z_axis()), 0.0, -0.707106769, 0.0, 0.707106769);

        // pi/2 around x
        expect_quaternion_eq!(
            Quaternion::from_vectors(&Float3::y_axis(), &Float3::z_axis()), 0.707106769, 0.0, 0.0, 0.707106769);

        // Non unit pi/2 around x
        expect_quaternion_eq!(
            Quaternion::from_vectors(&(Float3::y_axis() * 9.0), &(Float3::z_axis() * 13.0)), 0.707106769, 0.0, 0.0, 0.707106769);

        // pi/2 around z
        expect_quaternion_eq!(
            Quaternion::from_vectors(&Float3::x_axis(), &Float3::y_axis()), 0.0, 0.0, 0.707106769, 0.707106769);

        // pi/2 around z also
        expect_quaternion_eq!(
            Quaternion::from_vectors(&Float3::new(0.707106769, 0.707106769, 0.0),
                                    &Float3::new(-0.707106769, 0.707106769, 0.0)),
            0.0, 0.0, 0.707106769, 0.707106769);

        // Aligned vectors
        expect_quaternion_eq!(
            Quaternion::from_vectors(&Float3::x_axis(), &Float3::x_axis()), 0.0, 0.0, 0.0, 1.0);

        // Non-unit aligned vectors
        expect_quaternion_eq!(
            Quaternion::from_vectors(&Float3::x_axis(), &(Float3::x_axis() * 2.0)), 0.0, 0.0, 0.0, 1.0);

        // Opposed vectors
        expect_quaternion_eq!(
            Quaternion::from_vectors(&Float3::x_axis(), &-Float3::x_axis()), 0.0, 1.0, 0.0, 0.0);
        expect_quaternion_eq!(
            Quaternion::from_vectors(&-Float3::x_axis(), &Float3::x_axis()), 0.0, -1.0, 0.0, 0.0);
        expect_quaternion_eq!(
            Quaternion::from_vectors(&Float3::y_axis(), &-Float3::y_axis()), 0.0, 0.0, 1.0, 0.0);
        expect_quaternion_eq!(
            Quaternion::from_vectors(&-Float3::y_axis(), &Float3::y_axis()), 0.0, 0.0, -1.0, 0.0);
        expect_quaternion_eq!(
            Quaternion::from_vectors(&Float3::z_axis(), &-Float3::z_axis()), 0.0, -1.0, 0.0, 0.0);
        expect_quaternion_eq!(
            Quaternion::from_vectors(&-Float3::z_axis(), &Float3::z_axis()), 0.0, 1.0, 0.0, 0.0);
        expect_quaternion_eq!(
            Quaternion::from_vectors(&Float3::new(0.707106769, 0.707106769, 0.0),
                                    &-Float3::new(0.707106769, 0.707106769, 0.0)),
            -0.707106769, 0.707106769, 0.0, 0.0);
        expect_quaternion_eq!(
            Quaternion::from_vectors(&Float3::new(0.0, 0.707106769, 0.707106769),
                                    &-Float3::new(0.0, 0.707106769, 0.707106769)),
            0.0, -0.707106769, 0.707106769, 0.0);

        // Non-unit opposed vectors
        expect_quaternion_eq!(
            Quaternion::from_vectors(&Float3::new(2.0, 2.0, 2.0), &-Float3::new(2.0, 2.0, 2.0)),
            0.0, -0.707106769, 0.707106769, 0.0);
    }

    #[test]
    fn quaternion_from_unit_vectors() {
        // pi/2 around y
        expect_quaternion_eq!(
            Quaternion::from_unit_vectors(&Float3::z_axis(), &Float3::x_axis()), 0.0, 0.707106769, 0.0, 0.707106769);

        // Minus pi/2 around y
        expect_quaternion_eq!(
            Quaternion::from_unit_vectors(&Float3::x_axis(), &Float3::z_axis()), 0.0, -0.707106769, 0.0, 0.707106769);

        // pi/2 around x
        expect_quaternion_eq!(
            Quaternion::from_unit_vectors(&Float3::y_axis(), &Float3::z_axis()), 0.707106769, 0.0, 0.0, 0.707106769);

        // pi/2 around z
        expect_quaternion_eq!(
            Quaternion::from_unit_vectors(&Float3::x_axis(), &Float3::y_axis()), 0.0, 0.0, 0.707106769, 0.707106769);

        // pi/2 around z also
        expect_quaternion_eq!(
            Quaternion::from_unit_vectors(&Float3::new(0.707106769, 0.707106769, 0.0),
                                        &Float3::new(-0.707106769, 0.707106769, 0.0)),
            0.0, 0.0, 0.707106769, 0.707106769);

        // Aligned vectors
        expect_quaternion_eq!(
            Quaternion::from_unit_vectors(&Float3::x_axis(), &Float3::x_axis()), 0.0, 0.0, 0.0, 1.0);

        // Opposed vectors
        expect_quaternion_eq!(
            Quaternion::from_unit_vectors(&Float3::x_axis(), &-Float3::x_axis()), 0.0, 1.0, 0.0, 0.0);
        expect_quaternion_eq!(
            Quaternion::from_unit_vectors(&-Float3::x_axis(), &Float3::x_axis()), 0.0, -1.0, 0.0, 0.0);
        expect_quaternion_eq!(
            Quaternion::from_unit_vectors(&Float3::y_axis(), &-Float3::y_axis()), 0.0, 0.0, 1.0, 0.0);
        expect_quaternion_eq!(
            Quaternion::from_unit_vectors(&-Float3::y_axis(), &Float3::y_axis()), 0.0, 0.0, -1.0, 0.0);
        expect_quaternion_eq!(
            Quaternion::from_unit_vectors(&Float3::z_axis(), &-Float3::z_axis()), 0.0, -1.0, 0.0, 0.0);
        expect_quaternion_eq!(
            Quaternion::from_unit_vectors(&-Float3::z_axis(), &Float3::z_axis()), 0.0, 1.0, 0.0, 0.0);
        expect_quaternion_eq!(
            Quaternion::from_unit_vectors(&Float3::new(0.707106769, 0.707106769, 0.0),
                                        &-Float3::new(0.707106769, 0.707106769, 0.0)),
            -0.707106769, 0.707106769, 0.0, 0.0);
        expect_quaternion_eq!(
            Quaternion::from_unit_vectors(&Float3::new(0.0, 0.707106769, 0.707106769),
                                        &-Float3::new(0.0, 0.707106769, 0.707106769)),
            0.0, -0.707106769, 0.707106769, 0.0);
    }

    #[test]
    fn quaternion_compare() {
        assert_eq!(Quaternion::identity() == Quaternion::new(0.0, 0.0, 0.0, 1.0), true);
        assert_eq!(Quaternion::identity() != Quaternion::new(1.0, 0.0, 0.0, 0.0), true);
        assert_eq!(Quaternion::compare(&Quaternion::identity(), &Quaternion::identity(), f32::cos(0.5 * 0.0)), true);
        assert_eq!(Quaternion::compare(&Quaternion::identity(),
                                       &Quaternion::from_euler(0.0, 0.0, crate::math_constant::K_PI / 100.0),
                                       f32::cos(0.5 * crate::math_constant::K_PI / 50.0)), true);
        assert_eq!(Quaternion::compare(&Quaternion::identity(),
                                       &-Quaternion::from_euler(0.0, 0.0, crate::math_constant::K_PI / 100.0),
                                       f32::cos(0.5 * crate::math_constant::K_PI / 50.0)), true);
        assert_eq!(Quaternion::compare(&Quaternion::identity(),
                                       &Quaternion::from_euler(0.0, 0.0, crate::math_constant::K_PI / 100.0),
                                       f32::cos(0.5 * crate::math_constant::K_PI / 200.0)), false);
    }

    #[test]
    fn quaternion_arithmetic() {
        let a = Quaternion::new(0.70710677, 0.0, 0.0, 0.70710677);
        let b = Quaternion::new(0.0, 0.70710677, 0.0, 0.70710677);
        let c = Quaternion::new(0.0, 0.70710677, 0.0, -0.70710677);
        let denorm = Quaternion::new(1.414212, 0.0, 0.0, 1.414212);

        assert_eq!(a.is_normalized(), true);
        assert_eq!(b.is_normalized(), true);
        assert_eq!(c.is_normalized(), true);
        assert_eq!(denorm.is_normalized(), false);

        let conjugate = a.conjugate();
        expect_quaternion_eq!(conjugate, -a.x, -a.y, -a.z, a.w);
        assert_eq!(conjugate.is_normalized(), true);

        let negate = -a;
        expect_quaternion_eq!(negate, -a.x, -a.y, -a.z, -a.w);
        assert_eq!(negate.is_normalized(), true);

        let add = a + b;
        expect_quaternion_eq!(add, 0.70710677, 0.70710677, 0.0, 1.41421354);

        let mul0 = a * conjugate;
        expect_quaternion_eq!(mul0, 0.0, 0.0, 0.0, 1.0);
        assert_eq!(mul0.is_normalized(), true);

        let muls = a * 2.0;
        expect_quaternion_eq!(muls, 1.41421354, 0.0, 0.0, 1.41421354);

        let mul1 = conjugate * a;
        expect_quaternion_eq!(mul1, 0.0, 0.0, 0.0, 1.0);
        assert_eq!(mul1.is_normalized(), true);

        let q1234 = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let q5678 = Quaternion::new(5.0, 6.0, 7.0, 8.0);
        let mul12345678 = q1234 * q5678;
        expect_quaternion_eq!(mul12345678, 24.0, 48.0, 48.0, -6.0);

        // EXPECT_ASSERTION(Normalize(Quaternion(0.0, 0.0, 0.0, 0.0)),
        //                  "is not normalizable");
        let normalize = denorm.normalize();
        assert_eq!(normalize.is_normalized(), true);
        expect_quaternion_eq!(normalize, 0.7071068, 0.0, 0.0, 0.7071068);

        // EXPECT_ASSERTION(NormalizeSafe(denorm, Quaternion(0.0, 0.0, 0.0, 0.0)),
        //                  "_safer is not normalized");
        let normalize_safe = denorm.normalize_safe(&Quaternion::identity());
        assert_eq!(normalize_safe.is_normalized(), true);
        expect_quaternion_eq!(normalize_safe, 0.7071068, 0.0, 0.0, 0.7071068);

        let normalize_safer = Quaternion::new(0.0, 0.0, 0.0, 0.0).normalize_safe(&Quaternion::identity());
        assert_eq!(normalize_safer.is_normalized(), true);
        expect_quaternion_eq!(normalize_safer, 0.0, 0.0, 0.0, 1.0);

        let lerp_0 = Quaternion::lerp(&a, &b, 0.0);
        expect_quaternion_eq!(lerp_0, a.x, a.y, a.z, a.w);

        let lerp_1 = Quaternion::lerp(&a, &b, 1.0);
        expect_quaternion_eq!(lerp_1, b.x, b.y, b.z, b.w);

        let lerp_0_2 = Quaternion::lerp(&a, &b, 0.2);
        expect_quaternion_eq!(lerp_0_2, 0.5656853, 0.1414213, 0.0, 0.7071068);

        let nlerp_0 = Quaternion::nlerp(&a, &b, 0.0);
        assert_eq!(nlerp_0.is_normalized(), true);
        expect_quaternion_eq!(nlerp_0, a.x, a.y, a.z, a.w);

        let nlerp_1 = Quaternion::nlerp(&a, &b, 1.0);
        assert_eq!(nlerp_1.is_normalized(), true);
        expect_quaternion_eq!(nlerp_1, b.x, b.y, b.z, b.w);

        let nlerp_0_2 = Quaternion::nlerp(&a, &b, 0.2);
        assert_eq!(nlerp_0_2.is_normalized(), true);
        expect_quaternion_eq!(nlerp_0_2, 0.6172133, 0.1543033, 0.0, 0.7715167);

        // EXPECT_ASSERTION(slerp(denorm, b, 0.0), "is_normalized\\(_a\\)");
        // EXPECT_ASSERTION(slerp(a, denorm, 0.0), "is_normalized\\(_b\\)");

        let slerp_0 = Quaternion::slerp(&a, &b, 0.0);
        assert_eq!(slerp_0.is_normalized(), true);
        expect_quaternion_eq!(slerp_0, a.x, a.y, a.z, a.w);

        let slerp_c_0 = Quaternion::slerp(&a, &c, 0.0);
        assert_eq!(slerp_c_0.is_normalized(), true);
        expect_quaternion_eq!(slerp_c_0, a.x, a.y, a.z, a.w);

        let slerp_c_1 = Quaternion::slerp(&a, &c, 1.0);
        assert_eq!(slerp_c_1.is_normalized(), true);
        expect_quaternion_eq!(slerp_c_1, c.x, c.y, c.z, c.w);

        let slerp_c_0_6 = Quaternion::slerp(&a, &c, 0.6);
        assert_eq!(slerp_c_0_6.is_normalized(), true);
        expect_quaternion_eq!(slerp_c_0_6, 0.6067752, 0.7765344, 0.0, -0.1697592);

        let slerp_1 = Quaternion::slerp(&a, &b, 1.0);
        assert_eq!(slerp_1.is_normalized(), true);
        expect_quaternion_eq!(slerp_1, b.x, b.y, b.z, b.w);

        let slerp_0_2 = Quaternion::slerp(&a, &b, 0.2);
        assert_eq!(slerp_0_2.is_normalized(), true);
        expect_quaternion_eq!(slerp_0_2, 0.6067752, 0.1697592, 0.0, 0.7765344);

        let slerp_0_7 = Quaternion::slerp(&a, &b, 0.7);
        assert_eq!(slerp_0_7.is_normalized(), true);
        expect_quaternion_eq!(slerp_0_7, 0.2523113, 0.5463429, 0.0, 0.798654);

        let dot = Quaternion::dot(&a, &b);
        expect_near!(dot, 0.5, f32::EPSILON);
    }

    #[test]
    fn quaternion_transform_vector() {
        // 0 length
        expect_float3_eq!(Quaternion::from_axis_angle(&Float3::y_axis(), 0.0).transform_vector(&Float3::zero()),
                         0.0, 0.0, 0.0);

        // Unit length
        expect_float3_eq!(Quaternion::from_axis_angle(&Float3::y_axis(), 0.0).transform_vector(&Float3::z_axis()),
                         0.0, 0.0, 1.0);
        expect_float3_eq!(Quaternion::from_axis_angle(&Float3::y_axis(), crate::math_constant::K_PI_2).transform_vector(&Float3::y_axis()),
                         0.0, 1.0, 0.0);
        expect_float3_eq!(Quaternion::from_axis_angle(&Float3::y_axis(), crate::math_constant::K_PI_2).transform_vector(&Float3::x_axis()),
                         0.0, 0.0, -1.0);
        expect_float3_eq!(Quaternion::from_axis_angle(&Float3::y_axis(), crate::math_constant::K_PI_2).transform_vector(&Float3::z_axis()),
                         1.0, 0.0, 0.0);

        // Non unit
        expect_float3_eq!(Quaternion::from_axis_angle(&Float3::z_axis(), crate::math_constant::K_PI_2).transform_vector(&(Float3::x_axis() * 2.0)),
                         0.0, 2.0, 0.0);
    }
}