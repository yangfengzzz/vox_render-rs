/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::simd_math::*;
use std::ops::{Mul, Neg};

// Declare the Quaternion type.
#[derive(Clone, Copy)]
pub struct SimdQuaternion {
    pub xyzw: SimdFloat4,
}

impl SimdQuaternion {
    // Returns the identity quaternion.
    #[inline]
    pub fn identity() -> SimdQuaternion {
        return SimdQuaternion { xyzw: SimdFloat4::w_axis() };
    }

    // the angle in radian.
    #[inline]
    pub fn from_axis_angle(_axis: SimdFloat4,
                           _angle: SimdFloat4) -> SimdQuaternion {
        debug_assert!(_axis.is_normalized_est3().are_all_true1() && "axis is not normalized.".parse().unwrap_or(true));
        let half_angle = _angle * SimdFloat4::load1(0.5);
        let half_sin = (half_angle).sin_x();
        let half_cos = (half_angle).cos_x();

        return SimdQuaternion { xyzw: _axis * half_sin.splat_x().set_w(half_cos) };
    }

    // Returns a normalized quaternion initialized from an axis and angle cosine
    // representation.
    // Assumes the axis part (x, y, z) of _axis_angle is normalized.
    // _angle.x is the angle cosine in radian, it must be within [-1,1] range.
    #[inline]
    pub fn from_axis_cos_angle(_axis: SimdFloat4,
                               _cos: SimdFloat4) -> SimdQuaternion {
        let one = SimdFloat4::one();
        let half = SimdFloat4::load1(0.5);

        debug_assert!(_axis.is_normalized_est3().are_all_true1() && "axis is not normalized.".parse().unwrap_or(true));
        debug_assert!(SimdInt4::are_all_true1(&SimdInt4::and(&SimdFloat4::cmp_ge(&_cos, -one),
                                                             SimdFloat4::cmp_le(&_cos, one))) &&
            "cos is not in [-1,1] range.".parse().unwrap_or(true));

        let half_cos2 = (one + _cos) * half;
        let half_sin2 = one - half_cos2;
        let half_sincos2 = half_cos2.set_y(half_sin2);
        let half_sincos = SimdFloat4::sqrt(&half_sincos2);
        let half_sin = SimdFloat4::splat_y(&half_sincos);

        return SimdQuaternion { xyzw: (_axis * half_sin).set_w(half_sincos) };
    }

    // Returns the quaternion that will rotate vector _from into vector _to,
    // around their plan perpendicular axis.The input vectors don't need to be
    // normalized, they can be null also.
    #[inline]
    pub fn from_vectors(_from: SimdFloat4,
                        _to: SimdFloat4) -> SimdQuaternion {
        // http://lolengine.net/blog/2014/02/24/quaternion-from-two-vectors-final
        let norm_from_norm_to = (_from.length3sqr() * _to.length3sqr()).sqrt_x();
        let norm_from_norm_to_x = norm_from_norm_to.get_x();
        if norm_from_norm_to_x < 1.0e-6 {
            return SimdQuaternion::identity();
        }

        let real_part = norm_from_norm_to + _from.dot3(_to);
        let mut quat = SimdQuaternion { xyzw: SimdFloat4::zero() };
        if real_part.get_x() < 1.0e-6 * norm_from_norm_to_x {
            // If _from and _to are exactly opposite, rotate 180 degrees around an
            // arbitrary orthogonal axis. Axis normalization can happen later, when we
            // normalize the quaternion.
            let mut from: [f32; 4] = [0.0; 4];
            _from.store_ptr_u(&mut from);
            quat.xyzw = match f32::abs(from[0]) > f32::abs(from[2]) {
                true => SimdFloat4::load(-from[1], from[0], 0.0, 0.0),
                false => SimdFloat4::load(0.0, -from[2], from[1], 0.0)
            }
        } else {
            // This is the general code path.
            quat.xyzw = _from.cross3(_to).set_w(real_part);
        }
        return quat.normalize();
    }

    // Returns the quaternion that will rotate vector _from into vector _to,
    // around their plan perpendicular axis. The input vectors must be normalized.
    #[inline]
    pub fn from_unit_vectors(_from: SimdFloat4,
                             _to: SimdFloat4) -> SimdQuaternion {
        // http://lolengine.net/blog/2014/02/24/quaternion-from-two-vectors-final
        debug_assert!(SimdInt4::are_all_true1(
            &SimdInt4::and(&SimdFloat4::is_normalized_est3(&_from), SimdFloat4::is_normalized_est3(&_to))) &&
            "Input vectors must be normalized.".parse().unwrap_or(true));

        let real_part = SimdFloat4::x_axis() + _from.dot3(_to);
        return if real_part.get_x() < 1.0e-6 {
            // If _from and _to are exactly opposite, rotate 180 degrees around an
            // arbitrary orthogonal axis.
            // Normalization isn't needed, as from is already.
            let mut from: [f32; 4] = [0.0; 4];
            _from.store_ptr_u(&mut from);
            let quat = SimdQuaternion {
                xyzw:
                match f32::abs(from[0]) > f32::abs(from[2]) {
                    true => SimdFloat4::load(-from[1], from[0], 0.0, 0.0),
                    false => SimdFloat4::load(0.0, -from[2], from[1], 0.0)
                }
            };
            quat
        } else {
            // This is the general code path.
            let quat = SimdQuaternion { xyzw: _from.cross3(_to).set_w(real_part) };
            quat.normalize()
        };
    }
}

impl Mul for SimdQuaternion {
    type Output = SimdQuaternion;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
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
        let p1 =
            self.xyzw.swizzle3332() * rhs.xyzw.swizzle0122();
        let p2 =
            self.xyzw.swizzle0120() * rhs.xyzw.swizzle3330();
        let p13 = self.xyzw.swizzle1201().madd(rhs.xyzw.swizzle2011(), p1);
        let p24 = self.xyzw.swizzle2013().nmadd(rhs.xyzw.swizzle1203(), p2);
        return SimdQuaternion { xyzw: (p13 + p24).xor_fi(SimdInt4::mask_sign_w()) };
    }
}

impl Neg for SimdQuaternion {
    type Output = SimdQuaternion;
    #[inline]
    fn neg(self) -> Self::Output {
        return SimdQuaternion { xyzw: self.xyzw.xor_fi(SimdInt4::mask_sign()) };
    }
}

impl SimdQuaternion {
    // Returns the conjugate of _q. This is the same as the inverse if _q is
    // normalized. Otherwise the magnitude of the inverse is 1.0/|_q|.
    #[inline]
    pub fn conjugate(&self) -> SimdQuaternion {
        return SimdQuaternion { xyzw: self.xyzw.xor_fi(SimdInt4::mask_sign_xyz()) };
    }

    // Returns the normalized quaternion _q.
    #[inline]
    pub fn normalize(&self) -> SimdQuaternion {
        return SimdQuaternion { xyzw: self.xyzw.normalize4() };
    }

    // Returns the normalized quaternion _q if the norm of _q is not 0.
    // Otherwise returns _safer.
    #[inline]
    pub fn normalize_safe(&self,
                          _safer: &SimdQuaternion) -> SimdQuaternion {
        return SimdQuaternion { xyzw: self.xyzw.normalize_safe4(_safer.xyzw) };
    }

    // Returns the estimated normalized quaternion _q.
    #[inline]
    pub fn normalize_est(&self) -> SimdQuaternion {
        return SimdQuaternion { xyzw: self.xyzw.normalize_est4() };
    }

    // Returns the estimated normalized quaternion _q if the norm of _q is not 0.
    // Otherwise returns _safer.
    #[inline]
    pub fn normalize_safe_est(&self,
                              _safer: &SimdQuaternion) -> SimdQuaternion {
        return SimdQuaternion { xyzw: self.xyzw.normalize_safe_est4(_safer.xyzw) };
    }

    // Tests if the _q is a normalized quaternion.
    // Returns the result in the x component of the returned vector. y, z and w are
    // set to 0.
    #[inline]
    pub fn is_normalized(&self) -> SimdInt4 {
        return self.xyzw.is_normalized4();
    }

    // Tests if the _q is a normalized quaternion.
    // Uses the estimated normalization coefficient, that matches estimated math
    // functions (RecpEst, MormalizeEst...).
    // Returns the result in the x component of the returned vector. y, z and w are
    // set to 0.
    #[inline]
    pub fn is_normalized_est(&self) -> SimdInt4 {
        return self.xyzw.is_normalized_est4();
    }

    // Returns to an axis angle representation of quaternion _q.
    // Assumes quaternion _q is normalized.
    #[inline]
    pub fn to_axis_angle(&self) -> SimdFloat4 {
        debug_assert!(self.xyzw.is_normalized_est4().are_all_true1() && "self is not normalized.".parse().unwrap_or(true));
        let x_axis = SimdFloat4::x_axis();
        let clamped_w = SimdFloat4::clamp(&-x_axis, self.xyzw.splat_w(), x_axis);
        let half_angle = clamped_w.acos_x();

        // Assuming quaternion is normalized then s always positive.
        let s = SimdFloat4::nmadd(&clamped_w, clamped_w, x_axis).sqrt_x().splat_x();
        // If s is close to zero then direction of axis is not important.
        let low = SimdFloat4::cmp_lt(&s, SimdFloat4::load1(1e-3));

        return SimdFloat4::select(low, x_axis, (self.xyzw * s.rcp_est_nr()).set_w(half_angle + half_angle));
    }

    // Computes the transformation of a Quaternion and a vector _v.
    // This is equivalent to carrying out the quaternion multiplications:
    // _q.conjugate() * (*this) * _q
    // w component of the returned vector is undefined.
    #[inline]
    pub fn transform_vector(&self,
                            _v: SimdFloat4) -> SimdFloat4 {
        // http://www.neil.dantam.name/note/dantam-quaternion.pdf
        // _v + 2.0 * cross(_q.xyz, cross(_q.xyz, _v) + _q.w * _v)
        let cross1 = self.xyzw.splat_w().madd(_v, self.xyzw.cross3(_v));
        let cross2 = self.xyzw.cross3(cross1);
        return _v + cross2 + cross2;
    }
}

//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
#[cfg(test)]
mod ozz_simd_math {
    use crate::simd_quaternion::SimdQuaternion;
    use crate::simd_math::*;
    use crate::math_test_helper::*;
    use crate::*;

    #[test]
    fn quaternion_constant() {
        expect_simd_quaternion_eq!(SimdQuaternion::identity(), 0.0, 0.0, 0.0, 1.0);
    }

    #[test]
    #[allow(overflowing_literals)]
    fn quaternion_arithmetic() {
        let a = SimdQuaternion {
            xyzw:
            SimdFloat4::load(0.70710677, 0.0, 0.0, 0.70710677)
        };
        let b = SimdQuaternion {
            xyzw:
            SimdFloat4::load(0.0, 0.70710677, 0.0, 0.70710677)
        };
        let c = SimdQuaternion {
            xyzw:
            SimdFloat4::load(0.0, 0.70710677, 0.0, -0.70710677)
        };
        let denorm = SimdQuaternion {
            xyzw:
            SimdFloat4::load(1.414212, 0.0, 0.0, 1.414212)
        };
        let zero = SimdQuaternion {
            xyzw:
            SimdFloat4::zero()
        };

        expect_simd_int_eq!(a.is_normalized(), 0xffffffff, 0, 0, 0);
        expect_simd_int_eq!(b.is_normalized(), 0xffffffff, 0, 0, 0);
        expect_simd_int_eq!(c.is_normalized(), 0xffffffff, 0, 0, 0);
        expect_simd_int_eq!(denorm.is_normalized(), 0, 0, 0, 0);

        let conjugate = a.conjugate();
        expect_simd_quaternion_eq!(conjugate, -0.70710677, 0.0, 0.0, 0.70710677);

        let negate = -a;
        expect_simd_quaternion_eq!(negate, -0.70710677, 0.0, 0.0, -0.70710677);

        let mul0 = a * conjugate;
        expect_simd_quaternion_eq!(mul0, 0.0, 0.0, 0.0, 1.0);

        let mul1 = conjugate * a;
        expect_simd_quaternion_eq!(mul1, 0.0, 0.0, 0.0, 1.0);

        let q1234 = SimdQuaternion {
            xyzw:
            SimdFloat4::load(1.0, 2.0, 3.0, 4.0)
        };
        let q5678 = SimdQuaternion {
            xyzw:
            SimdFloat4::load(5.0, 6.0, 7.0, 8.0)
        };
        let mul12345678 = q1234 * q5678;
        expect_simd_quaternion_eq!(mul12345678, 24.0, 48.0, 48.0, -6.0);

        // EXPECT_ASSERTION(Normalize(zero), "is not normalizable");
        let norm = denorm.normalize();
        expect_simd_int_eq!(norm.is_normalized(), 0xffffffff, 0, 0, 0);
        expect_simd_quaternion_eq!(norm, 0.7071068, 0.0, 0.0, 0.7071068);

        // EXPECT_ASSERTION(NormalizeSafe(denorm, zero), "_safer is not normalized");
        let norm_safe = denorm.normalize_safe(&b);
        expect_simd_int_eq!(norm_safe.is_normalized(), 0xffffffff, 0, 0, 0);
        expect_simd_quaternion_eq!(norm_safe, 0.7071068, 0.0, 0.0, 0.7071068);
        let norm_safer = zero.normalize_safe(&b);
        expect_simd_int_eq!(norm_safer.is_normalized(), 0xffffffff, 0, 0, 0);
        expect_simd_quaternion_eq!(norm_safer, 0.0, 0.70710677, 0.0, 0.70710677);

        // EXPECT_ASSERTION(NormalizeEst(zero), "is not normalizable");
        let norm_est = denorm.normalize_est();
        expect_simd_int_eq!(norm_est.is_normalized_est(), 0xffffffff, 0, 0, 0);
        expect_simd_quaternion_eq_est!(norm_est, 0.7071068, 0.0, 0.0, 0.7071068);

        // EXPECT_ASSERTION(NormalizeSafe(denorm, zero), "_safer is not normalized");
        let norm_safe_est = denorm.normalize_safe_est(&b);
        expect_simd_int_eq!(norm_safe_est.is_normalized_est(), 0xffffffff, 0, 0, 0);
        expect_simd_quaternion_eq_est!(norm_safe_est, 0.7071068, 0.0, 0.0, 0.7071068);
        let norm_safer_est = zero.normalize_safe_est(&b);
        expect_simd_int_eq!(norm_safer_est.is_normalized_est(), 0xffffffff, 0, 0, 0);
        expect_simd_quaternion_eq_est!(norm_safer_est, 0.0, 0.70710677, 0.0,
                                     0.70710677);
    }

    #[test]
    fn quaternion_from_vectors() {
        // Returns identity for a 0 length vector
        expect_simd_quaternion_eq!(
            SimdQuaternion::from_vectors(SimdFloat4::zero(),
                                         SimdFloat4::x_axis()),
            0.0, 0.0, 0.0, 1.0);

        // pi/2 around y
        expect_simd_quaternion_eq!(
            SimdQuaternion::from_vectors(SimdFloat4::z_axis(),
                                         SimdFloat4::x_axis()),
            0.0, 0.707106769, 0.0, 0.707106769);

        // Non unit pi/2 around y
        expect_simd_quaternion_eq!(
            SimdQuaternion::from_vectors(SimdFloat4::z_axis(),
                                         SimdFloat4::x_axis() *
                                            SimdFloat4::load1(27.0)),
            0.0, 0.707106769, 0.0, 0.707106769);

        // Minus pi/2 around y
        expect_simd_quaternion_eq!(
            SimdQuaternion::from_vectors(SimdFloat4::x_axis(),
                                         SimdFloat4::z_axis()),
            0.0, -0.707106769, 0.0, 0.707106769);

        // pi/2 around x
        expect_simd_quaternion_eq!(
            SimdQuaternion::from_vectors(SimdFloat4::y_axis(),
                                         SimdFloat4::z_axis()),
            0.707106769, 0.0, 0.0, 0.707106769);

        // pi/2 around z
        expect_simd_quaternion_eq!(
            SimdQuaternion::from_vectors(SimdFloat4::x_axis(),
                                         SimdFloat4::y_axis()),
            0.0, 0.0, 0.707106769, 0.707106769);

        // pi/2 around z also
        expect_simd_quaternion_eq!(
            SimdQuaternion::from_vectors(
                SimdFloat4::load(0.707106769, 0.707106769, 0.0, 99.0),
                SimdFloat4::load(-0.707106769, 0.707106769, 0.0, 93.0)),
            0.0, 0.0, 0.707106769, 0.707106769);

        // Non unit pi/2 around z also
        expect_simd_quaternion_eq!(
            SimdQuaternion::from_vectors(
                SimdFloat4::load(0.707106769, 0.707106769, 0.0, 99.0) *
                    SimdFloat4::load1(9.0),
                SimdFloat4::load(-0.707106769, 0.707106769, 0.0, 93.0) *
                    SimdFloat4::load1(46.0)),
            0.0, 0.0, 0.707106769, 0.707106769);

        // Non-unit pi/2 around z
        expect_simd_quaternion_eq!(
            SimdQuaternion::from_vectors(SimdFloat4::x_axis(),
                                        SimdFloat4::y_axis() *
                                            SimdFloat4::load1(2.0)),
            0.0, 0.0, 0.707106769, 0.707106769);

        // Aligned vectors
        expect_simd_quaternion_eq!(
            SimdQuaternion::from_vectors(SimdFloat4::x_axis(),
                                         SimdFloat4::x_axis()),
            0.0, 0.0, 0.0, 1.0);

        // Non-unit aligned vectors
        expect_simd_quaternion_eq!(
            SimdQuaternion::from_vectors(SimdFloat4::x_axis(),
                                         SimdFloat4::x_axis() *
                                             SimdFloat4::load1(2.0)),
            0.0, 0.0, 0.0, 1.0);

        // Opposed vectors
        expect_simd_quaternion_eq!(
            SimdQuaternion::from_vectors(SimdFloat4::x_axis(),
                                         -SimdFloat4::x_axis()),
            0.0, 1.0, 0.0, 0.0);
        expect_simd_quaternion_eq!(
            SimdQuaternion::from_vectors(-SimdFloat4::x_axis(),
                                         SimdFloat4::x_axis()),
            0.0, -1.0, 0.0, 0.0);
        expect_simd_quaternion_eq!(
            SimdQuaternion::from_vectors(SimdFloat4::y_axis(),
                                         -SimdFloat4::y_axis()),
            0.0, 0.0, 1.0, 0.0);
        expect_simd_quaternion_eq!(
            SimdQuaternion::from_vectors(-SimdFloat4::y_axis(),
                                         SimdFloat4::y_axis()),
            0.0, 0.0, -1.0, 0.0);
        expect_simd_quaternion_eq!(
            SimdQuaternion::from_vectors(SimdFloat4::z_axis(),
                                         -SimdFloat4::z_axis()),
            0.0, -1.0, 0.0, 0.0);
        expect_simd_quaternion_eq!(
            SimdQuaternion::from_vectors(-SimdFloat4::z_axis(),
                                         SimdFloat4::z_axis()),
            0.0, 1.0, 0.0, 0.0);
        expect_simd_quaternion_eq!(
            SimdQuaternion::from_vectors(
                SimdFloat4::load(0.707106769, 0.707106769, 0.0, 93.0),
                -SimdFloat4::load(0.707106769, 0.707106769, 0.0, 99.0)),
            -0.707106769, 0.707106769, 0.0, 0.0);
        expect_simd_quaternion_eq!(
            SimdQuaternion::from_vectors(
                SimdFloat4::load(0.0, 0.707106769, 0.707106769, 93.0),
                -SimdFloat4::load(0.0, 0.707106769, 0.707106769, 99.0)),
            0.0, -0.707106769, 0.707106769, 0.0);

        // Non-unit opposed vectors
        expect_simd_quaternion_eq!(
            SimdQuaternion::from_vectors(
                SimdFloat4::load(2.0, 2.0, 2.0, 0.0),
                SimdFloat4::load(-2.0, -2.0, -2.0, 0.0)),
            0.0, -0.707106769, 0.707106769, 0.0);
    }

    #[test]
    fn quaternion_from_unit_vectors() {
        // // assert 0 length vectors
        // EXPECT_ASSERTION(
        //     SimdQuaternion::from_unit_vectors(SimdFloat4::zero(),
        //                                       SimdFloat4::x_axis()),
        //     "Input vectors must be normalized.");
        // EXPECT_ASSERTION(
        //     SimdQuaternion::from_unit_vectors(SimdFloat4::x_axis(),
        //                                       SimdFloat4::zero()),
        //     "Input vectors must be normalized.");
        // // assert non unit vectors
        // EXPECT_ASSERTION(
        //     SimdQuaternion::from_unit_vectors(
        //         SimdFloat4::x_axis() * SimdFloat4::Load1(2.0),
        //         SimdFloat4::x_axis()),
        //     "Input vectors must be normalized.");
        // EXPECT_ASSERTION(
        //     SimdQuaternion::from_unit_vectors(SimdFloat4::x_axis(),
        //                                       SimdFloat4::x_axis() *
        //                                           SimdFloat4::Load1(.5f)),
        //     "Input vectors must be normalized.");

        // pi/2 around y
        expect_simd_quaternion_eq!(
            SimdQuaternion::from_unit_vectors(SimdFloat4::z_axis(),
                                              SimdFloat4::x_axis()),
            0.0, 0.707106769, 0.0, 0.707106769);

        // Minus pi/2 around y
        expect_simd_quaternion_eq!(
            SimdQuaternion::from_unit_vectors(SimdFloat4::x_axis(),
                                              SimdFloat4::z_axis()),
            0.0, -0.707106769, 0.0, 0.707106769);

        // pi/2 around x
        expect_simd_quaternion_eq!(
            SimdQuaternion::from_unit_vectors(SimdFloat4::y_axis(),
                                              SimdFloat4::z_axis()),
            0.707106769, 0.0, 0.0, 0.707106769);

        // pi/2 around z
        expect_simd_quaternion_eq!(
            SimdQuaternion::from_unit_vectors(SimdFloat4::x_axis(),
                                              SimdFloat4::y_axis()),
            0.0, 0.0, 0.707106769, 0.707106769);

        // pi/2 around z also
        expect_simd_quaternion_eq!(
            SimdQuaternion::from_unit_vectors(
                SimdFloat4::load(0.707106769, 0.707106769, 0.0, 99.0),
                SimdFloat4::load(-0.707106769, 0.707106769, 0.0, 93.0)),
            0.0, 0.0, 0.707106769, 0.707106769);

        // Aligned vectors
        expect_simd_quaternion_eq!(
            SimdQuaternion::from_unit_vectors(SimdFloat4::x_axis(),
                                              SimdFloat4::x_axis()),
            0.0, 0.0, 0.0, 1.0);

        // Opposed vectors
        expect_simd_quaternion_eq!(
            SimdQuaternion::from_unit_vectors(SimdFloat4::x_axis(),
                                              -SimdFloat4::x_axis()),
            0.0, 1.0, 0.0, 0.0);
        expect_simd_quaternion_eq!(
            SimdQuaternion::from_unit_vectors(-SimdFloat4::x_axis(),
                                              SimdFloat4::x_axis()),
            0.0, -1.0, 0.0, 0.0);
        expect_simd_quaternion_eq!(
            SimdQuaternion::from_unit_vectors(SimdFloat4::y_axis(),
                                              -SimdFloat4::y_axis()),
            0.0, 0.0, 1.0, 0.0);
        expect_simd_quaternion_eq!(
            SimdQuaternion::from_unit_vectors(-SimdFloat4::y_axis(),
                                              SimdFloat4::y_axis()),
            0.0, 0.0, -1.0, 0.0);
        expect_simd_quaternion_eq!(
            SimdQuaternion::from_unit_vectors(SimdFloat4::z_axis(),
                                              -SimdFloat4::z_axis()),
            0.0, -1.0, 0.0, 0.0);
        expect_simd_quaternion_eq!(
            SimdQuaternion::from_unit_vectors(-SimdFloat4::z_axis(),
                                              SimdFloat4::z_axis()),
            0.0, 1.0, 0.0, 0.0);
        expect_simd_quaternion_eq!(
            SimdQuaternion::from_unit_vectors(
                SimdFloat4::load(0.707106769, 0.707106769, 0.0, 93.0),
                -SimdFloat4::load(0.707106769, 0.707106769, 0.0, 99.0)),
            -0.707106769, 0.707106769, 0.0, 0.0);
        expect_simd_quaternion_eq!(
            SimdQuaternion::from_unit_vectors(
                SimdFloat4::load(0.0, 0.707106769, 0.707106769, 93.0),
                -SimdFloat4::load(0.0, 0.707106769, 0.707106769, 99.0)),
            0.0, -0.707106769, 0.707106769, 0.0);
    }

    #[test]
    fn quaternion_axis_angle() {
        // // Expect assertions from invalid inputs
        // EXPECT_ASSERTION(
        //     SimdQuaternion::from_axis_angle(SimdFloat4::zero(),
        //                                   SimdFloat4::zero()),
        //     "axis is not normalized.");
        //
        // let unorm = SimdQuaternion{xyzw: SimdFloat4::load(0.0, 0.0, 0.0, 2.0)};
        //
        // EXPECT_ASSERTION(unorm.to_axis_angle(), "_q is not normalized.");

        // Identity
        expect_simd_quaternion_eq!(
            SimdQuaternion::from_axis_angle(SimdFloat4::x_axis(),
                                          SimdFloat4::zero()),
            0.0, 0.0, 0.0, 1.0);
        expect_simd_float_eq!(SimdQuaternion::identity().to_axis_angle(), 1.0, 0.0, 0.0, 0.0);

        // Other axis angles
        let pi_2 = SimdFloat4::load_x(crate::math_constant::K_PI_2);
        let qy_pi_2 = SimdQuaternion::from_axis_angle(SimdFloat4::y_axis(), pi_2);
        expect_simd_quaternion_eq!(qy_pi_2, 0.0, 0.70710677, 0.0, 0.70710677);
        expect_simd_float_eq!(qy_pi_2.to_axis_angle(), 0.0, 1.0, 0.0, crate::math_constant::K_PI_2);

        let qy_mpi_2 = SimdQuaternion::from_axis_angle(SimdFloat4::y_axis(), -pi_2);
        expect_simd_quaternion_eq!(qy_mpi_2, 0.0, -0.70710677, 0.0, 0.70710677);
        expect_simd_float_eq!(qy_mpi_2.to_axis_angle(), 0.0, -1.0, 0.0, crate::math_constant::K_PI_2);  // q = -q
        let qmy_pi_2 = SimdQuaternion::from_axis_angle(-SimdFloat4::y_axis(), pi_2);
        expect_simd_quaternion_eq!(qmy_pi_2, 0.0, -0.70710677, 0.0, 0.70710677);

        let any_axis = SimdFloat4::load(0.819865, 0.033034, -0.571604, 99.0);
        let any_angle = SimdFloat4::load(1.123, 99.0, 26.0, 93.0);
        let qany = SimdQuaternion::from_axis_angle(any_axis, any_angle);
        expect_simd_quaternion_eq!(qany, 0.4365425, 0.017589169, -0.30435428, 0.84645736);
        expect_simd_float_eq!(qany.to_axis_angle(), 0.819865, 0.033034, -0.571604, 1.123);
    }
}













