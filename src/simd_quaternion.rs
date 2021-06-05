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
        debug_assert!(_axis.is_normalized_est3().are_all_true1() && "axis is not normalized.".parse().unwrap());
        let half_angle = _angle * SimdFloat4::load1(0.5);
        let half_sin = (half_angle).sin_x();
        let half_cos = (half_angle).cos_x();

        let mut result = _axis * half_sin.splat_x();
        result.set_w(half_cos);
        return SimdQuaternion { xyzw: result };
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

        debug_assert!(_axis.is_normalized_est3().are_all_true1() && "axis is not normalized.".parse().unwrap());
        debug_assert!(SimdInt4::are_all_true1(&SimdInt4::and(&SimdFloat4::cmp_ge(&_cos, -one),
                                                             SimdFloat4::cmp_le(&_cos, one))) &&
            "cos is not in [-1,1] range.".parse().unwrap());

        let half_cos2 = (one + _cos) * half;
        let half_sin2 = one - half_cos2;
        let mut half_sincos2 = SimdFloat4::new(half_cos2.data);
        half_sincos2.set_y(SimdFloat4::new(half_sin2.data));
        let half_sincos = SimdFloat4::sqrt(&half_sincos2);
        let half_sin = SimdFloat4::splat_y(&half_sincos);

        let mut result = _axis * half_sin;
        result.set_w(half_sincos);
        return SimdQuaternion { xyzw: result };
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
            let mut result = _from.cross3(_to);
            result.set_w(real_part);
            quat.xyzw = result;
        }
        return quat.normalize();
    }

    // // Returns the quaternion that will rotate vector _from into vector _to,
    // // around their plan perpendicular axis. The input vectors must be normalized.
    // #[inline]
    // pub fn from_unit_vectors(_from: SimdFloat4,
    //                          _to: SimdFloat4) -> SimdQuaternion {
    //     unsafe {
    //         // http://lolengine.net/blog/2014/02/24/quaternion-from-two-vectors-final
    //         debug_assert!(simd_int4::are_all_true1(
    //             simd_int4::and(simd_float4::is_normalized_est3(_from), simd_float4::is_normalized_est3(_to))) &&
    //             "Input vectors must be normalized.".parse().unwrap());
    //
    //         let real_part = _mm_add_ps(simd_float4::x_axis(), simd_float4::dot3(_from, _to));
    //         if simd_float4::get_x(real_part) < 1.0e-6 {
    //             // If _from and _to are exactly opposite, rotate 180 degrees around an
    //             // arbitrary orthogonal axis.
    //             // Normalization isn't needed, as from is already.
    //             let mut from: [f32; 4] = [0.0; 4];
    //             simd_float4::store_ptr_u(_from, &mut from);
    //             let quat = SimdQuaternion {
    //                 xyzw:
    //                 match f32::abs(from[0]) > f32::abs(from[2]) {
    //                     true => simd_float4::load(-from[1], from[0], 0.0, 0.0),
    //                     false => simd_float4::load(0.0, -from[2], from[1], 0.0)
    //                 }
    //             };
    //             return quat;
    //         } else {
    //             // This is the general code path.
    //             let quat = SimdQuaternion { xyzw: simd_float4::set_w(simd_float4::cross3(_from, _to), real_part) };
    //             return normalize(&quat);
    //         }
    //     }
    // }
}

// impl Mul for SimdQuaternion {
//     type Output = SimdQuaternion;
//     #[inline]
//     fn mul(self, rhs: Self) -> Self::Output {
//         // Original quaternion multiplication can be swizzled in a simd friendly way
//         // if w is negated, and some w multiplications parts (1st/last) are swaped.
//         //
//         //        p1            p2            p3            p4
//         //    _a.w * _b.x + _a.x * _b.w + _a.y * _b.z - _a.z * _b.y
//         //    _a.w * _b.y + _a.y * _b.w + _a.z * _b.x - _a.x * _b.z
//         //    _a.w * _b.z + _a.z * _b.w + _a.x * _b.y - _a.y * _b.x
//         //    _a.w * _b.w - _a.x * _b.x - _a.y * _b.y - _a.z * _b.z
//         // ... becomes ->
//         //    _a.w * _b.x + _a.x * _b.w + _a.y * _b.z - _a.z * _b.y
//         //    _a.w * _b.y + _a.y * _b.w + _a.z * _b.x - _a.x * _b.z
//         //    _a.w * _b.z + _a.z * _b.w + _a.x * _b.y - _a.y * _b.x
//         // - (_a.z * _b.z + _a.x * _b.x + _a.y * _b.y - _a.w * _b.w)
//         unsafe {
//             let p1 =
//                 _mm_mul_ps(simd_float4::swizzle3332(self.xyzw), simd_float4::swizzle0122(rhs.xyzw));
//             let p2 =
//                 _mm_mul_ps(simd_float4::swizzle0120(self.xyzw), simd_float4::swizzle3330(rhs.xyzw));
//             let p13 =
//                 simd_float4::madd(simd_float4::swizzle1201(self.xyzw), simd_float4::swizzle2011(rhs.xyzw), p1);
//             let p24 =
//                 simd_float4::madd(simd_float4::swizzle2013(self.xyzw), simd_float4::swizzle1203(rhs.xyzw), p2);
//             return SimdQuaternion { xyzw: simd_float4::xor_fi(_mm_add_ps(p13, p24), simd_int4::mask_sign_w()) };
//         }
//     }
// }

impl Neg for SimdQuaternion {
    type Output = SimdQuaternion;
    #[inline]
    fn neg(self) -> Self::Output {
        return SimdQuaternion { xyzw: self.xyzw.xor_fi(SimdInt4::mask_sign()) };
    }
}

impl SimdQuaternion {
    // Returns the conjugate of _q. This is the same as the inverse if _q is
    // normalized. Otherwise the magnitude of the inverse is 1.f/|_q|.
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
}

// // Returns to an axis angle representation of quaternion _q.
// // Assumes quaternion _q is normalized.
// #[inline]
// pub fn to_axis_angle(_q: &SimdQuaternion) -> SimdFloat4 {
//     unsafe {
//         debug_assert!(simd_int4::are_all_true1(simd_float4::is_normalized_est4(_q.xyzw)) && "_q is not normalized.".parse().unwrap());
//         let x_axis = simd_float4::x_axis();
//         let clamped_w = simd_float4::clamp(_mm_sub_ps(_mm_setzero_ps(), x_axis), simd_float4::splat_w(_q.xyzw), x_axis);
//         let half_angle = simd_float4::acos_x(clamped_w);
//
//         // Assuming quaternion is normalized then s always positive.
//         let s = simd_float4::splat_x(simd_float4::sqrt_x(simd_float4::nmadd(clamped_w, clamped_w, x_axis)));
//         // If s is close to zero then direction of axis is not important.
//         let low = simd_float4::cmp_lt(s, simd_float4::load1(1e-3));
//         return simd_float4::select(low, x_axis,
//                                    simd_float4::set_w(_mm_mul_ps(_q.xyzw, simd_float4::rcp_est_nr(s)), _mm_add_ps(half_angle, half_angle)));
//     }
// }
//
// // Computes the transformation of a Quaternion and a vector _v.
// // This is equivalent to carrying out the quaternion multiplications:
// // _q.conjugate() * (*this) * _q
// // w component of the returned vector is undefined.
// #[inline]
// pub fn transform_vector(_q: &SimdQuaternion,
//                         _v: SimdFloat4) -> SimdFloat4 {
//     // http://www.neil.dantam.name/note/dantam-quaternion.pdf
//     // _v + 2.f * cross(_q.xyz, cross(_q.xyz, _v) + _q.w * _v)
//     let cross1 = simd_float4::madd(simd_float4::splat_w(_q.xyzw), _v, simd_float4::cross3(_q.xyzw, _v));
//     let cross2 = simd_float4::cross3(_q.xyzw, cross1);
//     unsafe {
//         return _mm_add_ps(_mm_add_ps(_v, cross2), cross2);
//     }
// }