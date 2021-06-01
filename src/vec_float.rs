/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use std::ops::{Add, Sub, Neg, Mul, Div, BitAnd};
use cgmath::*;

// Declares a 2d float vector.
#[derive(Clone)]
pub struct Float2 {
    pub value: cgmath::Vector2<f32>,
}

impl Float2 {
    // Constructs an uninitialized vector.
    #[inline]
    pub fn new_default() -> Float2 {
        return Float2 {
            value: Vector2::zero(),
        };
    }

    // Constructs a vector initialized with _f value.
    #[inline]
    pub fn new_scalar(_f: f32) -> Float2 {
        return Float2 {
            value: Vector2::from_value(_f)
        };
    }

    // Constructs a vector initialized with _x and _y values.
    #[inline]
    pub fn new(_x: f32, _y: f32) -> Float2 {
        return Float2 {
            value: Vector2::new(_x, _y)
        };
    }

    // Returns a vector with all components set to 0.
    #[inline]
    pub fn zero() -> Float2 { return Float2::new_scalar(0.0); }

    // Returns a vector with all components set to 1.
    #[inline]
    pub fn one() -> Float2 { return Float2::new_scalar(1.0); }

    // Returns a unitary vector x.
    #[inline]
    pub fn x_axis() -> Float2 { return Float2::new(1.0, 0.0); }

    // Returns a unitary vector y.
    #[inline]
    pub fn y_axis() -> Float2 { return Float2::new(0.0, 1.0); }
}

//--------------------------------------------------------------------------------------------------
// Declares a 3d float vector.
#[derive(Clone)]
pub struct Float3 {
    pub value: cgmath::Vector3<f32>,
}

impl Float3 {
    // Constructs an uninitialized vector.
    #[inline]
    pub fn new_default() -> Float3 {
        return Float3 {
            value: Vector3::zero(),
        };
    }

    // Constructs a vector initialized with _f value.
    #[inline]
    pub fn new_scalar(_f: f32) -> Float3 {
        return Float3 {
            value: Vector3::from_value(_f)
        };
    }

    // Constructs a vector initialized with _x, _y and _z values.
    #[inline]
    pub fn new(_x: f32, _y: f32, _z: f32) -> Float3 {
        return Float3 {
            value: Vector3::new(_x, _y, _z)
        };
    }

    // Returns a vector initialized with _v.value.x, _v.value.y and _z values.
    #[inline]
    pub fn new2(_v: Float2, _z: f32) -> Float3 {
        return Float3 {
            value: Vector3::new(_v.value.x, _v.value.y, _z)
        };
    }

    // Returns a vector with all components set to 0.
    #[inline]
    pub fn zero() -> Float3 { return Float3::new_scalar(0.0); }

    // Returns a vector with all components set to 1.
    #[inline]
    pub fn one() -> Float3 { return Float3::new_scalar(1.0); }

    // Returns a unitary vector x.
    #[inline]
    pub fn x_axis() -> Float3 { return Float3::new(1.0, 0.0, 0.0); }

    // Returns a unitary vector y.
    #[inline]
    pub fn y_axis() -> Float3 { return Float3::new(0.0, 1.0, 0.0); }

    // Returns a unitary vector z.
    #[inline]
    pub fn z_axis() -> Float3 { return Float3::new(0.0, 0.0, 1.0); }
}

//--------------------------------------------------------------------------------------------------
// Declares a 4d float vector.
#[derive(Clone)]
pub struct Float4 {
    pub value: cgmath::Vector4<f32>,
}

impl Float4 {
    // Constructs an uninitialized vector.
    #[inline]
    pub fn new_default() -> Float4 {
        return Float4 {
            value: Vector4::zero(),
        };
    }

    // Constructs a vector initialized with _f value.
    #[inline]
    pub fn new_scalar(_f: f32) -> Float4 {
        return Float4 {
            value: Vector4::from_value(_f)
        };
    }

    // Constructs a vector initialized with _x, _y, _z and _w values.
    #[inline]
    pub fn new(_x: f32, _y: f32, _z: f32, _w: f32) -> Float4 {
        return Float4 {
            value: Vector4::new(_x, _y, _z, _w)
        };
    }

    // Constructs a vector initialized with _v.value.x, _v.value.y, _v.value.z and _w values.
    #[inline]
    pub fn new3(_v: Float3, _w: f32) -> Float4 {
        return Float4 {
            value: Vector4::new(_v.value.x, _v.value.y, _v.value.z, _w)
        };
    }

    // Constructs a vector initialized with _v.value.x, _v.value.y, _z and _w values.
    #[inline]
    pub fn new2(_v: Float2, _z: f32, _w: f32) -> Float4 {
        return Float4 {
            value: Vector4::new(_v.value.x, _v.value.y, _z, _w)
        };
    }

    // Returns a vector with all components set to 0.
    #[inline]
    pub fn zero() -> Float4 { return Float4::new_scalar(0.0); }

    // Returns a vector with all components set to 1.
    #[inline]
    pub fn one() -> Float4 { return Float4::new_scalar(1.0); }

    // Returns a unitary vector x.
    #[inline]
    pub fn x_axis() -> Float4 { return Float4::new(1.0, 0.0, 0.0, 0.0); }

    // Returns a unitary vector y.
    #[inline]
    pub fn y_axis() -> Float4 { return Float4::new(0.0, 1.0, 0.0, 0.0); }

    // Returns a unitary vector z.
    #[inline]
    pub fn z_axis() -> Float4 { return Float4::new(0.0, 0.0, 1.0, 0.0); }

    // Returns a unitary vector w.
    #[inline]
    pub fn w_axis() -> Float4 { return Float4::new(0.0, 0.0, 0.0, 1.0); }
}
//--------------------------------------------------------------------------------------------------

macro_rules! impl_add4 {
    ($rhs:ty) => {
        impl Add for $rhs {
            type Output = Float4;
            #[inline]
            fn add(self, rhs: Self) -> Self::Output {
                return Float4::new(self.value.x + rhs.value.x, self.value.y + rhs.value.y,
                                   self.value.z + rhs.value.z, self.value.w + rhs.value.w);
            }
        }
    }
}
impl_add4!(Float4);
impl_add4!(&Float4);

macro_rules! impl_add3 {
    ($rhs:ty) => {
        impl Add for $rhs {
            type Output = Float3;
            #[inline]
            fn add(self, rhs: Self) -> Self::Output {
                return Float3::new(self.value.x + rhs.value.x, self.value.y + rhs.value.y, self.value.z + rhs.value.z);
            }
        }
    }
}
impl_add3!(Float3);
impl_add3!(&Float3);

macro_rules! impl_add2 {
    ($rhs:ty) => {
        impl Add for $rhs {
            type Output = Float2;
            #[inline]
            fn add(self, rhs: Self) -> Self::Output {
                return Float2::new(self.value.x + rhs.value.x, self.value.y + rhs.value.y);
            }
        }
    }
}
impl_add2!(Float2);
impl_add2!(&Float2);

//--------------------------------------------------------------------------------------------------
macro_rules! impl_sub4 {
    ($rhs:ty) => {
        impl Sub for $rhs {
            type Output = Float4;
            #[inline]
            fn sub(self, rhs: Self) -> Self::Output {
                return Float4::new(self.value.x - rhs.value.x, self.value.y - rhs.value.y,
                                   self.value.z - rhs.value.z, self.value.w - rhs.value.w);
            }
        }
    }
}
impl_sub4!(Float4);
impl_sub4!(&Float4);

macro_rules! impl_sub3 {
    ($rhs:ty) => {
        impl Sub for $rhs {
            type Output = Float3;
            #[inline]
            fn sub(self, rhs: Self) -> Self::Output {
                return Float3::new(self.value.x - rhs.value.x, self.value.y - rhs.value.y, self.value.z - rhs.value.z);
            }
        }
    }
}
impl_sub3!(Float3);
impl_sub3!(&Float3);

macro_rules! impl_sub2 {
    ($rhs:ty) => {
        impl Sub for $rhs {
            type Output = Float2;
            #[inline]
            fn sub(self, rhs: Self) -> Self::Output {
                return Float2::new(self.value.x - rhs.value.x, self.value.y - rhs.value.y);
            }
        }
    }
}
impl_sub2!(Float2);
impl_sub2!(&Float2);

//--------------------------------------------------------------------------------------------------
impl Neg for Float4 {
    type Output = Float4;
    #[inline]
    fn neg(self) -> Self::Output {
        return Float4::new(-self.value.x, -self.value.y, -self.value.z, -self.value.w);
    }
}

impl Neg for Float3 {
    type Output = Float3;
    #[inline]
    fn neg(self) -> Self::Output {
        return Float3::new(-self.value.x, -self.value.y, -self.value.z);
    }
}

impl Neg for Float2 {
    type Output = Float2;
    #[inline]
    fn neg(self) -> Self::Output {
        return Float2::new(-self.value.x, -self.value.y);
    }
}

//--------------------------------------------------------------------------------------------------
macro_rules! impl_mul4 {
    ($rhs:ty) => {
        impl Mul for $rhs {
            type Output = Float4;
            #[inline]
            fn mul(self, rhs: Self) -> Self::Output {
                return Float4::new(self.value.x * rhs.value.x, self.value.y * rhs.value.y, self.value.z * rhs.value.z, self.value.w * rhs.value.w);
            }
        }
    }
}
impl_mul4!(Float4);
impl_mul4!(&Float4);

macro_rules! impl_mul3 {
    ($rhs:ty) => {
        impl Mul for $rhs {
            type Output = Float3;
            #[inline]
            fn mul(self, rhs: Self) -> Self::Output {
                return Float3::new(self.value.x * rhs.value.x, self.value.y * rhs.value.y, self.value.z * rhs.value.z);
            }
        }
    }
}
impl_mul3!(Float3);
impl_mul3!(&Float3);

macro_rules! impl_mul2 {
    ($rhs:ty) => {
        impl Mul for $rhs {
            type Output = Float2;
            #[inline]
            fn mul(self, rhs: Self) -> Self::Output {
                return Float2::new(self.value.x * rhs.value.x, self.value.y * rhs.value.y);
            }
        }
    }
}
impl_mul2!(Float2);
impl_mul2!(&Float2);

//--------------------------------------------------------------------------------------------------

impl Mul<f32> for Float4 {
    type Output = Float4;
    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        return Float4::new(self.value.x * rhs, self.value.y * rhs, self.value.z * rhs, self.value.w * rhs);
    }
}

impl Mul<f32> for Float3 {
    type Output = Float3;
    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        return Float3::new(self.value.x * rhs, self.value.y * rhs, self.value.z * rhs);
    }
}

impl Mul<f32> for Float2 {
    type Output = Float2;
    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        return Float2::new(self.value.x * rhs, self.value.y * rhs);
    }
}

//--------------------------------------------------------------------------------------------------
macro_rules! impl_div4 {
    ($rhs:ty) => {
        impl Div for $rhs {
            type Output = Float4;
            #[inline]
            fn div(self, rhs: Self) -> Self::Output {
                return Float4::new(self.value.x / rhs.value.x, self.value.y / rhs.value.y, self.value.z / rhs.value.z, self.value.w / rhs.value.w);
            }
        }
    };
}
impl_div4!(Float4);
impl_div4!(&Float4);

macro_rules! impl_div3 {
    ($rhs:ty) => {
        impl Div for $rhs {
            type Output = Float3;
            #[inline]
            fn div(self, rhs: Self) -> Self::Output {
                return Float3::new(self.value.x / rhs.value.x, self.value.y / rhs.value.y, self.value.z / rhs.value.z);
            }
        }
    }
}
impl_div3!(Float3);
impl_div3!(&Float3);

macro_rules! impl_div2 {
    ($rhs:ty) => {
        impl Div for $rhs {
            type Output = Float2;
            #[inline]
            fn div(self, rhs: Self) -> Self::Output {
                return Float2::new(self.value.x / rhs.value.x, self.value.y / rhs.value.y);
            }
        }
    }
}
impl_div2!(Float2);
impl_div2!(&Float2);

//--------------------------------------------------------------------------------------------------
// Returns the (horizontal) addition of each element of _v.
#[inline]
pub fn h_add4(_v: &Float4) -> f32 {
    return _v.value.x + _v.value.y + _v.value.z + _v.value.w;
}

#[inline]
pub fn h_add3(_v: &Float3) -> f32 {
    return _v.value.x + _v.value.y + _v.value.z;
}

#[inline]
pub fn h_add2(_v: &Float2) -> f32 {
    return _v.value.x + _v.value.y;
}

//--------------------------------------------------------------------------------------------------
// Returns the dot product of _a and _b.
#[inline]
pub fn dot4(_a: &Float4, _b: &Float4) -> f32 {
    return _a.value.x * _b.value.x + _a.value.y * _b.value.y + _a.value.z * _b.value.z + _a.value.w * _b.value.w;
}

#[inline]
pub fn dot3(_a: &Float3, _b: &Float3) -> f32 {
    return _a.value.x * _b.value.x + _a.value.y * _b.value.y + _a.value.z * _b.value.z;
}

#[inline]
pub fn dot2(_a: &Float2, _b: &Float2) -> f32 {
    return _a.value.x * _b.value.x + _a.value.y * _b.value.y;
}

//--------------------------------------------------------------------------------------------------
// Returns the cross product of _a and _b.
#[inline]
pub fn cross(_a: &Float3, _b: &Float3) -> Float3 {
    return Float3::new(_a.value.y * _b.value.z - _b.value.y * _a.value.z,
                       _a.value.z * _b.value.x - _b.value.z * _a.value.x,
                       _a.value.x * _b.value.y - _b.value.x * _a.value.y);
}

//--------------------------------------------------------------------------------------------------
// Returns the length |_v| of _v.
#[inline]
pub fn length4(_v: &Float4) -> f32 {
    let len2 = _v.value.x * _v.value.x + _v.value.y * _v.value.y + _v.value.z * _v.value.z + _v.value.w * _v.value.w;
    return f32::sqrt(len2);
}

#[inline]
pub fn length3(_v: &Float3) -> f32 {
    let len2 = _v.value.x * _v.value.x + _v.value.y * _v.value.y + _v.value.z * _v.value.z;
    return f32::sqrt(len2);
}

#[inline]
pub fn length2(_v: &Float2) -> f32 {
    let len2 = _v.value.x * _v.value.x + _v.value.y * _v.value.y;
    return f32::sqrt(len2);
}

//--------------------------------------------------------------------------------------------------
// Returns the square length |_v|^2 of _v.
#[inline]
pub fn length_sqr4(_v: &Float4) -> f32 {
    return _v.value.x * _v.value.x + _v.value.y * _v.value.y + _v.value.z * _v.value.z + _v.value.w * _v.value.w;
}

#[inline]
pub fn length_sqr3(_v: &Float3) -> f32 {
    return _v.value.x * _v.value.x + _v.value.y * _v.value.y + _v.value.z * _v.value.z;
}

#[inline]
pub fn length_sqr2(_v: &Float2) -> f32 {
    return _v.value.x * _v.value.x + _v.value.y * _v.value.y;
}

//--------------------------------------------------------------------------------------------------
// Returns the normalized vector _v.
#[inline]
pub fn normalize4(_v: &Float4) -> Float4 {
    let len2 = _v.value.x * _v.value.x + _v.value.y * _v.value.y + _v.value.z * _v.value.z + _v.value.w * _v.value.w;
    debug_assert!(len2 != 0.0 && "_v is not normalizable".parse().unwrap());
    let len = f32::sqrt(len2);
    return Float4::new(_v.value.x / len, _v.value.y / len, _v.value.z / len, _v.value.w / len);
}

#[inline]
pub fn normalize3(_v: &Float3) -> Float3 {
    let len2 = _v.value.x * _v.value.x + _v.value.y * _v.value.y + _v.value.z * _v.value.z;
    debug_assert!(len2 != 0.0 && "_v is not normalizable".parse().unwrap());
    let len = f32::sqrt(len2);
    return Float3::new(_v.value.x / len, _v.value.y / len, _v.value.z / len);
}

#[inline]
pub fn normalize2(_v: &Float2) -> Float2 {
    let len2 = _v.value.x * _v.value.x + _v.value.y * _v.value.y;
    debug_assert!(len2 != 0.0 && "_v is not normalizable".parse().unwrap());
    let len = f32::sqrt(len2);
    return Float2::new(_v.value.x / len, _v.value.y / len);
}

//--------------------------------------------------------------------------------------------------
// Returns true if _v is normalized.
#[inline]
pub fn is_normalized4(_v: &Float4) -> bool {
    let len2 = _v.value.x * _v.value.x + _v.value.y * _v.value.y + _v.value.z * _v.value.z + _v.value.w * _v.value.w;
    return f32::abs(len2 - 1.0) < crate::math_constant::K_NORMALIZATION_TOLERANCE_SQ;
}

#[inline]
pub fn is_normalized3(_v: &Float3) -> bool {
    let len2 = _v.value.x * _v.value.x + _v.value.y * _v.value.y + _v.value.z * _v.value.z;
    return f32::abs(len2 - 1.0) < crate::math_constant::K_NORMALIZATION_TOLERANCE_SQ;
}

#[inline]
pub fn is_normalized2(_v: &Float2) -> bool {
    let len2 = _v.value.x * _v.value.x + _v.value.y * _v.value.y;
    return f32::abs(len2 - 1.0) < crate::math_constant::K_NORMALIZATION_TOLERANCE_SQ;
}

//--------------------------------------------------------------------------------------------------
// Returns the normalized vector _v if the norm of _v is not 0.
// Otherwise returns _safer.
#[inline]
pub fn normalize_safe4(_v: &Float4, _safer: &Float4) -> Float4 {
    debug_assert!(is_normalized4(_safer) && "_safer is not normalized".parse().unwrap());
    let len2 = _v.value.x * _v.value.x + _v.value.y * _v.value.y + _v.value.z * _v.value.z + _v.value.w * _v.value.w;
    if len2 <= 0.0 {
        return _safer.clone();
    }
    let len = f32::sqrt(len2);
    return Float4::new(_v.value.x / len, _v.value.y / len, _v.value.z / len, _v.value.w / len);
}

#[inline]
pub fn normalize_safe3(_v: &Float3, _safer: &Float3) -> Float3 {
    debug_assert!(is_normalized3(_safer) && "_safer is not normalized".parse().unwrap());
    let len2 = _v.value.x * _v.value.x + _v.value.y * _v.value.y + _v.value.z * _v.value.z;
    if len2 <= 0.0 {
        return _safer.clone();
    }
    let len = f32::sqrt(len2);
    return Float3::new(_v.value.x / len, _v.value.y / len, _v.value.z / len);
}

#[inline]
pub fn normalize_safe2(_v: &Float2, _safer: &Float2) -> Float2 {
    debug_assert!(is_normalized2(_safer) && "_safer is not normalized".parse().unwrap());
    let len2 = _v.value.x * _v.value.x + _v.value.y * _v.value.y;
    if len2 <= 0.0 {
        return _safer.clone();
    }
    let len = f32::sqrt(len2);
    return Float2::new(_v.value.x / len, _v.value.y / len);
}

//--------------------------------------------------------------------------------------------------
// Returns the linear interpolation of _a and _b with coefficient _f.
// _f is not limited to range [0,1].
#[inline]
pub fn lerp4(_a: &Float4, _b: &Float4, _f: f32) -> Float4 {
    return Float4::new((_b.value.x - _a.value.x) * _f + _a.value.x, (_b.value.y - _a.value.y) * _f + _a.value.y,
                       (_b.value.z - _a.value.z) * _f + _a.value.z, (_b.value.w - _a.value.w) * _f + _a.value.w);
}

#[inline]
pub fn lerp3(_a: &Float3, _b: &Float3, _f: f32) -> Float3 {
    return Float3::new((_b.value.x - _a.value.x) * _f + _a.value.x, (_b.value.y - _a.value.y) * _f + _a.value.y,
                       (_b.value.z - _a.value.z) * _f + _a.value.z);
}

#[inline]
pub fn lerp2(_a: &Float2, _b: &Float2, _f: f32) -> Float2 {
    return Float2::new((_b.value.x - _a.value.x) * _f + _a.value.x, (_b.value.y - _a.value.y) * _f + _a.value.y);
}

//--------------------------------------------------------------------------------------------------
// Returns true if the distance between _a and _b is less than _tolerance.
#[inline]
pub fn compare4(_a: &Float4, _b: &Float4, _tolerance: f32) -> bool {
    let diff = _a - _b;
    return dot4(&diff, &diff) <= _tolerance * _tolerance;
}

#[inline]
pub fn compare3(_a: &Float3, _b: &Float3, _tolerance: f32) -> bool {
    let diff = _a - _b;
    return dot3(&diff, &diff) <= _tolerance * _tolerance;
}

#[inline]
pub fn compare2(_a: &Float2, _b: &Float2, _tolerance: f32) -> bool {
    let diff = _a - _b;
    return dot2(&diff, &diff) <= _tolerance * _tolerance;
}

//--------------------------------------------------------------------------------------------------
impl Float4 {
    #[inline]
    pub fn lt(&self, other: &Self) -> bool {
        let x = self.value.x.lt(&other.value.x);
        let y = self.value.y.lt(&other.value.y);
        let z = self.value.z.lt(&other.value.z);
        let w = self.value.w.lt(&other.value.w);
        return x.bitand(y).bitand(z).bitand(w);
    }

    #[inline]
    pub fn le(&self, other: &Self) -> bool {
        let x = self.value.x.le(&other.value.x);
        let y = self.value.y.le(&other.value.y);
        let z = self.value.z.le(&other.value.z);
        let w = self.value.w.le(&other.value.w);
        return x.bitand(y).bitand(z).bitand(w);
    }

    #[inline]
    pub fn gt(&self, other: &Self) -> bool {
        let x = self.value.x.gt(&other.value.x);
        let y = self.value.y.gt(&other.value.y);
        let z = self.value.z.gt(&other.value.z);
        let w = self.value.w.gt(&other.value.w);
        return x.bitand(y).bitand(z).bitand(w);
    }

    #[inline]
    pub fn ge(&self, other: &Self) -> bool {
        let x = self.value.x.ge(&other.value.x);
        let y = self.value.y.ge(&other.value.y);
        let z = self.value.z.ge(&other.value.z);
        let w = self.value.w.ge(&other.value.w);
        return x.bitand(y).bitand(z).bitand(w);
    }

    #[inline]
    pub fn eq(&self, other: &Self) -> bool {
        let x = self.value.x.eq(&other.value.x);
        let y = self.value.y.eq(&other.value.y);
        let z = self.value.z.eq(&other.value.z);
        let w = self.value.w.eq(&other.value.w);
        return x.bitand(y).bitand(z).bitand(w);
    }

    #[inline]
    pub fn ne(&self, other: &Self) -> bool {
        let x = self.value.x.ne(&other.value.x);
        let y = self.value.y.ne(&other.value.y);
        let z = self.value.z.ne(&other.value.z);
        let w = self.value.w.ne(&other.value.w);
        return x.bitand(y).bitand(z).bitand(w);
    }
}

impl PartialEq for Float4 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        return self.value.x == other.value.x && self.value.y == other.value.y && self.value.z == other.value.z && self.value.w == other.value.w;
    }
}

impl Float3 {
    #[inline]
    pub fn lt(&self, other: &Self) -> bool {
        let x = self.value.x.lt(&other.value.x);
        let y = self.value.y.lt(&other.value.y);
        let z = self.value.z.lt(&other.value.z);
        return x.bitand(y).bitand(z);
    }

    #[inline]
    pub fn le(&self, other: &Self) -> bool {
        let x = self.value.x.le(&other.value.x);
        let y = self.value.y.le(&other.value.y);
        let z = self.value.z.le(&other.value.z);
        return x.bitand(y).bitand(z);
    }

    #[inline]
    pub fn gt(&self, other: &Self) -> bool {
        let x = self.value.x.gt(&other.value.x);
        let y = self.value.y.gt(&other.value.y);
        let z = self.value.z.gt(&other.value.z);
        return x.bitand(y).bitand(z);
    }

    #[inline]
    pub fn ge(&self, other: &Self) -> bool {
        let x = self.value.x.ge(&other.value.x);
        let y = self.value.y.ge(&other.value.y);
        let z = self.value.z.ge(&other.value.z);
        return x.bitand(y).bitand(z);
    }

    #[inline]
    pub fn eq(&self, other: &Self) -> bool {
        let x = self.value.x.eq(&other.value.x);
        let y = self.value.y.eq(&other.value.y);
        let z = self.value.z.eq(&other.value.z);
        return x.bitand(y).bitand(z);
    }

    #[inline]
    pub fn ne(&self, other: &Self) -> bool {
        let x = self.value.x.ne(&other.value.x);
        let y = self.value.y.ne(&other.value.y);
        let z = self.value.z.ne(&other.value.z);
        return x.bitand(y).bitand(z);
    }
}

impl PartialEq for Float3 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        return self.value.x == other.value.x && self.value.y == other.value.y && self.value.z == other.value.z;
    }
}

impl Float2 {
    #[inline]
    pub fn lt(&self, other: &Self) -> bool {
        let x = self.value.x.lt(&other.value.x);
        let y = self.value.y.lt(&other.value.y);
        return x.bitand(y);
    }

    #[inline]
    pub fn le(&self, other: &Self) -> bool {
        let x = self.value.x.le(&other.value.x);
        let y = self.value.y.le(&other.value.y);
        return x.bitand(y);
    }

    #[inline]
    pub fn gt(&self, other: &Self) -> bool {
        let x = self.value.x.gt(&other.value.x);
        let y = self.value.y.gt(&other.value.y);
        return x.bitand(y);
    }

    #[inline]
    pub fn ge(&self, other: &Self) -> bool {
        let x = self.value.x.ge(&other.value.x);
        let y = self.value.y.ge(&other.value.y);
        return x.bitand(y);
    }

    #[inline]
    pub fn eq(&self, other: &Self) -> bool {
        let x = self.value.x.eq(&other.value.x);
        let y = self.value.y.eq(&other.value.y);
        return x.bitand(y);
    }

    #[inline]
    pub fn ne(&self, other: &Self) -> bool {
        let x = self.value.x.ne(&other.value.x);
        let y = self.value.y.ne(&other.value.y);
        return x.bitand(y);
    }
}

impl PartialEq for Float2 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        return self.value.x == other.value.x && self.value.y == other.value.y;
    }
}

//--------------------------------------------------------------------------------------------------
// Returns the minimum of each element of _a and _b.
#[inline]
pub fn min4(_a: &Float4, _b: &Float4) -> Float4 {
    return Float4::new(
        match _a.value.x < _b.value.x {
            true => _a.value.x,
            false => _b.value.x
        },
        match _a.value.y < _b.value.y {
            true => _a.value.y,
            false => _b.value.y
        },
        match _a.value.z < _b.value.z {
            true => _a.value.z,
            false => _b.value.z
        },
        match _a.value.w < _b.value.w {
            true => _a.value.w,
            false => _b.value.w
        });
}

#[inline]
pub fn min3(_a: &Float3, _b: &Float3) -> Float3 {
    return Float3::new(
        match _a.value.x < _b.value.x {
            true => _a.value.x,
            false => _b.value.x
        },
        match _a.value.y < _b.value.y {
            true => _a.value.y,
            false => _b.value.y
        },
        match _a.value.z < _b.value.z {
            true => _a.value.z,
            false => _b.value.z
        });
}

#[inline]
pub fn min2(_a: &Float2, _b: &Float2) -> Float2 {
    return Float2::new(
        match _a.value.x < _b.value.x {
            true => _a.value.x,
            false => _b.value.x
        },
        match _a.value.y < _b.value.y {
            true => _a.value.y,
            false => _b.value.y
        });
}

//--------------------------------------------------------------------------------------------------
// Returns the maximum of each element of _a and _b.
#[inline]
pub fn max4(_a: &Float4, _b: &Float4) -> Float4 {
    return Float4::new(
        match _a.value.x > _b.value.x {
            true => _a.value.x,
            false => _b.value.x
        },
        match _a.value.y > _b.value.y {
            true => _a.value.y,
            false => _b.value.y
        },
        match _a.value.z > _b.value.z {
            true => _a.value.z,
            false => _b.value.z
        },
        match _a.value.w > _b.value.w {
            true => _a.value.w,
            false => _b.value.w
        });
}

#[inline]
pub fn max3(_a: &Float3, _b: &Float3) -> Float3 {
    return Float3::new(
        match _a.value.x > _b.value.x {
            true => _a.value.x,
            false => _b.value.x
        },
        match _a.value.y > _b.value.y {
            true => _a.value.y,
            false => _b.value.y
        },
        match _a.value.z > _b.value.z {
            true => _a.value.z,
            false => _b.value.z
        });
}

#[inline]
pub fn max2(_a: &Float2, _b: &Float2) -> Float2 {
    return Float2::new(
        match _a.value.x > _b.value.x {
            true => _a.value.x,
            false => _b.value.x
        },
        match _a.value.y > _b.value.y {
            true => _a.value.y,
            false => _b.value.y
        });
}

//--------------------------------------------------------------------------------------------------
// Clamps each element of _x between _a and _b.
// _a must be less or equal to b;
#[inline]
pub fn clamp4(_a: &Float4, _v: &Float4, _b: &Float4) -> Float4 {
    let min = Float4::new(
        match _v.value.x < _b.value.x {
            true => _v.value.x,
            false => _b.value.x
        },
        match _v.value.y < _b.value.y {
            true => _v.value.y,
            false => _b.value.y
        },
        match _v.value.z < _b.value.z {
            true => _v.value.z,
            false => _b.value.z
        },
        match _v.value.w < _b.value.w {
            true => _v.value.w,
            false => _b.value.w
        });

    return Float4::new(
        match _a.value.x < min.value.x {
            true => _a.value.x,
            false => min.value.x
        },
        match _a.value.y < min.value.y {
            true => _a.value.y,
            false => min.value.y
        },
        match _a.value.z < min.value.z {
            true => _a.value.z,
            false => min.value.z
        },
        match _a.value.w < min.value.w {
            true => _a.value.w,
            false => min.value.w
        });
}

#[inline]
pub fn clamp3(_a: &Float3, _v: &Float3, _b: &Float3) -> Float3 {
    let min = Float3::new(
        match _v.value.x < _b.value.x {
            true => _v.value.x,
            false => _b.value.x
        },
        match _v.value.y < _b.value.y {
            true => _v.value.y,
            false => _b.value.y
        },
        match _v.value.z < _b.value.z {
            true => _v.value.z,
            false => _b.value.z
        });

    return Float3::new(
        match _a.value.x < min.value.x {
            true => _a.value.x,
            false => min.value.x
        },
        match _a.value.y < min.value.y {
            true => _a.value.y,
            false => min.value.y
        },
        match _a.value.z < min.value.z {
            true => _a.value.z,
            false => min.value.z
        });
}

#[inline]
pub fn clamp2(_a: &Float2, _v: &Float2, _b: &Float2) -> Float2 {
    let min = Float2::new(
        match _v.value.x < _b.value.x {
            true => _v.value.x,
            false => _b.value.x
        },
        match _v.value.y < _b.value.y {
            true => _v.value.y,
            false => _b.value.y
        });

    return Float2::new(
        match _a.value.x < min.value.x {
            true => _a.value.x,
            false => min.value.x
        },
        match _a.value.y < min.value.y {
            true => _a.value.y,
            false => min.value.y
        });
}