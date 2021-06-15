/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use std::ops::{Add, Sub, Neg, Mul, Div, BitAnd};

pub trait FloatType: Clone {
    type ImplType;
    // Constructs an uninitialized vector.
    fn new_default() -> Self::ImplType;
}

// Declares a 1d float vector.
#[derive(Clone, Copy)]
pub struct Float {
    pub x: f32,
}

impl FloatType for Float {
    type ImplType = Float;
    #[inline]
    fn new_default() -> Self::ImplType {
        return Float {
            x: 0.0
        };
    }
}

//--------------------------------------------------------------------------------------------------
// Declares a 2d float vector.
#[derive(Clone, Copy)]
pub struct Float2 {
    pub x: f32,
    pub y: f32,
}

impl FloatType for Float2 {
    type ImplType = Float2;
    #[inline]
    fn new_default() -> Self::ImplType {
        return Float2 {
            x: 0.0,
            y: 0.0,
        };
    }
}

impl Float2 {
    // Constructs a vector initialized with _f value.
    #[inline]
    pub fn new_scalar(_f: f32) -> Float2 {
        return Float2 {
            x: _f,
            y: _f,
        };
    }

    // Constructs a vector initialized with _x and _y values.
    #[inline]
    pub fn new(_x: f32, _y: f32) -> Float2 {
        return Float2 {
            x: _x,
            y: _y,
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
#[derive(Clone, Copy)]
pub struct Float3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl FloatType for Float3 {
    type ImplType = Float3;
    #[inline]
    fn new_default() -> Self::ImplType {
        return Float3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
    }
}

impl Float3 {
    // Constructs a vector initialized with _f value.
    #[inline]
    pub fn new_scalar(_f: f32) -> Float3 {
        return Float3 {
            x: _f,
            y: _f,
            z: _f,
        };
    }

    // Constructs a vector initialized with _x, _y and _z values.
    #[inline]
    pub fn new(_x: f32, _y: f32, _z: f32) -> Float3 {
        return Float3 {
            x: _x,
            y: _y,
            z: _z,
        };
    }

    // Returns a vector initialized with _v.x, _v.y and _z values.
    #[inline]
    pub fn new2(_v: Float2, _z: f32) -> Float3 {
        return Float3 {
            x: _v.x,
            y: _v.y,
            z: _z,
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

    #[inline]
    pub fn to_vec(&self) -> [f32; 3] {
        return [self.x, self.y, self.z];
    }

    #[inline]
    pub fn to_vec4(&self) -> [f32; 4] {
        return [self.x, self.y, self.z, 0.0];
    }
}

//--------------------------------------------------------------------------------------------------
// Declares a 4d float vector.
#[derive(Clone, Copy)]
pub struct Float4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl FloatType for Float4 {
    type ImplType = Float4;
    #[inline]
    fn new_default() -> Self::ImplType {
        return Float4 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 0.0,
        };
    }
}

impl Float4 {
    // Constructs a vector initialized with _f value.
    #[inline]
    pub fn new_scalar(_f: f32) -> Float4 {
        return Float4 {
            x: _f,
            y: _f,
            z: _f,
            w: _f,
        };
    }

    // Constructs a vector initialized with _x, _y, _z and _w values.
    #[inline]
    pub fn new(_x: f32, _y: f32, _z: f32, _w: f32) -> Float4 {
        return Float4 {
            x: _x,
            y: _y,
            z: _z,
            w: _w,
        };
    }

    // Constructs a vector initialized with _v.x, _v.y, _v.z and _w values.
    #[inline]
    pub fn new3(_v: Float3, _w: f32) -> Float4 {
        return Float4 {
            x: _v.x,
            y: _v.y,
            z: _v.z,
            w: _w,
        };
    }

    // Constructs a vector initialized with _v.x, _v.y, _z and _w values.
    #[inline]
    pub fn new2(_v: Float2, _z: f32, _w: f32) -> Float4 {
        return Float4 {
            x: _v.x,
            y: _v.y,
            z: _z,
            w: _w,
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
                return Float4::new(self.x + rhs.x, self.y + rhs.y,
                                   self.z + rhs.z, self.w + rhs.w);
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
                return Float3::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z);
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
                return Float2::new(self.x + rhs.x, self.y + rhs.y);
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
                return Float4::new(self.x - rhs.x, self.y - rhs.y,
                                   self.z - rhs.z, self.w - rhs.w);
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
                return Float3::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z);
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
                return Float2::new(self.x - rhs.x, self.y - rhs.y);
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
        return Float4::new(-self.x, -self.y, -self.z, -self.w);
    }
}

impl Neg for Float3 {
    type Output = Float3;
    #[inline]
    fn neg(self) -> Self::Output {
        return Float3::new(-self.x, -self.y, -self.z);
    }
}

impl Neg for Float2 {
    type Output = Float2;
    #[inline]
    fn neg(self) -> Self::Output {
        return Float2::new(-self.x, -self.y);
    }
}

//--------------------------------------------------------------------------------------------------
macro_rules! impl_mul4 {
    ($rhs:ty) => {
        impl Mul for $rhs {
            type Output = Float4;
            #[inline]
            fn mul(self, rhs: Self) -> Self::Output {
                return Float4::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z, self.w * rhs.w);
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
                return Float3::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z);
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
                return Float2::new(self.x * rhs.x, self.y * rhs.y);
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
        return Float4::new(self.x * rhs, self.y * rhs, self.z * rhs, self.w * rhs);
    }
}

impl Mul<f32> for Float3 {
    type Output = Float3;
    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        return Float3::new(self.x * rhs, self.y * rhs, self.z * rhs);
    }
}

impl Mul<f32> for Float2 {
    type Output = Float2;
    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        return Float2::new(self.x * rhs, self.y * rhs);
    }
}

//--------------------------------------------------------------------------------------------------
macro_rules! impl_div4 {
    ($rhs:ty) => {
        impl Div for $rhs {
            type Output = Float4;
            #[inline]
            fn div(self, rhs: Self) -> Self::Output {
                return Float4::new(self.x / rhs.x, self.y / rhs.y, self.z / rhs.z, self.w / rhs.w);
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
                return Float3::new(self.x / rhs.x, self.y / rhs.y, self.z / rhs.z);
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
                return Float2::new(self.x / rhs.x, self.y / rhs.y);
            }
        }
    }
}
impl_div2!(Float2);
impl_div2!(&Float2);

//--------------------------------------------------------------------------------------------------
impl Div<f32> for Float4 {
    type Output = Float4;

    fn div(self, rhs: f32) -> Self::Output {
        return Float4::new(self.x / rhs, self.y / rhs, self.z / rhs, self.w / rhs);
    }
}

impl Div<f32> for Float3 {
    type Output = Float3;

    fn div(self, rhs: f32) -> Self::Output {
        return Float3::new(self.x / rhs, self.y / rhs, self.z / rhs);
    }
}

impl Div<f32> for Float2 {
    type Output = Float2;

    fn div(self, rhs: f32) -> Self::Output {
        return Float2::new(self.x / rhs, self.y / rhs);
    }
}

//--------------------------------------------------------------------------------------------------
impl Float4 {
    // Returns the (horizontal) addition of each element of self.
    #[inline]
    pub fn h_add(&self) -> f32 {
        return self.x + self.y + self.z + self.w;
    }
}

impl Float3 {
    #[inline]
    pub fn h_add(&self) -> f32 {
        return self.x + self.y + self.z;
    }
}

impl Float2 {
    #[inline]
    pub fn h_add(&self) -> f32 {
        return self.x + self.y;
    }
}

//--------------------------------------------------------------------------------------------------
impl Float4 {
    // Returns the dot product of _a and _b.
    #[inline]
    pub fn dot(&self, _b: &Float4) -> f32 {
        return self.x * _b.x + self.y * _b.y + self.z * _b.z + self.w * _b.w;
    }
}

impl Float3 {
    #[inline]
    pub fn dot(&self, _b: &Float3) -> f32 {
        return self.x * _b.x + self.y * _b.y + self.z * _b.z;
    }
}

impl Float2 {
    #[inline]
    pub fn dot(&self, _b: &Float2) -> f32 {
        return self.x * _b.x + self.y * _b.y;
    }
}

//--------------------------------------------------------------------------------------------------
impl Float3 {
    // Returns the cross product of _a and _b.
    #[inline]
    pub fn cross(&self, _b: &Float3) -> Float3 {
        return Float3::new(self.y * _b.z - _b.y * self.z,
                           self.z * _b.x - _b.z * self.x,
                           self.x * _b.y - _b.x * self.y);
    }
}

//--------------------------------------------------------------------------------------------------
impl Float4 {
    // Returns the length |self| of self.
    #[inline]
    pub fn length(&self) -> f32 {
        let len2 = self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w;
        return f32::sqrt(len2);
    }
}

impl Float3 {
    #[inline]
    pub fn length(&self) -> f32 {
        let len2 = self.x * self.x + self.y * self.y + self.z * self.z;
        return f32::sqrt(len2);
    }
}

impl Float2 {
    #[inline]
    pub fn length(&self) -> f32 {
        let len2 = self.x * self.x + self.y * self.y;
        return f32::sqrt(len2);
    }
}

//--------------------------------------------------------------------------------------------------
impl Float4 {
    // Returns the square length |self|^2 of self.
    #[inline]
    pub fn length_sqr(&self) -> f32 {
        return self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w;
    }
}

impl Float3 {
    #[inline]
    pub fn length_sqr(&self) -> f32 {
        return self.x * self.x + self.y * self.y + self.z * self.z;
    }
}

impl Float2 {
    #[inline]
    pub fn length_sqr(&self) -> f32 {
        return self.x * self.x + self.y * self.y;
    }
}

//--------------------------------------------------------------------------------------------------
impl Float4 {
    // Returns the normalized vector self.
    #[inline]
    pub fn normalize(&self) -> Float4 {
        let len2 = self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w;
        debug_assert!(len2 != 0.0 && "self is not normalizable".parse().unwrap_or(true));
        let len = f32::sqrt(len2);
        return Float4::new(self.x / len, self.y / len, self.z / len, self.w / len);
    }
}

impl Float3 {
    #[inline]
    pub fn normalize(&self) -> Float3 {
        let len2 = self.x * self.x + self.y * self.y + self.z * self.z;
        debug_assert!(len2 != 0.0 && "self is not normalizable".parse().unwrap_or(true));
        let len = f32::sqrt(len2);
        return Float3::new(self.x / len, self.y / len, self.z / len);
    }
}

impl Float2 {
    #[inline]
    pub fn normalize(&self) -> Float2 {
        let len2 = self.x * self.x + self.y * self.y;
        debug_assert!(len2 != 0.0 && "self is not normalizable".parse().unwrap_or(true));
        let len = f32::sqrt(len2);
        return Float2::new(self.x / len, self.y / len);
    }
}

//--------------------------------------------------------------------------------------------------
impl Float4 {
    // Returns true if self is normalized.
    #[inline]
    pub fn is_normalized(&self) -> bool {
        let len2 = self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w;
        return f32::abs(len2 - 1.0) < crate::math_constant::K_NORMALIZATION_TOLERANCE_SQ;
    }
}

impl Float3 {
    #[inline]
    pub fn is_normalized(&self) -> bool {
        let len2 = self.x * self.x + self.y * self.y + self.z * self.z;
        return f32::abs(len2 - 1.0) < crate::math_constant::K_NORMALIZATION_TOLERANCE_SQ;
    }
}

impl Float2 {
    #[inline]
    pub fn is_normalized(&self) -> bool {
        let len2 = self.x * self.x + self.y * self.y;
        return f32::abs(len2 - 1.0) < crate::math_constant::K_NORMALIZATION_TOLERANCE_SQ;
    }
}

//--------------------------------------------------------------------------------------------------
impl Float4 {
    // Returns the normalized vector self if the norm of self is not 0.
    // Otherwise returns _safer.
    #[inline]
    pub fn normalize_safe(&self, _safer: &Float4) -> Float4 {
        debug_assert!(_safer.is_normalized() && "_safer is not normalized".parse().unwrap_or(true));
        let len2 = self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w;
        if len2 <= 0.0 {
            return _safer.clone();
        }
        let len = f32::sqrt(len2);
        return Float4::new(self.x / len, self.y / len, self.z / len, self.w / len);
    }
}

impl Float3 {
    #[inline]
    pub fn normalize_safe(&self, _safer: &Float3) -> Float3 {
        debug_assert!(_safer.is_normalized() && "_safer is not normalized".parse().unwrap_or(true));
        let len2 = self.x * self.x + self.y * self.y + self.z * self.z;
        if len2 <= 0.0 {
            return _safer.clone();
        }
        let len = f32::sqrt(len2);
        return Float3::new(self.x / len, self.y / len, self.z / len);
    }
}

impl Float2 {
    #[inline]
    pub fn normalize_safe(&self, _safer: &Float2) -> Float2 {
        debug_assert!(_safer.is_normalized() && "_safer is not normalized".parse().unwrap_or(true));
        let len2 = self.x * self.x + self.y * self.y;
        if len2 <= 0.0 {
            return _safer.clone();
        }
        let len = f32::sqrt(len2);
        return Float2::new(self.x / len, self.y / len);
    }
}

//--------------------------------------------------------------------------------------------------
impl Float4 {
    // Returns the linear interpolation of _a and _b with coefficient _f.
    // _f is not limited to range [0,1].
    #[inline]
    pub fn lerp(&self, _b: &Float4, _f: f32) -> Float4 {
        return Float4::new((_b.x - self.x) * _f + self.x, (_b.y - self.y) * _f + self.y,
                           (_b.z - self.z) * _f + self.z, (_b.w - self.w) * _f + self.w);
    }
}

impl Float3 {
    #[inline]
    pub fn lerp(&self, _b: &Float3, _f: f32) -> Float3 {
        return Float3::new((_b.x - self.x) * _f + self.x, (_b.y - self.y) * _f + self.y,
                           (_b.z - self.z) * _f + self.z);
    }
}

impl Float2 {
    #[inline]
    pub fn lerp(&self, _b: &Float2, _f: f32) -> Float2 {
        return Float2::new((_b.x - self.x) * _f + self.x, (_b.y - self.y) * _f + self.y);
    }
}

//--------------------------------------------------------------------------------------------------
impl Float4 {
    // Returns true if the distance between _a and _b is less than _tolerance.
    #[inline]
    pub fn compare(&self, _b: &Float4, _tolerance: f32) -> bool {
        let diff = self - _b;
        return diff.dot(&diff) <= _tolerance * _tolerance;
    }
}

impl Float3 {
    #[inline]
    pub fn compare(&self, _b: &Float3, _tolerance: f32) -> bool {
        let diff = self - _b;
        return diff.dot(&diff) <= _tolerance * _tolerance;
    }
}

impl Float2 {
    #[inline]
    pub fn compare(&self, _b: &Float2, _tolerance: f32) -> bool {
        let diff = self - _b;
        return diff.dot(&diff) <= _tolerance * _tolerance;
    }
}

//--------------------------------------------------------------------------------------------------
impl Float4 {
    #[inline]
    pub fn lt(&self, other: &Self) -> bool {
        let x = self.x.lt(&other.x);
        let y = self.y.lt(&other.y);
        let z = self.z.lt(&other.z);
        let w = self.w.lt(&other.w);
        return x.bitand(y).bitand(z).bitand(w);
    }

    #[inline]
    pub fn le(&self, other: &Self) -> bool {
        let x = self.x.le(&other.x);
        let y = self.y.le(&other.y);
        let z = self.z.le(&other.z);
        let w = self.w.le(&other.w);
        return x.bitand(y).bitand(z).bitand(w);
    }

    #[inline]
    pub fn gt(&self, other: &Self) -> bool {
        let x = self.x.gt(&other.x);
        let y = self.y.gt(&other.y);
        let z = self.z.gt(&other.z);
        let w = self.w.gt(&other.w);
        return x.bitand(y).bitand(z).bitand(w);
    }

    #[inline]
    pub fn ge(&self, other: &Self) -> bool {
        let x = self.x.ge(&other.x);
        let y = self.y.ge(&other.y);
        let z = self.z.ge(&other.z);
        let w = self.w.ge(&other.w);
        return x.bitand(y).bitand(z).bitand(w);
    }

    #[inline]
    pub fn eq(&self, other: &Self) -> bool {
        let x = self.x.eq(&other.x);
        let y = self.y.eq(&other.y);
        let z = self.z.eq(&other.z);
        let w = self.w.eq(&other.w);
        return x.bitand(y).bitand(z).bitand(w);
    }

    #[inline]
    pub fn ne(&self, other: &Self) -> bool {
        let x = self.x.ne(&other.x);
        let y = self.y.ne(&other.y);
        let z = self.z.ne(&other.z);
        let w = self.w.ne(&other.w);
        return x.bitand(y).bitand(z).bitand(w);
    }
}

impl PartialEq for Float4 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        return self.x == other.x && self.y == other.y && self.z == other.z && self.w == other.w;
    }
}

impl Float3 {
    #[inline]
    pub fn lt(&self, other: &Self) -> bool {
        let x = self.x.lt(&other.x);
        let y = self.y.lt(&other.y);
        let z = self.z.lt(&other.z);
        return x.bitand(y).bitand(z);
    }

    #[inline]
    pub fn le(&self, other: &Self) -> bool {
        let x = self.x.le(&other.x);
        let y = self.y.le(&other.y);
        let z = self.z.le(&other.z);
        return x.bitand(y).bitand(z);
    }

    #[inline]
    pub fn gt(&self, other: &Self) -> bool {
        let x = self.x.gt(&other.x);
        let y = self.y.gt(&other.y);
        let z = self.z.gt(&other.z);
        return x.bitand(y).bitand(z);
    }

    #[inline]
    pub fn ge(&self, other: &Self) -> bool {
        let x = self.x.ge(&other.x);
        let y = self.y.ge(&other.y);
        let z = self.z.ge(&other.z);
        return x.bitand(y).bitand(z);
    }

    #[inline]
    pub fn eq(&self, other: &Self) -> bool {
        let x = self.x.eq(&other.x);
        let y = self.y.eq(&other.y);
        let z = self.z.eq(&other.z);
        return x.bitand(y).bitand(z);
    }

    #[inline]
    pub fn ne(&self, other: &Self) -> bool {
        let x = self.x.ne(&other.x);
        let y = self.y.ne(&other.y);
        let z = self.z.ne(&other.z);
        return x.bitand(y).bitand(z);
    }
}

impl PartialEq for Float3 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        return self.x == other.x && self.y == other.y && self.z == other.z;
    }
}

impl Float2 {
    #[inline]
    pub fn lt(&self, other: &Self) -> bool {
        let x = self.x.lt(&other.x);
        let y = self.y.lt(&other.y);
        return x.bitand(y);
    }

    #[inline]
    pub fn le(&self, other: &Self) -> bool {
        let x = self.x.le(&other.x);
        let y = self.y.le(&other.y);
        return x.bitand(y);
    }

    #[inline]
    pub fn gt(&self, other: &Self) -> bool {
        let x = self.x.gt(&other.x);
        let y = self.y.gt(&other.y);
        return x.bitand(y);
    }

    #[inline]
    pub fn ge(&self, other: &Self) -> bool {
        let x = self.x.ge(&other.x);
        let y = self.y.ge(&other.y);
        return x.bitand(y);
    }

    #[inline]
    pub fn eq(&self, other: &Self) -> bool {
        let x = self.x.eq(&other.x);
        let y = self.y.eq(&other.y);
        return x.bitand(y);
    }

    #[inline]
    pub fn ne(&self, other: &Self) -> bool {
        let x = self.x.ne(&other.x);
        let y = self.y.ne(&other.y);
        return x.bitand(y);
    }
}

impl PartialEq for Float2 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        return self.x == other.x && self.y == other.y;
    }
}

//--------------------------------------------------------------------------------------------------
impl Float4 {
    // Returns the minimum of each element of self and _b.
    #[inline]
    pub fn min(&self, _b: &Float4) -> Float4 {
        return Float4::new(
            match self.x < _b.x {
                true => self.x,
                false => _b.x
            },
            match self.y < _b.y {
                true => self.y,
                false => _b.y
            },
            match self.z < _b.z {
                true => self.z,
                false => _b.z
            },
            match self.w < _b.w {
                true => self.w,
                false => _b.w
            });
    }
}

impl Float3 {
    #[inline]
    pub fn min(&self, _b: &Float3) -> Float3 {
        return Float3::new(
            match self.x < _b.x {
                true => self.x,
                false => _b.x
            },
            match self.y < _b.y {
                true => self.y,
                false => _b.y
            },
            match self.z < _b.z {
                true => self.z,
                false => _b.z
            });
    }
}

impl Float2 {
    #[inline]
    pub fn min(&self, _b: &Float2) -> Float2 {
        return Float2::new(
            match self.x < _b.x {
                true => self.x,
                false => _b.x
            },
            match self.y < _b.y {
                true => self.y,
                false => _b.y
            });
    }
}

//--------------------------------------------------------------------------------------------------
impl Float4 {
    // Returns the maximum of each element of self and _b.
    #[inline]
    pub fn max(&self, _b: &Float4) -> Float4 {
        return Float4::new(
            match self.x > _b.x {
                true => self.x,
                false => _b.x
            },
            match self.y > _b.y {
                true => self.y,
                false => _b.y
            },
            match self.z > _b.z {
                true => self.z,
                false => _b.z
            },
            match self.w > _b.w {
                true => self.w,
                false => _b.w
            });
    }
}

impl Float3 {
    #[inline]
    pub fn max(&self, _b: &Float3) -> Float3 {
        return Float3::new(
            match self.x > _b.x {
                true => self.x,
                false => _b.x
            },
            match self.y > _b.y {
                true => self.y,
                false => _b.y
            },
            match self.z > _b.z {
                true => self.z,
                false => _b.z
            });
    }
}

impl Float2 {
    #[inline]
    pub fn max(&self, _b: &Float2) -> Float2 {
        return Float2::new(
            match self.x > _b.x {
                true => self.x,
                false => _b.x
            },
            match self.y > _b.y {
                true => self.y,
                false => _b.y
            });
    }
}

//--------------------------------------------------------------------------------------------------
impl Float4 {
    // Clamps each element of _x between self and _b.
    // _a must be less or equal to b;
    #[inline]
    pub fn clamp(&self, _a: &Float4, _b: &Float4) -> Float4 {
        let min = Float4::new(
            match self.x < _b.x {
                true => self.x,
                false => _b.x
            },
            match self.y < _b.y {
                true => self.y,
                false => _b.y
            },
            match self.z < _b.z {
                true => self.z,
                false => _b.z
            },
            match self.w < _b.w {
                true => self.w,
                false => _b.w
            });

        return Float4::new(
            match _a.x > min.x {
                true => _a.x,
                false => min.x
            },
            match _a.y > min.y {
                true => _a.y,
                false => min.y
            },
            match _a.z > min.z {
                true => _a.z,
                false => min.z
            },
            match _a.w > min.w {
                true => _a.w,
                false => min.w
            });
    }
}

impl Float3 {
    #[inline]
    pub fn clamp(&self, _a: &Float3, _b: &Float3) -> Float3 {
        let min = Float3::new(
            match self.x < _b.x {
                true => self.x,
                false => _b.x
            },
            match self.y < _b.y {
                true => self.y,
                false => _b.y
            },
            match self.z < _b.z {
                true => self.z,
                false => _b.z
            });

        return Float3::new(
            match _a.x > min.x {
                true => _a.x,
                false => min.x
            },
            match _a.y > min.y {
                true => _a.y,
                false => min.y
            },
            match _a.z > min.z {
                true => _a.z,
                false => min.z
            });
    }
}

impl Float2 {
    #[inline]
    pub fn clamp(&self, _a: &Float2, _b: &Float2) -> Float2 {
        let min = Float2::new(
            match self.x < _b.x {
                true => self.x,
                false => _b.x
            },
            match self.y < _b.y {
                true => self.y,
                false => _b.y
            });

        return Float2::new(
            match _a.x > min.x {
                true => _a.x,
                false => min.x
            },
            match _a.y > min.y {
                true => _a.y,
                false => min.y
            });
    }
}

//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
#[cfg(test)]
mod ozz_math {
    use crate::math_test_helper::*;
    use crate::*;
    use crate::vec_float::*;

    #[test]
    fn vector_load4() {
        expect_float4_eq!(Float4::new_scalar(46.0), 46.0, 46.0, 46.0, 46.0);
        expect_float4_eq!(Float4::new(-1.0, 0.0, 1.0, 2.0), -1.0, 0.0, 1.0, 2.0);
        let f3 = Float3::new(-1.0, 0.0, 1.0);
        expect_float4_eq!(Float4::new3(f3, 2.0), -1.0, 0.0, 1.0, 2.0);
        let f2 = Float2::new(-1.0, 0.0);
        expect_float4_eq!(Float4::new2(f2, 1.0, 2.0), -1.0, 0.0, 1.0, 2.0);
    }

    #[test]
    fn vector_load3() {
        expect_float3_eq!(Float3::new_scalar(46.0), 46.0, 46.0, 46.0);
        expect_float3_eq!(Float3::new(-1.0, 0.0, 1.0), -1.0, 0.0, 1.0);
        let f2 = Float2::new(-1.0, 0.0);
        expect_float3_eq!(Float3::new2(f2, 1.0), -1.0, 0.0, 1.0);
    }

    #[test]
    fn vector_load2() {
        expect_float2_eq!(Float2::new_scalar(46.0), 46.0, 46.0);
        expect_float2_eq!(Float2::new(-1.0, 0.0), -1.0, 0.0);
    }

    #[test]
    fn vector_constant4() {
        expect_float4_eq!(Float4::zero(), 0.0, 0.0, 0.0, 0.0);
        expect_float4_eq!(Float4::one(), 1.0, 1.0, 1.0, 1.0);
        expect_float4_eq!(Float4::x_axis(), 1.0, 0.0, 0.0, 0.0);
        expect_float4_eq!(Float4::y_axis(), 0.0, 1.0, 0.0, 0.0);
        expect_float4_eq!(Float4::z_axis(), 0.0, 0.0, 1.0, 0.0);
        expect_float4_eq!(Float4::w_axis(), 0.0, 0.0, 0.0, 1.0);
    }

    #[test]
    fn vector_constant3() {
        expect_float3_eq!(Float3::zero(), 0.0, 0.0, 0.0);
        expect_float3_eq!(Float3::one(), 1.0, 1.0, 1.0);
        expect_float3_eq!(Float3::x_axis(), 1.0, 0.0, 0.0);
        expect_float3_eq!(Float3::y_axis(), 0.0, 1.0, 0.0);
        expect_float3_eq!(Float3::z_axis(), 0.0, 0.0, 1.0);
    }

    #[test]
    fn vector_constant2() {
        expect_float2_eq!(Float2::zero(), 0.0, 0.0);
        expect_float2_eq!(Float2::one(), 1.0, 1.0);
        expect_float2_eq!(Float2::x_axis(), 1.0, 0.0);
        expect_float2_eq!(Float2::y_axis(), 0.0, 1.0);
    }

    #[test]
    fn vector_arithmetic4() {
        let a = Float4::new(0.5, 1.0, 2.0, 3.0);
        let b = Float4::new(4.0, 5.0, -6.0, 7.0);

        let add = a + b;
        expect_float4_eq!(add, 4.5, 6.0, -4.0, 10.0);

        let sub = a - b;
        expect_float4_eq!(sub, -3.5, -4.0, 8.0, -4.0);

        let neg = -b;
        expect_float4_eq!(neg, -4.0, -5.0, 6.0, -7.0);

        let mul = a * b;
        expect_float4_eq!(mul, 2.0, 5.0, -12.0, 21.0);

        let mul_scal = a * 2.0;
        expect_float4_eq!(mul_scal, 1.0, 2.0, 4.0, 6.0);

        let div = a / b;
        expect_float4_eq!(div, 0.5 / 4.0, 1.0 / 5.0, -2.0 / 6.0, 3.0 / 7.0);

        let div_scal = a / 2.0;
        expect_float4_eq!(div_scal, 0.5 / 2.0, 1.0 / 2.0, 2.0 / 2.0, 3.0 / 2.0);

        let hadd4 = a.h_add();
        assert_eq!(hadd4, 6.5);

        let dot = a.dot(&b);
        assert_eq!(dot, 16.0);

        let length = a.length();
        assert_eq!(length, f32::sqrt(14.25));

        let length2 = a.length_sqr();
        assert_eq!(length2, 14.25);

        // EXPECT_ASSERTION(Float4::zero().normalize(), "is not normalizable");
        assert_eq!(a.is_normalized(), false);
        let normalize = a.normalize();
        assert_eq!(normalize.is_normalized(), true);
        expect_float4_eq!(normalize, 0.13245323, 0.26490647, 0.52981293, 0.79471946);

        // EXPECT_ASSERTION(a.normalize_safe(&a), "_safer is not normalized");
        let safe = Float4::new(1.0, 0.0, 0.0, 0.0);
        let normalize_safe = a.normalize_safe(&safe);
        assert_eq!(normalize_safe.is_normalized(), true);
        expect_float4_eq!(normalize_safe, 0.13245323, 0.26490647, 0.52981293,
                         0.79471946);

        let normalize_safer = Float4::zero().normalize_safe(&safe);
        assert_eq!(normalize_safer.is_normalized(), true);
        expect_float4_eq!(normalize_safer, safe.x, safe.y, safe.z, safe.w);

        let lerp_0 = a.lerp(&b, 0.0);
        expect_float4_eq!(lerp_0, a.x, a.y, a.z, a.w);

        let lerp_1 = a.lerp(&b, 1.0);
        expect_float4_eq!(lerp_1, b.x, b.y, b.z, b.w);

        let lerp_0_5 = a.lerp(&b, 0.5);
        expect_float4_eq!(lerp_0_5, (a.x + b.x) * 0.5, (a.y + b.y) * 0.5,
                         (a.z + b.z) * 0.5, (a.w + b.w) * 0.5);

        let lerp_2 = a.lerp(&b, 2.0);
        expect_float4_eq!(lerp_2, 2.0 * b.x - a.x, 2.0 * b.y - a.y, 2.0 * b.z - a.z,
                         2.0 * b.w - a.w);
    }

    #[test]
    fn vector_arithmetic3() {
        let a = Float3::new(0.5, 1.0, 2.0);
        let b = Float3::new(4.0, 5.0, -6.0);

        let add = a + b;
        expect_float3_eq!(add, 4.5, 6.0, -4.0);

        let sub = a - b;
        expect_float3_eq!(sub, -3.5, -4.0, 8.0);

        let neg = -b;
        expect_float3_eq!(neg, -4.0, -5.0, 6.0);

        let mul = a * b;
        expect_float3_eq!(mul, 2.0, 5.0, -12.0);

        let mul_scal = a * 2.0;
        expect_float3_eq!(mul_scal, 1.0, 2.0, 4.0);

        let div = a / b;
        expect_float3_eq!(div, 0.5 / 4.0, 1.0 / 5.0, -2.0 / 6.0);

        let div_scal = a / 2.0;
        expect_float3_eq!(div_scal, 0.5 / 2.0, 1.0 / 2.0, 2.0 / 2.0);

        let hadd4 = a.h_add();
        assert_eq!(hadd4, 3.5);

        let dot = a.dot(&b);
        assert_eq!(dot, -5.0);

        let cross = a.cross(&b);
        expect_float3_eq!(cross, -16.0, 11.0, -1.5);

        let length = a.length();
        assert_eq!(length, f32::sqrt(5.25));

        let length2 = a.length_sqr();
        assert_eq!(length2, 5.25);

        // EXPECT_ASSERTION(Float3::zero().normalize(), "is not normalizable");
        assert_eq!(a.is_normalized(), false);
        let normalize = a.normalize();
        assert_eq!(normalize.is_normalized(), true);
        expect_float3_eq!(normalize, 0.21821788, 0.43643576, 0.87287152);

        // EXPECT_ASSERTION(a.normalize_safe(&a), "_safer is not normalized");
        let safe = Float3::new(1.0, 0.0, 0.0);
        let normalize_safe = a.normalize_safe(&safe);
        assert_eq!(normalize_safe.is_normalized(), true);
        expect_float3_eq!(normalize_safe, 0.21821788, 0.43643576, 0.87287152);

        let normalize_safer = Float3::zero().normalize_safe(&safe);
        assert_eq!(normalize_safer.is_normalized(), true);
        expect_float3_eq!(normalize_safer, safe.x, safe.y, safe.z);

        let lerp_0 = a.lerp(&b, 0.0);
        expect_float3_eq!(lerp_0, a.x, a.y, a.z);

        let lerp_1 = a.lerp(&b, 1.0);
        expect_float3_eq!(lerp_1, b.x, b.y, b.z);

        let lerp_0_5 = a.lerp(&b, 0.5);
        expect_float3_eq!(lerp_0_5, (a.x + b.x) * 0.5, (a.y + b.y) * 0.5,
                         (a.z + b.z) * 0.5);

        let lerp_2 = a.lerp(&b, 2.0);
        expect_float3_eq!(lerp_2, 2.0 * b.x - a.x, 2.0 * b.y - a.y, 2.0 * b.z - a.z);
    }

    #[test]
    fn vector_arithmetic2() {
        let a = Float2::new(0.5, 1.0);
        let b = Float2::new(4.0, 5.0);

        let add = a + b;
        expect_float2_eq!(add, 4.5, 6.0);

        let sub = a - b;
        expect_float2_eq!(sub, -3.5, -4.0);

        let neg = -b;
        expect_float2_eq!(neg, -4.0, -5.0);

        let mul = a * b;
        expect_float2_eq!(mul, 2.0, 5.0);

        let mul_scal = a * 2.0;
        expect_float2_eq!(mul_scal, 1.0, 2.0);

        let div = a / b;
        expect_float2_eq!(div, 0.5 / 4.0, 1.0 / 5.0);
        let div_scal = a / 2.0;
        expect_float2_eq!(div_scal, 0.5 / 2.0, 1.0 / 2.0);

        let hadd4 = a.h_add();
        assert_eq!(hadd4, 1.5);

        let dot = a.dot(&b);
        assert_eq!(dot, 7.0);

        let length = a.length();
        assert_eq!(length, f32::sqrt(1.25));

        let length2 = a.length_sqr();
        assert_eq!(length2, 1.25);

        // EXPECT_ASSERTION(Float2::zero().normalize(), "is not normalizable");
        assert_eq!(a.is_normalized(), false);
        let normalize = a.normalize();
        assert_eq!(normalize.is_normalized(), true);
        expect_float2_eq!(normalize, 0.44721359, 0.89442718);

        // EXPECT_ASSERTION(a.normalize_safe(&a), "_safer is not normalized");
        let safe = Float2::new(1.0, 0.0);
        let normalize_safe = a.normalize_safe(&safe);
        assert_eq!(normalize_safe.is_normalized(), true);
        expect_float2_eq!(normalize_safe, 0.44721359, 0.89442718);

        let normalize_safer = Float2::zero().normalize_safe(&safe);
        assert_eq!(normalize_safer.is_normalized(), true);
        expect_float2_eq!(normalize_safer, safe.x, safe.y);

        let lerp_0 = a.lerp(&b, 0.0);
        expect_float2_eq!(lerp_0, a.x, a.y);

        let lerp_1 = a.lerp(&b, 1.0);
        expect_float2_eq!(lerp_1, b.x, b.y);

        let lerp_0_5 = a.lerp(&b, 0.5);
        expect_float2_eq!(lerp_0_5, (a.x + b.x) * 0.5, (a.y + b.y) * 0.5);

        let lerp_2 = a.lerp(&b, 2.0);
        expect_float2_eq!(lerp_2, 2.0 * b.x - a.x, 2.0 * b.y - a.y);
    }

    #[test]
    fn vector_comparison4() {
        let a = Float4::new(0.5, 1.0, 2.0, 3.0);
        let b = Float4::new(4.0, 5.0, -6.0, 7.0);
        let c = Float4::new(4.0, 5.0, 6.0, 7.0);
        let d = Float4::new(4.0, 5.0, 6.0, 7.1);

        let min = a.min(&b);
        expect_float4_eq!(min, 0.5, 1.0, -6.0, 3.0);

        let max = a.max(&b);
        expect_float4_eq!(max, 4.0, 5.0, 2.0, 7.0);

        expect_float4_eq!(Float4::new(-12.0, 2.0, 9.0, 3.0).clamp(&a, &c), 0.5, 2.0, 6.0, 3.0);

        assert_eq!(a.lt(&c), true);
        assert_eq!(a.le(&c), true);
        assert_eq!(c.le(&c), true);

        assert_eq!(c.gt(&a), true);
        assert_eq!(c.ge(&a), true);
        assert_eq!(a.ge(&a), true);

        assert_eq!(a.eq(&a), true);
        assert_eq!(a.ne(&b), true);

        assert_eq!(a.compare(&a, 0.0), true);
        assert_eq!(c.compare(&d, 0.2), true);
        assert_eq!(c.compare(&d, 0.05), false);
    }

    #[test]
    fn vector_comparison3() {
        let a = Float3::new(0.5, -1.0, 2.0);
        let b = Float3::new(4.0, 5.0, -6.0);
        let c = Float3::new(4.0, 5.0, 6.0);
        let d = Float3::new(4.0, 5.0, 6.1);

        let min = a.min(&b);
        expect_float3_eq!(min, 0.5, -1.0, -6.0);

        let max = a.max(&b);
        expect_float3_eq!(max, 4.0, 5.0, 2.0);

        expect_float3_eq!(Float3::new(-12.0, 2.0, 9.0).clamp(&a, &c), 0.5, 2.0, 6.0);

        assert_eq!(a.lt(&c), true);
        assert_eq!(a.le(&c), true);
        assert_eq!(c.le(&c), true);

        assert_eq!(c.gt(&a), true);
        assert_eq!(c.ge(&a), true);
        assert_eq!(a.ge(&a), true);

        assert_eq!(a.eq(&a), true);
        assert_eq!(a.ne(&b), true);

        assert_eq!(a.compare(&a, 1e-3), true);
        assert_eq!(c.compare(&d, 0.2), true);
        assert_eq!(c.compare(&d, 0.05), false);
    }

    #[test]
    fn vector_comparison2() {
        let a = Float2::new(0.5, 1.0);
        let b = Float2::new(4.0, -5.0);
        let c = Float2::new(4.0, 5.0);
        let d = Float2::new(4.0, 5.1);

        let min = a.min(&b);
        expect_float2_eq!(min, 0.5, -5.0);

        let max = a.max(&b);
        expect_float2_eq!(max, 4.0, 1.0);

        expect_float2_eq!(Float2::new(-12.0, 2.0).clamp(&a, &c), 0.5, 2.0);

        assert_eq!(a.lt(&c), true);
        assert_eq!(a.le(&c), true);
        assert_eq!(c.le(&c), true);

        assert_eq!(c.gt(&a), true);
        assert_eq!(c.ge(&a), true);
        assert_eq!(a.ge(&a), true);

        assert_eq!(a.eq(&a), true);
        assert_eq!(a.ne(&b), true);

        assert_eq!(a.compare(&a, 1e-3), true);
        assert_eq!(c.compare(&d, 0.2), true);
        assert_eq!(c.compare(&d, 0.05), false);
    }
}