/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use std::ops::{Add, Sub, Neg, Mul, Div, BitAnd};

// Declares a 2d float vector.
#[derive(Clone, Copy)]
pub struct Float2 {
    pub x: f32,
    pub y: f32,
}

impl Float2 {
    // Constructs an uninitialized vector.
    #[inline]
    pub fn new_default() -> Float2 {
        return Float2 {
            x: 0.0,
            y: 0.0,
        };
    }

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

impl Float3 {
    // Constructs an uninitialized vector.
    #[inline]
    pub fn new_default() -> Float3 {
        return Float3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
    }

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

impl Float4 {
    // Constructs an uninitialized vector.
    #[inline]
    pub fn new_default() -> Float4 {
        return Float4 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 0.0,
        };
    }

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
// Returns the (horizontal) addition of each element of _v.
#[inline]
pub fn h_add4(_v: &Float4) -> f32 {
    return _v.x + _v.y + _v.z + _v.w;
}

#[inline]
pub fn h_add3(_v: &Float3) -> f32 {
    return _v.x + _v.y + _v.z;
}

#[inline]
pub fn h_add2(_v: &Float2) -> f32 {
    return _v.x + _v.y;
}

//--------------------------------------------------------------------------------------------------
// Returns the dot product of _a and _b.
#[inline]
pub fn dot4(_a: &Float4, _b: &Float4) -> f32 {
    return _a.x * _b.x + _a.y * _b.y + _a.z * _b.z + _a.w * _b.w;
}

#[inline]
pub fn dot3(_a: &Float3, _b: &Float3) -> f32 {
    return _a.x * _b.x + _a.y * _b.y + _a.z * _b.z;
}

#[inline]
pub fn dot2(_a: &Float2, _b: &Float2) -> f32 {
    return _a.x * _b.x + _a.y * _b.y;
}

//--------------------------------------------------------------------------------------------------
// Returns the cross product of _a and _b.
#[inline]
pub fn cross(_a: &Float3, _b: &Float3) -> Float3 {
    return Float3::new(_a.y * _b.z - _b.y * _a.z,
                       _a.z * _b.x - _b.z * _a.x,
                       _a.x * _b.y - _b.x * _a.y);
}

//--------------------------------------------------------------------------------------------------
// Returns the length |_v| of _v.
#[inline]
pub fn length4(_v: &Float4) -> f32 {
    let len2 = _v.x * _v.x + _v.y * _v.y + _v.z * _v.z + _v.w * _v.w;
    return f32::sqrt(len2);
}

#[inline]
pub fn length3(_v: &Float3) -> f32 {
    let len2 = _v.x * _v.x + _v.y * _v.y + _v.z * _v.z;
    return f32::sqrt(len2);
}

#[inline]
pub fn length2(_v: &Float2) -> f32 {
    let len2 = _v.x * _v.x + _v.y * _v.y;
    return f32::sqrt(len2);
}

//--------------------------------------------------------------------------------------------------
// Returns the square length |_v|^2 of _v.
#[inline]
pub fn length_sqr4(_v: &Float4) -> f32 {
    return _v.x * _v.x + _v.y * _v.y + _v.z * _v.z + _v.w * _v.w;
}

#[inline]
pub fn length_sqr3(_v: &Float3) -> f32 {
    return _v.x * _v.x + _v.y * _v.y + _v.z * _v.z;
}

#[inline]
pub fn length_sqr2(_v: &Float2) -> f32 {
    return _v.x * _v.x + _v.y * _v.y;
}

//--------------------------------------------------------------------------------------------------
// Returns the normalized vector _v.
#[inline]
pub fn normalize4(_v: &Float4) -> Float4 {
    let len2 = _v.x * _v.x + _v.y * _v.y + _v.z * _v.z + _v.w * _v.w;
    debug_assert!(len2 != 0.0 && "_v is not normalizable".parse().unwrap_or(true));
    let len = f32::sqrt(len2);
    return Float4::new(_v.x / len, _v.y / len, _v.z / len, _v.w / len);
}

#[inline]
pub fn normalize3(_v: &Float3) -> Float3 {
    let len2 = _v.x * _v.x + _v.y * _v.y + _v.z * _v.z;
    debug_assert!(len2 != 0.0 && "_v is not normalizable".parse().unwrap_or(true));
    let len = f32::sqrt(len2);
    return Float3::new(_v.x / len, _v.y / len, _v.z / len);
}

#[inline]
pub fn normalize2(_v: &Float2) -> Float2 {
    let len2 = _v.x * _v.x + _v.y * _v.y;
    debug_assert!(len2 != 0.0 && "_v is not normalizable".parse().unwrap_or(true));
    let len = f32::sqrt(len2);
    return Float2::new(_v.x / len, _v.y / len);
}

//--------------------------------------------------------------------------------------------------
// Returns true if _v is normalized.
#[inline]
pub fn is_normalized4(_v: &Float4) -> bool {
    let len2 = _v.x * _v.x + _v.y * _v.y + _v.z * _v.z + _v.w * _v.w;
    return f32::abs(len2 - 1.0) < crate::math_constant::K_NORMALIZATION_TOLERANCE_SQ;
}

#[inline]
pub fn is_normalized3(_v: &Float3) -> bool {
    let len2 = _v.x * _v.x + _v.y * _v.y + _v.z * _v.z;
    return f32::abs(len2 - 1.0) < crate::math_constant::K_NORMALIZATION_TOLERANCE_SQ;
}

#[inline]
pub fn is_normalized2(_v: &Float2) -> bool {
    let len2 = _v.x * _v.x + _v.y * _v.y;
    return f32::abs(len2 - 1.0) < crate::math_constant::K_NORMALIZATION_TOLERANCE_SQ;
}

//--------------------------------------------------------------------------------------------------
// Returns the normalized vector _v if the norm of _v is not 0.
// Otherwise returns _safer.
#[inline]
pub fn normalize_safe4(_v: &Float4, _safer: &Float4) -> Float4 {
    debug_assert!(is_normalized4(_safer) && "_safer is not normalized".parse().unwrap_or(true));
    let len2 = _v.x * _v.x + _v.y * _v.y + _v.z * _v.z + _v.w * _v.w;
    if len2 <= 0.0 {
        return _safer.clone();
    }
    let len = f32::sqrt(len2);
    return Float4::new(_v.x / len, _v.y / len, _v.z / len, _v.w / len);
}

#[inline]
pub fn normalize_safe3(_v: &Float3, _safer: &Float3) -> Float3 {
    debug_assert!(is_normalized3(_safer) && "_safer is not normalized".parse().unwrap_or(true));
    let len2 = _v.x * _v.x + _v.y * _v.y + _v.z * _v.z;
    if len2 <= 0.0 {
        return _safer.clone();
    }
    let len = f32::sqrt(len2);
    return Float3::new(_v.x / len, _v.y / len, _v.z / len);
}

#[inline]
pub fn normalize_safe2(_v: &Float2, _safer: &Float2) -> Float2 {
    debug_assert!(is_normalized2(_safer) && "_safer is not normalized".parse().unwrap_or(true));
    let len2 = _v.x * _v.x + _v.y * _v.y;
    if len2 <= 0.0 {
        return _safer.clone();
    }
    let len = f32::sqrt(len2);
    return Float2::new(_v.x / len, _v.y / len);
}

//--------------------------------------------------------------------------------------------------
// Returns the linear interpolation of _a and _b with coefficient _f.
// _f is not limited to range [0,1].
#[inline]
pub fn lerp4(_a: &Float4, _b: &Float4, _f: f32) -> Float4 {
    return Float4::new((_b.x - _a.x) * _f + _a.x, (_b.y - _a.y) * _f + _a.y,
                       (_b.z - _a.z) * _f + _a.z, (_b.w - _a.w) * _f + _a.w);
}

#[inline]
pub fn lerp3(_a: &Float3, _b: &Float3, _f: f32) -> Float3 {
    return Float3::new((_b.x - _a.x) * _f + _a.x, (_b.y - _a.y) * _f + _a.y,
                       (_b.z - _a.z) * _f + _a.z);
}

#[inline]
pub fn lerp2(_a: &Float2, _b: &Float2, _f: f32) -> Float2 {
    return Float2::new((_b.x - _a.x) * _f + _a.x, (_b.y - _a.y) * _f + _a.y);
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
// Returns the minimum of each element of _a and _b.
#[inline]
pub fn min4(_a: &Float4, _b: &Float4) -> Float4 {
    return Float4::new(
        match _a.x < _b.x {
            true => _a.x,
            false => _b.x
        },
        match _a.y < _b.y {
            true => _a.y,
            false => _b.y
        },
        match _a.z < _b.z {
            true => _a.z,
            false => _b.z
        },
        match _a.w < _b.w {
            true => _a.w,
            false => _b.w
        });
}

#[inline]
pub fn min3(_a: &Float3, _b: &Float3) -> Float3 {
    return Float3::new(
        match _a.x < _b.x {
            true => _a.x,
            false => _b.x
        },
        match _a.y < _b.y {
            true => _a.y,
            false => _b.y
        },
        match _a.z < _b.z {
            true => _a.z,
            false => _b.z
        });
}

#[inline]
pub fn min2(_a: &Float2, _b: &Float2) -> Float2 {
    return Float2::new(
        match _a.x < _b.x {
            true => _a.x,
            false => _b.x
        },
        match _a.y < _b.y {
            true => _a.y,
            false => _b.y
        });
}

//--------------------------------------------------------------------------------------------------
// Returns the maximum of each element of _a and _b.
#[inline]
pub fn max4(_a: &Float4, _b: &Float4) -> Float4 {
    return Float4::new(
        match _a.x > _b.x {
            true => _a.x,
            false => _b.x
        },
        match _a.y > _b.y {
            true => _a.y,
            false => _b.y
        },
        match _a.z > _b.z {
            true => _a.z,
            false => _b.z
        },
        match _a.w > _b.w {
            true => _a.w,
            false => _b.w
        });
}

#[inline]
pub fn max3(_a: &Float3, _b: &Float3) -> Float3 {
    return Float3::new(
        match _a.x > _b.x {
            true => _a.x,
            false => _b.x
        },
        match _a.y > _b.y {
            true => _a.y,
            false => _b.y
        },
        match _a.z > _b.z {
            true => _a.z,
            false => _b.z
        });
}

#[inline]
pub fn max2(_a: &Float2, _b: &Float2) -> Float2 {
    return Float2::new(
        match _a.x > _b.x {
            true => _a.x,
            false => _b.x
        },
        match _a.y > _b.y {
            true => _a.y,
            false => _b.y
        });
}

//--------------------------------------------------------------------------------------------------
// Clamps each element of _x between _a and _b.
// _a must be less or equal to b;
#[inline]
pub fn clamp4(_a: &Float4, _v: &Float4, _b: &Float4) -> Float4 {
    let min = Float4::new(
        match _v.x < _b.x {
            true => _v.x,
            false => _b.x
        },
        match _v.y < _b.y {
            true => _v.y,
            false => _b.y
        },
        match _v.z < _b.z {
            true => _v.z,
            false => _b.z
        },
        match _v.w < _b.w {
            true => _v.w,
            false => _b.w
        });

    return Float4::new(
        match _a.x < min.x {
            true => _a.x,
            false => min.x
        },
        match _a.y < min.y {
            true => _a.y,
            false => min.y
        },
        match _a.z < min.z {
            true => _a.z,
            false => min.z
        },
        match _a.w < min.w {
            true => _a.w,
            false => min.w
        });
}

#[inline]
pub fn clamp3(_a: &Float3, _v: &Float3, _b: &Float3) -> Float3 {
    let min = Float3::new(
        match _v.x < _b.x {
            true => _v.x,
            false => _b.x
        },
        match _v.y < _b.y {
            true => _v.y,
            false => _b.y
        },
        match _v.z < _b.z {
            true => _v.z,
            false => _b.z
        });

    return Float3::new(
        match _a.x < min.x {
            true => _a.x,
            false => min.x
        },
        match _a.y < min.y {
            true => _a.y,
            false => min.y
        },
        match _a.z < min.z {
            true => _a.z,
            false => min.z
        });
}

#[inline]
pub fn clamp2(_a: &Float2, _v: &Float2, _b: &Float2) -> Float2 {
    let min = Float2::new(
        match _v.x < _b.x {
            true => _v.x,
            false => _b.x
        },
        match _v.y < _b.y {
            true => _v.y,
            false => _b.y
        });

    return Float2::new(
        match _a.x < min.x {
            true => _a.x,
            false => min.x
        },
        match _a.y < min.y {
            true => _a.y,
            false => min.y
        });
}

//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
#[cfg(test)]
mod ozz_math {
    use crate::math_test_helper::*;
    use crate::*;
    use crate::transform::Transform;
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
}