/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::math_constant::*;
use packed_simd_2::{f32x4, m32x4};
use std::ops::{Add, Sub, Neg, Mul, Div, BitAnd};

#[derive(Clone)]
pub struct SoaFloat2 {
    pub x: f32x4,
    pub y: f32x4,
}

impl SoaFloat2 {
    pub fn load(_x: f32x4, _y: f32x4) -> SoaFloat2 {
        return SoaFloat2 {
            x: _x,
            y: _y,
        };
    }

    #[inline]
    pub fn zero() -> SoaFloat2 {
        return SoaFloat2 {
            x: f32x4::new(0.0, 0.0, 0.0, 0.0),
            y: f32x4::new(0.0, 0.0, 0.0, 0.0),
        };
    }

    #[inline]
    pub fn one() -> SoaFloat2 {
        return SoaFloat2 {
            x: f32x4::new(1.0, 1.0, 1.0, 1.0),
            y: f32x4::new(1.0, 1.0, 1.0, 1.0),
        };
    }

    #[inline]
    pub fn x_axis() -> SoaFloat2 {
        return SoaFloat2 {
            x: f32x4::new(1.0, 1.0, 1.0, 1.0),
            y: f32x4::new(0.0, 0.0, 0.0, 0.0),
        };
    }

    #[inline]
    pub fn y_axis() -> SoaFloat2 {
        return SoaFloat2 {
            x: f32x4::new(0.0, 0.0, 0.0, 0.0),
            y: f32x4::new(1.0, 1.0, 1.0, 1.0),
        };
    }
}

//--------------------------------------------------------------------------------------------------
#[derive(Clone)]
pub struct SoaFloat3 {
    pub x: f32x4,
    pub y: f32x4,
    pub z: f32x4,
}

impl SoaFloat3 {
    #[inline]
    pub fn load(_x: f32x4, _y: f32x4,
                _z: f32x4) -> SoaFloat3 {
        return SoaFloat3 {
            x: _x,
            y: _y,
            z: _z,
        };
    }

    #[inline]
    pub fn loa2d(_v: &SoaFloat2, _z: f32x4) -> SoaFloat3 {
        return SoaFloat3 {
            x: _v.x,
            y: _v.y,
            z: _z,
        };
    }

    #[inline]
    pub fn zero() -> SoaFloat3 {
        return SoaFloat3 {
            x: f32x4::new(0.0, 0.0, 0.0, 0.0),
            y: f32x4::new(0.0, 0.0, 0.0, 0.0),
            z: f32x4::new(0.0, 0.0, 0.0, 0.0),
        };
    }

    #[inline]
    pub fn one() -> SoaFloat3 {
        return SoaFloat3 {
            x: f32x4::new(1.0, 1.0, 1.0, 1.0),
            y: f32x4::new(1.0, 1.0, 1.0, 1.0),
            z: f32x4::new(1.0, 1.0, 1.0, 1.0),
        };
    }

    #[inline]
    pub fn x_axis() -> SoaFloat3 {
        return SoaFloat3 {
            x: f32x4::new(1.0, 1.0, 1.0, 1.0),
            y: f32x4::new(0.0, 0.0, 0.0, 0.0),
            z: f32x4::new(0.0, 0.0, 0.0, 0.0),
        };
    }

    #[inline]
    pub fn y_axis() -> SoaFloat3 {
        return SoaFloat3 {
            x: f32x4::new(0.0, 0.0, 0.0, 0.0),
            y: f32x4::new(1.0, 1.0, 1.0, 1.0),
            z: f32x4::new(0.0, 0.0, 0.0, 0.0),
        };
    }

    #[inline]
    pub fn z_axis() -> SoaFloat3 {
        return SoaFloat3 {
            x: f32x4::new(0.0, 0.0, 0.0, 0.0),
            y: f32x4::new(0.0, 0.0, 0.0, 0.0),
            z: f32x4::new(1.0, 1.0, 1.0, 1.0),
        };
    }
}

//--------------------------------------------------------------------------------------------------
#[derive(Clone)]
pub struct SoaFloat4 {
    pub x: f32x4,
    pub y: f32x4,
    pub z: f32x4,
    pub w: f32x4,
}

impl SoaFloat4 {
    #[inline]
    pub fn load(_x: f32x4, _y: f32x4,
                _z: f32x4, _w: f32x4) -> SoaFloat4 {
        return SoaFloat4 {
            x: _x,
            y: _y,
            z: _z,
            w: _w,
        };
    }

    #[inline]
    pub fn load3(_v: &SoaFloat3, _w: f32x4) -> SoaFloat4 {
        return SoaFloat4 {
            x: _v.x,
            y: _v.y,
            z: _v.z,
            w: _w,
        };
    }

    pub fn load2(_v: &SoaFloat2, _z: f32x4,
                 _w: f32x4) -> SoaFloat4 {
        return SoaFloat4 {
            x: _v.x,
            y: _v.y,
            z: _z,
            w: _w,
        };
    }

    #[inline]
    pub fn zero() -> SoaFloat4 {
        let zero = f32x4::new(0.0, 0.0, 0.0, 0.0);
        return SoaFloat4 {
            x: zero,
            y: zero,
            z: zero,
            w: zero,
        };
    }

    #[inline]
    pub fn one() -> SoaFloat4 {
        let one = f32x4::new(1.0, 1.0, 1.0, 1.0);
        return SoaFloat4 {
            x: one,
            y: one,
            z: one,
            w: one,
        };
    }

    #[inline]
    pub fn x_axis() -> SoaFloat4 {
        let zero = f32x4::new(0.0, 0.0, 0.0, 0.0);
        let one = f32x4::new(1.0, 1.0, 1.0, 1.0);
        return SoaFloat4 {
            x: one,
            y: zero,
            z: zero,
            w: zero,
        };
    }

    #[inline]
    pub fn y_axis() -> SoaFloat4 {
        let zero = f32x4::new(0.0, 0.0, 0.0, 0.0);
        let one = f32x4::new(1.0, 1.0, 1.0, 1.0);
        return SoaFloat4 {
            x: zero,
            y: one,
            z: zero,
            w: zero,
        };
    }

    #[inline]
    pub fn z_axis() -> SoaFloat4 {
        let zero = f32x4::new(0.0, 0.0, 0.0, 0.0);
        let one = f32x4::new(1.0, 1.0, 1.0, 1.0);
        return SoaFloat4 {
            x: zero,
            y: zero,
            z: one,
            w: zero,
        };
    }

    #[inline]
    pub fn w_axis() -> SoaFloat4 {
        let zero = f32x4::new(0.0, 0.0, 0.0, 0.0);
        let one = f32x4::new(1.0, 1.0, 1.0, 1.0);
        return SoaFloat4 {
            x: zero,
            y: zero,
            z: zero,
            w: one,
        };
    }
}

//--------------------------------------------------------------------------------------------------
// Returns per element addition of _a and _b using operator +.
macro_rules! impl_add4 {
    ($rhs:ty) => {
        impl Add for $rhs {
            type Output = SoaFloat4;
            #[inline]
            fn add(self, rhs: Self) -> Self::Output {
                return SoaFloat4::load(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z,
                                       self.w + rhs.w);
            }
        }
    }
}
impl_add4!(SoaFloat4);
impl_add4!(&SoaFloat4);

macro_rules! impl_add3 {
    ($rhs:ty) => {
        impl Add for $rhs {
            type Output = SoaFloat3;
            #[inline]
            fn add(self, rhs: Self) -> Self::Output {
                return SoaFloat3::load(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z);
            }
        }
    }
}
impl_add3!(SoaFloat3);
impl_add3!(&SoaFloat3);

macro_rules! impl_add2 {
    ($rhs:ty) => {
        impl Add for $rhs {
            type Output = SoaFloat2;
            #[inline]
            fn add(self, rhs: Self) -> Self::Output {
                return SoaFloat2::load(self.x + rhs.x, self.y + rhs.y);
            }
        }
    }
}
impl_add2!(SoaFloat2);
impl_add2!(&SoaFloat2);

//--------------------------------------------------------------------------------------------------
// Returns per element subtraction of _a and _b using operator -.
macro_rules! impl_sub4 {
    ($rhs:ty) => {
        impl Sub for $rhs {
            type Output = SoaFloat4;
            #[inline]
            fn sub(self, rhs: Self) -> Self::Output {
                return SoaFloat4::load(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z,
                                       self.w - rhs.w);
            }
        }
    }
}
impl_sub4!(SoaFloat4);
impl_sub4!(&SoaFloat4);

macro_rules! impl_sub3 {
    ($rhs:ty) => {
        impl Sub for $rhs {
            type Output = SoaFloat3;
            #[inline]
            fn sub(self, rhs: Self) -> Self::Output {
                return SoaFloat3::load(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z);
            }
        }
    }
}
impl_sub3!(SoaFloat3);
impl_sub3!(&SoaFloat3);

macro_rules! impl_sub2 {
    ($rhs:ty) => {
        impl Sub for $rhs {
            type Output = SoaFloat2;
            #[inline]
            fn sub(self, rhs: Self) -> Self::Output {
                return SoaFloat2::load(self.x - rhs.x, self.y - rhs.y);
            }
        }
    }
}
impl_sub2!(SoaFloat2);
impl_sub2!(&SoaFloat2);

//--------------------------------------------------------------------------------------------------
// Returns per element negative value of _v.
impl Neg for SoaFloat4 {
    type Output = SoaFloat4;
    #[inline]
    fn neg(self) -> Self::Output {
        return SoaFloat4::load(-self.x, -self.y, -self.z, -self.w);
    }
}

impl Neg for SoaFloat3 {
    type Output = SoaFloat3;
    #[inline]
    fn neg(self) -> Self::Output {
        return SoaFloat3::load(-self.x, -self.y, -self.z);
    }
}

impl Neg for SoaFloat2 {
    type Output = SoaFloat2;
    #[inline]
    fn neg(self) -> Self::Output {
        return SoaFloat2::load(-self.x, -self.y);
    }
}

//--------------------------------------------------------------------------------------------------
// Returns per element multiplication of _a and _b using operator *.
macro_rules! impl_mul4 {
    ($rhs:ty) => {
        impl Mul for $rhs {
            type Output = SoaFloat4;
            #[inline]
            fn mul(self, rhs: Self) -> Self::Output {
                return SoaFloat4::load(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z,
                                       self.w * rhs.w);
            }
        }
    }
}
impl_mul4!(SoaFloat4);
impl_mul4!(&SoaFloat4);

macro_rules! impl_mul3 {
    ($rhs:ty) => {
        impl Mul for $rhs {
            type Output = SoaFloat3;
            #[inline]
            fn mul(self, rhs: Self) -> Self::Output {
                return SoaFloat3::load(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z);
            }
        }
    }
}
impl_mul3!(SoaFloat3);
impl_mul3!(&SoaFloat3);

macro_rules! impl_mul2 {
    ($rhs:ty) => {
        impl Mul for $rhs {
            type Output = SoaFloat2;
            #[inline]
            fn mul(self, rhs: Self) -> Self::Output {
                return SoaFloat2::load(self.x * rhs.x, self.y * rhs.y);
            }
        }
    }
}
impl_mul2!(SoaFloat2);
impl_mul2!(&SoaFloat2);

//--------------------------------------------------------------------------------------------------
// Returns per element multiplication of _a and scalar value _f using operator *.
impl Mul<f32x4> for SoaFloat4 {
    type Output = SoaFloat4;
    #[inline]
    fn mul(self, rhs: f32x4) -> Self::Output {
        return SoaFloat4::load(self.x * rhs, self.y * rhs, self.z * rhs,
                               self.w * rhs);
    }
}

impl Mul<f32x4> for SoaFloat3 {
    type Output = SoaFloat3;
    #[inline]
    fn mul(self, rhs: f32x4) -> Self::Output {
        return SoaFloat3::load(self.x * rhs, self.y * rhs, self.z * rhs);
    }
}

impl Mul<f32x4> for SoaFloat2 {
    type Output = SoaFloat2;
    #[inline]
    fn mul(self, rhs: f32x4) -> Self::Output {
        return SoaFloat2::load(self.x * rhs, self.y * rhs);
    }
}

//--------------------------------------------------------------------------------------------------
// Multiplies _a and _b, then adds _addend.
// v = (_a * _b) + _addend
#[inline]
pub fn m_add2(_a: &SoaFloat2,
              _b: &SoaFloat2,
              _addend: &SoaFloat2) -> SoaFloat2 {
    return SoaFloat2 {
        x: _a.x.mul_add(_b.x, _addend.x),
        y: _a.y.mul_add(_b.y, _addend.y),
    };
}

#[inline]
pub fn m_add3(_a: &SoaFloat3,
              _b: &SoaFloat3,
              _addend: &SoaFloat3) -> SoaFloat3 {
    return SoaFloat3 {
        x: _a.x.mul_add(_b.x, _addend.x),
        y: _a.y.mul_add(_b.y, _addend.y),
        z: _a.z.mul_add(_b.z, _addend.z),
    };
}

#[inline]
pub fn m_add4(_a: &SoaFloat4,
              _b: &SoaFloat4,
              _addend: &SoaFloat4) -> SoaFloat4 {
    return SoaFloat4 {
        x: _a.x.mul_add(_b.x, _addend.x),
        y: _a.y.mul_add(_b.y, _addend.y),
        z: _a.z.mul_add(_b.z, _addend.z),
        w: _a.w.mul_add(_b.w, _addend.w),
    };
}

//--------------------------------------------------------------------------------------------------
// Returns per element division of _a and _b using operator /.
macro_rules! impl_div4 {
    ($rhs:ty) => {
        impl Div for $rhs {
            type Output = SoaFloat4;
            #[inline]
            fn div(self, rhs: Self) -> Self::Output {
                return SoaFloat4::load(self.x / rhs.x, self.y / rhs.y, self.z / rhs.z,
                                       self.w / rhs.w);
            }
        }
    };
}
impl_div4!(SoaFloat4);
impl_div4!(&SoaFloat4);

macro_rules! impl_div3 {
    ($rhs:ty) => {
        impl Div for $rhs {
            type Output = SoaFloat3;
            #[inline]
            fn div(self, rhs: Self) -> Self::Output {
                return SoaFloat3::load(self.x / rhs.x, self.y / rhs.y, self.z / rhs.z);
            }
        }
    }
}
impl_div3!(SoaFloat3);
impl_div3!(&SoaFloat3);

macro_rules! impl_div2 {
    ($rhs:ty) => {
        impl Div for $rhs {
            type Output = SoaFloat2;
            #[inline]
            fn div(self, rhs: Self) -> Self::Output {
                return SoaFloat2::load(self.x / rhs.x, self.y / rhs.y);
            }
        }
    }
}
impl_div2!(SoaFloat2);
impl_div2!(&SoaFloat2);

//--------------------------------------------------------------------------------------------------
// Returns per element division of _a and scalar value _f using operator/.
impl Div<f32x4> for SoaFloat4 {
    type Output = SoaFloat4;
    #[inline]
    fn div(self, rhs: f32x4) -> Self::Output {
        return SoaFloat4::load(self.x / rhs, self.y / rhs, self.z / rhs,
                               self.w / rhs);
    }
}

impl Div<f32x4> for SoaFloat3 {
    type Output = SoaFloat3;
    #[inline]
    fn div(self, rhs: f32x4) -> Self::Output {
        return SoaFloat3::load(self.x / rhs, self.y / rhs, self.z / rhs);
    }
}

impl Div<f32x4> for SoaFloat2 {
    type Output = SoaFloat2;
    #[inline]
    fn div(self, rhs: f32x4) -> Self::Output {
        return SoaFloat2::load(self.x / rhs, self.y / rhs);
    }
}

//--------------------------------------------------------------------------------------------------
// Returns true if each element of a is less than each element of _b.
impl SoaFloat4 {
    #[inline]
    pub fn lt(&self, other: &Self) -> m32x4 {
        let x = self.x.lt(other.x);
        let y = self.y.lt(other.y);
        let z = self.z.lt(other.z);
        let w = self.w.lt(other.w);
        return x.bitand(y).bitand(z).bitand(w);
    }

    #[inline]
    pub fn le(&self, other: &Self) -> m32x4 {
        let x = self.x.le(other.x);
        let y = self.y.le(other.y);
        let z = self.z.le(other.z);
        let w = self.w.le(other.w);
        return x.bitand(y).bitand(z).bitand(w);
    }

    #[inline]
    pub fn gt(&self, other: &Self) -> m32x4 {
        let x = self.x.gt(other.x);
        let y = self.y.gt(other.y);
        let z = self.z.gt(other.z);
        let w = self.w.gt(other.w);
        return x.bitand(y).bitand(z).bitand(w);
    }

    #[inline]
    pub fn ge(&self, other: &Self) -> m32x4 {
        let x = self.x.ge(other.x);
        let y = self.y.ge(other.y);
        let z = self.z.ge(other.z);
        let w = self.w.ge(other.w);
        return x.bitand(y).bitand(z).bitand(w);
    }

    #[inline]
    pub fn eq(&self, other: &Self) -> m32x4 {
        let x = self.x.eq(other.x);
        let y = self.y.eq(other.y);
        let z = self.z.eq(other.z);
        let w = self.w.eq(other.w);
        return x.bitand(y).bitand(z).bitand(w);
    }

    #[inline]
    pub fn ne(&self, other: &Self) -> m32x4 {
        let x = self.x.ne(other.x);
        let y = self.y.ne(other.y);
        let z = self.z.ne(other.z);
        let w = self.w.ne(other.w);
        return x.bitand(y).bitand(z).bitand(w);
    }
}

impl PartialEq for SoaFloat4 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        let x = self.x == other.x;
        let y = self.y == other.y;
        let z = self.z == other.z;
        let w = self.w == other.w;
        return x && y && z && w;
    }
}

impl SoaFloat3 {
    #[inline]
    pub fn lt(&self, other: &Self) -> m32x4 {
        let x = self.x.lt(other.x);
        let y = self.y.lt(other.y);
        let z = self.z.lt(other.z);
        return x.bitand(y).bitand(z);
    }

    #[inline]
    pub fn le(&self, other: &Self) -> m32x4 {
        let x = self.x.le(other.x);
        let y = self.y.le(other.y);
        let z = self.z.le(other.z);
        return x.bitand(y).bitand(z);
    }

    #[inline]
    pub fn gt(&self, other: &Self) -> m32x4 {
        let x = self.x.gt(other.x);
        let y = self.y.gt(other.y);
        let z = self.z.gt(other.z);
        return x.bitand(y).bitand(z);
    }

    #[inline]
    pub fn ge(&self, other: &Self) -> m32x4 {
        let x = self.x.ge(other.x);
        let y = self.y.ge(other.y);
        let z = self.z.ge(other.z);
        return x.bitand(y).bitand(z);
    }

    #[inline]
    pub fn eq(&self, other: &Self) -> m32x4 {
        let x = self.x.eq(other.x);
        let y = self.y.eq(other.y);
        let z = self.z.eq(other.z);
        return x.bitand(y).bitand(z);
    }

    #[inline]
    pub fn ne(&self, other: &Self) -> m32x4 {
        let x = self.x.ne(other.x);
        let y = self.y.ne(other.y);
        let z = self.z.ne(other.z);
        return x.bitand(y).bitand(z);
    }
}

impl PartialEq for SoaFloat3 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        let x = self.x == other.x;
        let y = self.y == other.y;
        let z = self.z == other.z;
        return x && y && z;
    }
}

impl SoaFloat2 {
    #[inline]
    pub fn lt(&self, other: &Self) -> m32x4 {
        let x = self.x.lt(other.x);
        let y = self.y.lt(other.y);
        return x.bitand(y);
    }

    #[inline]
    pub fn le(&self, other: &Self) -> m32x4 {
        let x = self.x.le(other.x);
        let y = self.y.le(other.y);
        return x.bitand(y);
    }

    #[inline]
    pub fn gt(&self, other: &Self) -> m32x4 {
        let x = self.x.gt(other.x);
        let y = self.y.gt(other.y);
        return x.bitand(y);
    }

    #[inline]
    pub fn ge(&self, other: &Self) -> m32x4 {
        let x = self.x.ge(other.x);
        let y = self.y.ge(other.y);
        return x.bitand(y);
    }

    #[inline]
    pub fn eq(&self, other: &Self) -> m32x4 {
        let x = self.x.eq(other.x);
        let y = self.y.eq(other.y);
        return x.bitand(y);
    }

    #[inline]
    pub fn ne(&self, other: &Self) -> m32x4 {
        let x = self.x.ne(other.x);
        let y = self.y.ne(other.y);
        return x.bitand(y);
    }
}

impl PartialEq for SoaFloat2 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        let x = self.x == other.x;
        let y = self.y == other.y;
        return x && y;
    }
}

//--------------------------------------------------------------------------------------------------
// Returns the (horizontal) addition of each element of _v.
pub fn h_add4(_v: &SoaFloat4) -> f32x4 { return _v.x + _v.y + _v.z + _v.w; }

pub fn h_add3(_v: &SoaFloat3) -> f32x4 { return _v.x + _v.y + _v.z; }

pub fn h_add2(_v: &SoaFloat2) -> f32x4 { return _v.x + _v.y; }

//--------------------------------------------------------------------------------------------------
// Returns the dot product of _a and _b.
pub fn dot4(_a: &SoaFloat4, _b: &SoaFloat4) -> f32x4 {
    return _a.x * _b.x + _a.y * _b.y + _a.z * _b.z + _a.w * _b.w;
}

pub fn dot3(_a: &SoaFloat3, _b: &SoaFloat3) -> f32x4 {
    return _a.x * _b.x + _a.y * _b.y + _a.z * _b.z;
}

pub fn dot2(_a: &SoaFloat2, _b: &SoaFloat2) -> f32x4 {
    return _a.x * _b.x + _a.y * _b.y;
}

//--------------------------------------------------------------------------------------------------
// Returns the cross product of _a and _b.
pub fn cross(_a: &SoaFloat3, _b: &SoaFloat3) -> SoaFloat3 {
    return SoaFloat3::load(_a.y * _b.z - _b.y * _a.z,
                           _a.z * _b.x - _b.z * _a.x,
                           _a.x * _b.y - _b.x * _a.y);
}

//--------------------------------------------------------------------------------------------------
// Returns the length |_v| of _v.
pub fn length4(_v: &SoaFloat4) -> f32x4 {
    let len2 = _v.x * _v.x + _v.y * _v.y + _v.z * _v.z + _v.w * _v.w;
    return len2.sqrt();
}

pub fn length3(_v: &SoaFloat3) -> f32x4 {
    let len2 = _v.x * _v.x + _v.y * _v.y + _v.z * _v.z;
    return len2.sqrt();
}

pub fn length2(_v: &SoaFloat2) -> f32x4 {
    let len2 = _v.x * _v.x + _v.y * _v.y;
    return len2.sqrt();
}

//--------------------------------------------------------------------------------------------------
// Returns the square length |_v|^2 of _v.
pub fn length_sqr4(_v: &SoaFloat4) -> f32x4 {
    return _v.x * _v.x + _v.y * _v.y + _v.z * _v.z + _v.w * _v.w;
}

pub fn length_sqr3(_v: &SoaFloat3) -> f32x4 {
    return _v.x * _v.x + _v.y * _v.y + _v.z * _v.z;
}

pub fn length_sqr2(_v: &SoaFloat2) -> f32x4 {
    return _v.x * _v.x + _v.y * _v.y;
}

//--------------------------------------------------------------------------------------------------
// Returns the normalized vector _v.
pub fn normalize4(_v: &SoaFloat4) -> SoaFloat4 {
    let len2 = _v.x * _v.x + _v.y * _v.y + _v.z * _v.z + _v.w * _v.w;
    debug_assert!(len2.ne(f32x4::new(0.0, 0.0, 0.0, 0.0)).all()
        && "_v is not normalizable".parse().unwrap());

    let inv_len = f32x4::new(1.0, 1.0, 1.0, 1.0) / len2.sqrt();
    return SoaFloat4::load(
        _v.x * inv_len, _v.y * inv_len, _v.z * inv_len,
        _v.w * inv_len,
    );
}

pub fn normalize3(_v: &SoaFloat3) -> SoaFloat3 {
    let len2 = _v.x * _v.x + _v.y * _v.y + _v.z * _v.z;
    debug_assert!(len2.ne(f32x4::new(0.0, 0.0, 0.0, 0.0)).all()
        && "_v is not normalizable".parse().unwrap());

    let inv_len = f32x4::new(1.0, 1.0, 1.0, 1.0) / len2.sqrt();
    return SoaFloat3::load(_v.x * inv_len, _v.y * inv_len, _v.z * inv_len);
}

pub fn normalize2(_v: &SoaFloat2) -> SoaFloat2 {
    let len2 = _v.x * _v.x + _v.y * _v.y;
    debug_assert!(len2.ne(f32x4::new(0.0, 0.0, 0.0, 0.0)).all()
        && "_v is not normalizable".parse().unwrap());

    let inv_len = f32x4::new(1.0, 1.0, 1.0, 1.0) / len2.sqrt();
    return SoaFloat2::load(_v.x * inv_len, _v.y * inv_len);
}

//--------------------------------------------------------------------------------------------------
// Test if each vector _v is normalized using estimated tolerance.
pub fn is_normalized_est4(_v: &SoaFloat4) -> m32x4 {
    let len2 = _v.x * _v.x + _v.y * _v.y + _v.z * _v.z + _v.w * _v.w;
    return (len2 - f32x4::new(1.0, 1.0, 1.0, 1.0)).abs().
        le(f32x4::new(K_NORMALIZATION_TOLERANCE_EST_SQ, K_NORMALIZATION_TOLERANCE_EST_SQ,
                      K_NORMALIZATION_TOLERANCE_EST_SQ, K_NORMALIZATION_TOLERANCE_EST_SQ));
}

pub fn is_normalized_est3(_v: &SoaFloat3) -> m32x4 {
    let len2 = _v.x * _v.x + _v.y * _v.y + _v.z * _v.z;
    return (len2 - f32x4::new(1.0, 1.0, 1.0, 1.0)).abs().
        le(f32x4::new(K_NORMALIZATION_TOLERANCE_EST_SQ, K_NORMALIZATION_TOLERANCE_EST_SQ,
                      K_NORMALIZATION_TOLERANCE_EST_SQ, K_NORMALIZATION_TOLERANCE_EST_SQ));
}

pub fn is_normalized_est2(_v: &SoaFloat2) -> m32x4 {
    let len2 = _v.x * _v.x + _v.y * _v.y;
    return (len2 - f32x4::new(1.0, 1.0, 1.0, 1.0)).abs().
        le(f32x4::new(K_NORMALIZATION_TOLERANCE_EST_SQ, K_NORMALIZATION_TOLERANCE_EST_SQ,
                      K_NORMALIZATION_TOLERANCE_EST_SQ, K_NORMALIZATION_TOLERANCE_EST_SQ));
}

//--------------------------------------------------------------------------------------------------
// Returns the normalized vector _v if the norm of _v is not 0.
// Otherwise returns _safer.
pub fn normalize_safe4(_v: &SoaFloat4, _safer: &SoaFloat4) -> SoaFloat4 {
    debug_assert!(is_normalized_est4(_safer).all() && "_safer is not normalized".parse().unwrap());
    let len2 = _v.x * _v.x + _v.y * _v.y + _v.z * _v.z + _v.w * _v.w;
    let b = len2.ne(f32x4::new(0.0, 0.0, 0.0, 0.0));
    let inv_len = f32x4::new(1.0, 1.0, 1.0, 1.0) / len2.sqrt();
    return SoaFloat4::load(
        b.select(_v.x * inv_len, _safer.x), b.select(_v.y * inv_len, _safer.y),
        b.select(_v.z * inv_len, _safer.z), b.select(_v.w * inv_len, _safer.w));
}

pub fn normalize_safe3(_v: &SoaFloat3, _safer: &SoaFloat3) -> SoaFloat3 {
    debug_assert!(is_normalized_est3(_safer).all() && "_safer is not normalized".parse().unwrap());

    let len2 = _v.x * _v.x + _v.y * _v.y + _v.z * _v.z;
    let b = len2.ne(f32x4::new(0.0, 0.0, 0.0, 0.0));
    let inv_len = f32x4::new(1.0, 1.0, 1.0, 1.0) / len2.sqrt();
    return SoaFloat3::load(b.select(_v.x * inv_len, _safer.x),
                           b.select(_v.y * inv_len, _safer.y),
                           b.select(_v.z * inv_len, _safer.z));
}

pub fn normalize_safe2(_v: &SoaFloat2, _safer: &SoaFloat2) -> SoaFloat2 {
    debug_assert!(is_normalized_est2(_safer).all() && "_safer is not normalized".parse().unwrap());
    let len2 = _v.x * _v.x + _v.y * _v.y;
    let b = len2.ne(f32x4::new(0.0, 0.0, 0.0, 0.0));
    let inv_len = f32x4::new(1.0, 1.0, 1.0, 1.0) / len2.sqrt();
    return SoaFloat2::load(b.select(_v.x * inv_len, _safer.x),
                           b.select(_v.y * inv_len, _safer.y));
}

//--------------------------------------------------------------------------------------------------
// Returns the linear interpolation of _a and _b with coefficient _f.
// _f is not limited to range [0,1].
pub fn lerp4(_a: &SoaFloat4, _b: &SoaFloat4, _f: f32x4) -> SoaFloat4 {
    return SoaFloat4::load((_b.x - _a.x) * _f + _a.x, (_b.y - _a.y) * _f + _a.y,
                           (_b.z - _a.z) * _f + _a.z, (_b.w - _a.w) * _f + _a.w);
}

pub fn lerp3(_a: &SoaFloat3, _b: &SoaFloat3, _f: f32x4) -> SoaFloat3 {
    return SoaFloat3::load((_b.x - _a.x) * _f + _a.x, (_b.y - _a.y) * _f + _a.y,
                           (_b.z - _a.z) * _f + _a.z);
}

pub fn lerp2(_a: &SoaFloat2, _b: &SoaFloat2, _f: f32x4) -> SoaFloat2 {
    return SoaFloat2::load((_b.x - _a.x) * _f + _a.x, (_b.y - _a.y) * _f + _a.y);
}

//--------------------------------------------------------------------------------------------------
// Returns the minimum of each element of _a and _b.
pub fn min4(_a: &SoaFloat4, _b: &SoaFloat4) -> SoaFloat4 {
    return SoaFloat4::load(f32x4::min(_a.x, _b.x), f32x4::min(_a.y, _b.y),
                           f32x4::min(_a.z, _b.z),
                           f32x4::min(_a.w, _b.w));
}

pub fn min3(_a: &SoaFloat3, _b: &SoaFloat3) -> SoaFloat3 {
    return SoaFloat3::load(f32x4::min(_a.x, _b.x), f32x4::min(_a.y, _b.y),
                           f32x4::min(_a.z, _b.z));
}

pub fn min2(_a: &SoaFloat2, _b: &SoaFloat2) -> SoaFloat2 {
    return SoaFloat2::load(f32x4::min(_a.x, _b.x), f32x4::min(_a.y, _b.y));
}

//--------------------------------------------------------------------------------------------------
// Returns the maximum of each element of _a and _b.
pub fn max4(_a: &SoaFloat4, _b: &SoaFloat4) -> SoaFloat4 {
    return SoaFloat4::load(f32x4::max(_a.x, _b.x), f32x4::max(_a.y, _b.y),
                           f32x4::max(_a.z, _b.z),
                           f32x4::max(_a.w, _b.w));
}

pub fn max3(_a: &SoaFloat3, _b: &SoaFloat3) -> SoaFloat3 {
    return SoaFloat3::load(f32x4::max(_a.x, _b.x), f32x4::max(_a.y, _b.y),
                           f32x4::max(_a.z, _b.z));
}

pub fn max2(_a: &SoaFloat2, _b: &SoaFloat2) -> SoaFloat2 {
    return SoaFloat2::load(f32x4::max(_a.x, _b.x), f32x4::max(_a.y, _b.y));
}

//--------------------------------------------------------------------------------------------------
// Clamps each element of _x between _a and _b.
// _a must be less or equal to b;
pub fn clamp4(_a: &SoaFloat4, _v: &SoaFloat4, _b: &SoaFloat4) -> SoaFloat4 {
    return max4(_a, &min4(_v, _b));
}

pub fn clamp3(_a: &SoaFloat3, _v: &SoaFloat3, _b: &SoaFloat3) -> SoaFloat3 {
    return max3(_a, &min3(_v, _b));
}

pub fn clamp2(_a: &SoaFloat2, _v: &SoaFloat2, _b: &SoaFloat2) -> SoaFloat2 {
    return max2(_a, &min2(_v, _b));
}