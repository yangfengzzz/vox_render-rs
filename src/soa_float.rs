/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::math_constant::*;
use crate::simd_math::*;
use std::ops::{Add, Sub, Neg, Mul, Div};

#[derive(Clone)]
pub struct SoaFloat2 {
    pub x: SimdFloat4,
    pub y: SimdFloat4,
}

impl SoaFloat2 {
    pub fn load(_x: SimdFloat4, _y: SimdFloat4) -> SoaFloat2 {
        return SoaFloat2 {
            x: _x,
            y: _y,
        };
    }

    #[inline]
    pub fn zero() -> SoaFloat2 {
        return SoaFloat2 {
            x: SimdFloat4::zero(),
            y: SimdFloat4::zero(),
        };
    }

    #[inline]
    pub fn one() -> SoaFloat2 {
        return SoaFloat2 {
            x: SimdFloat4::one(),
            y: SimdFloat4::one(),
        };
    }

    #[inline]
    pub fn x_axis() -> SoaFloat2 {
        return SoaFloat2 {
            x: SimdFloat4::one(),
            y: SimdFloat4::zero(),
        };
    }

    #[inline]
    pub fn y_axis() -> SoaFloat2 {
        return SoaFloat2 {
            x: SimdFloat4::zero(),
            y: SimdFloat4::one(),
        };
    }
}

//--------------------------------------------------------------------------------------------------
#[derive(Clone)]
pub struct SoaFloat3 {
    pub x: SimdFloat4,
    pub y: SimdFloat4,
    pub z: SimdFloat4,
}

impl SoaFloat3 {
    #[inline]
    pub fn load(_x: SimdFloat4, _y: SimdFloat4,
                _z: SimdFloat4) -> SoaFloat3 {
        return SoaFloat3 {
            x: _x,
            y: _y,
            z: _z,
        };
    }

    #[inline]
    pub fn loa2d(_v: &SoaFloat2, _z: SimdFloat4) -> SoaFloat3 {
        return SoaFloat3 {
            x: _v.x,
            y: _v.y,
            z: _z,
        };
    }

    #[inline]
    pub fn zero() -> SoaFloat3 {
        return SoaFloat3 {
            x: SimdFloat4::zero(),
            y: SimdFloat4::zero(),
            z: SimdFloat4::zero(),
        };
    }

    #[inline]
    pub fn one() -> SoaFloat3 {
        return SoaFloat3 {
            x: SimdFloat4::one(),
            y: SimdFloat4::one(),
            z: SimdFloat4::one(),
        };
    }

    #[inline]
    pub fn x_axis() -> SoaFloat3 {
        return SoaFloat3 {
            x: SimdFloat4::one(),
            y: SimdFloat4::zero(),
            z: SimdFloat4::zero(),
        };
    }

    #[inline]
    pub fn y_axis() -> SoaFloat3 {
        return SoaFloat3 {
            x: SimdFloat4::zero(),
            y: SimdFloat4::one(),
            z: SimdFloat4::zero(),
        };
    }

    #[inline]
    pub fn z_axis() -> SoaFloat3 {
        return SoaFloat3 {
            x: SimdFloat4::zero(),
            y: SimdFloat4::zero(),
            z: SimdFloat4::one(),
        };
    }
}

//--------------------------------------------------------------------------------------------------
#[derive(Clone)]
pub struct SoaFloat4 {
    pub x: SimdFloat4,
    pub y: SimdFloat4,
    pub z: SimdFloat4,
    pub w: SimdFloat4,
}

impl SoaFloat4 {
    #[inline]
    pub fn load(_x: SimdFloat4, _y: SimdFloat4,
                _z: SimdFloat4, _w: SimdFloat4) -> SoaFloat4 {
        return SoaFloat4 {
            x: _x,
            y: _y,
            z: _z,
            w: _w,
        };
    }

    #[inline]
    pub fn load3(_v: &SoaFloat3, _w: SimdFloat4) -> SoaFloat4 {
        return SoaFloat4 {
            x: _v.x,
            y: _v.y,
            z: _v.z,
            w: _w,
        };
    }

    pub fn load2(_v: &SoaFloat2, _z: SimdFloat4,
                 _w: SimdFloat4) -> SoaFloat4 {
        return SoaFloat4 {
            x: _v.x,
            y: _v.y,
            z: _z,
            w: _w,
        };
    }

    #[inline]
    pub fn zero() -> SoaFloat4 {
        let zero = SimdFloat4::zero();
        return SoaFloat4 {
            x: zero,
            y: zero,
            z: zero,
            w: zero,
        };
    }

    #[inline]
    pub fn one() -> SoaFloat4 {
        let one = SimdFloat4::one();
        return SoaFloat4 {
            x: one,
            y: one,
            z: one,
            w: one,
        };
    }

    #[inline]
    pub fn x_axis() -> SoaFloat4 {
        let zero = SimdFloat4::zero();
        let one = SimdFloat4::one();
        return SoaFloat4 {
            x: one,
            y: zero,
            z: zero,
            w: zero,
        };
    }

    #[inline]
    pub fn y_axis() -> SoaFloat4 {
        let zero = SimdFloat4::zero();
        let one = SimdFloat4::one();
        return SoaFloat4 {
            x: zero,
            y: one,
            z: zero,
            w: zero,
        };
    }

    #[inline]
    pub fn z_axis() -> SoaFloat4 {
        let zero = SimdFloat4::zero();
        let one = SimdFloat4::one();
        return SoaFloat4 {
            x: zero,
            y: zero,
            z: one,
            w: zero,
        };
    }

    #[inline]
    pub fn w_axis() -> SoaFloat4 {
        let zero = SimdFloat4::zero();
        let one = SimdFloat4::one();
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
impl Mul<SimdFloat4> for SoaFloat4 {
    type Output = SoaFloat4;
    #[inline]
    fn mul(self, rhs: SimdFloat4) -> Self::Output {
        return SoaFloat4::load(self.x * rhs, self.y * rhs, self.z * rhs,
                               self.w * rhs);
    }
}

impl Mul<SimdFloat4> for SoaFloat3 {
    type Output = SoaFloat3;
    #[inline]
    fn mul(self, rhs: SimdFloat4) -> Self::Output {
        return SoaFloat3::load(self.x * rhs, self.y * rhs, self.z * rhs);
    }
}

impl Mul<SimdFloat4> for SoaFloat2 {
    type Output = SoaFloat2;
    #[inline]
    fn mul(self, rhs: SimdFloat4) -> Self::Output {
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
        x: SimdFloat4::madd(&_a.x, _b.x, _addend.x),
        y: SimdFloat4::madd(&_a.y, _b.y, _addend.y),
    };
}

#[inline]
pub fn m_add3(_a: &SoaFloat3,
              _b: &SoaFloat3,
              _addend: &SoaFloat3) -> SoaFloat3 {
    return SoaFloat3 {
        x: SimdFloat4::madd(&_a.x, _b.x, _addend.x),
        y: SimdFloat4::madd(&_a.y, _b.y, _addend.y),
        z: SimdFloat4::madd(&_a.z, _b.z, _addend.z),
    };
}

#[inline]
pub fn m_add4(_a: &SoaFloat4,
              _b: &SoaFloat4,
              _addend: &SoaFloat4) -> SoaFloat4 {
    return SoaFloat4 {
        x: SimdFloat4::madd(&_a.x, _b.x, _addend.x),
        y: SimdFloat4::madd(&_a.y, _b.y, _addend.y),
        z: SimdFloat4::madd(&_a.z, _b.z, _addend.z),
        w: SimdFloat4::madd(&_a.w, _b.w, _addend.w),
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
impl Div<SimdFloat4> for SoaFloat4 {
    type Output = SoaFloat4;
    #[inline]
    fn div(self, rhs: SimdFloat4) -> Self::Output {
        return SoaFloat4::load(self.x / rhs, self.y / rhs, self.z / rhs,
                               self.w / rhs);
    }
}

impl Div<SimdFloat4> for SoaFloat3 {
    type Output = SoaFloat3;
    #[inline]
    fn div(self, rhs: SimdFloat4) -> Self::Output {
        return SoaFloat3::load(self.x / rhs, self.y / rhs, self.z / rhs);
    }
}

impl Div<SimdFloat4> for SoaFloat2 {
    type Output = SoaFloat2;
    #[inline]
    fn div(self, rhs: SimdFloat4) -> Self::Output {
        return SoaFloat2::load(self.x / rhs, self.y / rhs);
    }
}

//--------------------------------------------------------------------------------------------------
// Returns true if each element of a is less than each element of _b.
impl SoaFloat4 {
    #[inline]
    pub fn lt(&self, other: &Self) -> SimdInt4 {
        let x = SimdFloat4::cmp_lt(&self.x, other.x);
        let y = SimdFloat4::cmp_lt(&self.y, other.y);
        let z = SimdFloat4::cmp_lt(&self.z, other.z);
        let w = SimdFloat4::cmp_lt(&self.w, other.w);
        return SimdInt4::and(&SimdInt4::and(&SimdInt4::and(&x, y), z), w);
    }

    #[inline]
    pub fn le(&self, other: &Self) -> SimdInt4 {
        let x = SimdFloat4::cmp_le(&self.x, other.x);
        let y = SimdFloat4::cmp_le(&self.y, other.y);
        let z = SimdFloat4::cmp_le(&self.z, other.z);
        let w = SimdFloat4::cmp_le(&self.w, other.w);
        return SimdInt4::and(&SimdInt4::and(&SimdInt4::and(&x, y), z), w);
    }

    #[inline]
    pub fn gt(&self, other: &Self) -> SimdInt4 {
        let x = SimdFloat4::cmp_gt(&self.x, other.x);
        let y = SimdFloat4::cmp_gt(&self.y, other.y);
        let z = SimdFloat4::cmp_gt(&self.z, other.z);
        let w = SimdFloat4::cmp_gt(&self.w, other.w);
        return SimdInt4::and(&SimdInt4::and(&SimdInt4::and(&x, y), z), w);
    }

    #[inline]
    pub fn ge(&self, other: &Self) -> SimdInt4 {
        let x = SimdFloat4::cmp_ge(&self.x, other.x);
        let y = SimdFloat4::cmp_ge(&self.y, other.y);
        let z = SimdFloat4::cmp_ge(&self.z, other.z);
        let w = SimdFloat4::cmp_ge(&self.w, other.w);
        return SimdInt4::and(&SimdInt4::and(&SimdInt4::and(&x, y), z), w);
    }

    #[inline]
    pub fn eq(&self, other: &Self) -> SimdInt4 {
        let x = SimdFloat4::cmp_eq(&self.x, other.x);
        let y = SimdFloat4::cmp_eq(&self.y, other.y);
        let z = SimdFloat4::cmp_eq(&self.z, other.z);
        let w = SimdFloat4::cmp_eq(&self.w, other.w);
        return SimdInt4::and(&SimdInt4::and(&SimdInt4::and(&x, y), z), w);
    }

    #[inline]
    pub fn ne(&self, other: &Self) -> SimdInt4 {
        let x = SimdFloat4::cmp_ne(&self.x, other.x);
        let y = SimdFloat4::cmp_ne(&self.y, other.y);
        let z = SimdFloat4::cmp_ne(&self.z, other.z);
        let w = SimdFloat4::cmp_ne(&self.w, other.w);
        return SimdInt4::and(&SimdInt4::and(&SimdInt4::and(&x, y), z), w);
    }
}

impl SoaFloat3 {
    #[inline]
    pub fn lt(&self, other: &Self) -> SimdInt4 {
        let x = SimdFloat4::cmp_lt(&self.x, other.x);
        let y = SimdFloat4::cmp_lt(&self.y, other.y);
        let z = SimdFloat4::cmp_lt(&self.z, other.z);
        return SimdInt4::and(&SimdInt4::and(&x, y), z);
    }

    #[inline]
    pub fn le(&self, other: &Self) -> SimdInt4 {
        let x = SimdFloat4::cmp_le(&self.x, other.x);
        let y = SimdFloat4::cmp_le(&self.y, other.y);
        let z = SimdFloat4::cmp_le(&self.z, other.z);
        return SimdInt4::and(&SimdInt4::and(&x, y), z);
    }

    #[inline]
    pub fn gt(&self, other: &Self) -> SimdInt4 {
        let x = SimdFloat4::cmp_gt(&self.x, other.x);
        let y = SimdFloat4::cmp_gt(&self.y, other.y);
        let z = SimdFloat4::cmp_gt(&self.z, other.z);
        return SimdInt4::and(&SimdInt4::and(&x, y), z);
    }

    #[inline]
    pub fn ge(&self, other: &Self) -> SimdInt4 {
        let x = SimdFloat4::cmp_ge(&self.x, other.x);
        let y = SimdFloat4::cmp_ge(&self.y, other.y);
        let z = SimdFloat4::cmp_ge(&self.z, other.z);
        return SimdInt4::and(&SimdInt4::and(&x, y), z);
    }

    #[inline]
    pub fn eq(&self, other: &Self) -> SimdInt4 {
        let x = SimdFloat4::cmp_eq(&self.x, other.x);
        let y = SimdFloat4::cmp_eq(&self.y, other.y);
        let z = SimdFloat4::cmp_eq(&self.z, other.z);
        return SimdInt4::and(&SimdInt4::and(&x, y), z);
    }

    #[inline]
    pub fn ne(&self, other: &Self) -> SimdInt4 {
        let x = SimdFloat4::cmp_ne(&self.x, other.x);
        let y = SimdFloat4::cmp_ne(&self.y, other.y);
        let z = SimdFloat4::cmp_ne(&self.z, other.z);
        return SimdInt4::and(&SimdInt4::and(&x, y), z);
    }
}

impl SoaFloat2 {
    #[inline]
    pub fn lt(&self, other: &Self) -> SimdInt4 {
        let x = SimdFloat4::cmp_lt(&self.x, other.x);
        let y = SimdFloat4::cmp_lt(&self.y, other.y);
        return SimdInt4::and(&x, y);
    }

    #[inline]
    pub fn le(&self, other: &Self) -> SimdInt4 {
        let x = SimdFloat4::cmp_le(&self.x, other.x);
        let y = SimdFloat4::cmp_le(&self.y, other.y);
        return SimdInt4::and(&x, y);
    }

    #[inline]
    pub fn gt(&self, other: &Self) -> SimdInt4 {
        let x = SimdFloat4::cmp_gt(&self.x, other.x);
        let y = SimdFloat4::cmp_gt(&self.y, other.y);
        return SimdInt4::and(&x, y);
    }

    #[inline]
    pub fn ge(&self, other: &Self) -> SimdInt4 {
        let x = SimdFloat4::cmp_ge(&self.x, other.x);
        let y = SimdFloat4::cmp_ge(&self.y, other.y);
        return SimdInt4::and(&x, y);
    }

    #[inline]
    pub fn eq(&self, other: &Self) -> SimdInt4 {
        let x = SimdFloat4::cmp_eq(&self.x, other.x);
        let y = SimdFloat4::cmp_eq(&self.y, other.y);
        return SimdInt4::and(&x, y);
    }

    #[inline]
    pub fn ne(&self, other: &Self) -> SimdInt4 {
        let x = SimdFloat4::cmp_ne(&self.x, other.x);
        let y = SimdFloat4::cmp_ne(&self.y, other.y);
        return SimdInt4::and(&x, y);
    }
}

//--------------------------------------------------------------------------------------------------
// Returns the (horizontal) addition of each element of _v.
pub fn h_add4(_v: &SoaFloat4) -> SimdFloat4 { return _v.x + _v.y + _v.z + _v.w; }

pub fn h_add3(_v: &SoaFloat3) -> SimdFloat4 { return _v.x + _v.y + _v.z; }

pub fn h_add2(_v: &SoaFloat2) -> SimdFloat4 { return _v.x + _v.y; }

//--------------------------------------------------------------------------------------------------
// Returns the dot product of _a and _b.
pub fn dot4(_a: &SoaFloat4, _b: &SoaFloat4) -> SimdFloat4 {
    return _a.x * _b.x + _a.y * _b.y + _a.z * _b.z + _a.w * _b.w;
}

pub fn dot3(_a: &SoaFloat3, _b: &SoaFloat3) -> SimdFloat4 {
    return _a.x * _b.x + _a.y * _b.y + _a.z * _b.z;
}

pub fn dot2(_a: &SoaFloat2, _b: &SoaFloat2) -> SimdFloat4 {
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
pub fn length4(_v: &SoaFloat4) -> SimdFloat4 {
    let len2 = _v.x * _v.x + _v.y * _v.y + _v.z * _v.z + _v.w * _v.w;
    return len2.sqrt();
}

pub fn length3(_v: &SoaFloat3) -> SimdFloat4 {
    let len2 = _v.x * _v.x + _v.y * _v.y + _v.z * _v.z;
    return len2.sqrt();
}

pub fn length2(_v: &SoaFloat2) -> SimdFloat4 {
    let len2 = _v.x * _v.x + _v.y * _v.y;
    return len2.sqrt();
}

//--------------------------------------------------------------------------------------------------
// Returns the square length |_v|^2 of _v.
pub fn length_sqr4(_v: &SoaFloat4) -> SimdFloat4 {
    return _v.x * _v.x + _v.y * _v.y + _v.z * _v.z + _v.w * _v.w;
}

pub fn length_sqr3(_v: &SoaFloat3) -> SimdFloat4 {
    return _v.x * _v.x + _v.y * _v.y + _v.z * _v.z;
}

pub fn length_sqr2(_v: &SoaFloat2) -> SimdFloat4 {
    return _v.x * _v.x + _v.y * _v.y;
}

//--------------------------------------------------------------------------------------------------
// Returns the normalized vector _v.
pub fn normalize4(_v: &SoaFloat4) -> SoaFloat4 {
    let len2 = _v.x * _v.x + _v.y * _v.y + _v.z * _v.z + _v.w * _v.w;
    debug_assert!(len2.ne(SimdFloat4::load(0.0, 0.0, 0.0, 0.0)).all()
        && "_v is not normalizable".parse().unwrap());

    let inv_len = SimdFloat4::load(1.0, 1.0, 1.0, 1.0) / len2.sqrt();
    return SoaFloat4::load(
        _v.x * inv_len, _v.y * inv_len, _v.z * inv_len,
        _v.w * inv_len,
    );
}

pub fn normalize3(_v: &SoaFloat3) -> SoaFloat3 {
    let len2 = _v.x * _v.x + _v.y * _v.y + _v.z * _v.z;
    debug_assert!(len2.ne(SimdFloat4::load(0.0, 0.0, 0.0, 0.0)).all()
        && "_v is not normalizable".parse().unwrap());

    let inv_len = SimdFloat4::load(1.0, 1.0, 1.0, 1.0) / len2.sqrt();
    return SoaFloat3::load(_v.x * inv_len, _v.y * inv_len, _v.z * inv_len);
}

pub fn normalize2(_v: &SoaFloat2) -> SoaFloat2 {
    let len2 = _v.x * _v.x + _v.y * _v.y;
    debug_assert!(len2.ne(SimdFloat4::load(0.0, 0.0, 0.0, 0.0)).all()
        && "_v is not normalizable".parse().unwrap());

    let inv_len = SimdFloat4::load(1.0, 1.0, 1.0, 1.0) / len2.sqrt();
    return SoaFloat2::load(_v.x * inv_len, _v.y * inv_len);
}

//--------------------------------------------------------------------------------------------------
// Test if each vector _v is normalized.
pub fn is_normalized4(_v: &SoaFloat4) -> SimdInt4 {
    let len2 = _v.x * _v.x + _v.y * _v.y + _v.z * _v.z + _v.w * _v.w;
    return (len2 - SimdFloat4::load(1.0, 1.0, 1.0, 1.0)).abs().
        lt(SimdFloat4::load(K_NORMALIZATION_TOLERANCE_SQ, K_NORMALIZATION_TOLERANCE_SQ,
                            K_NORMALIZATION_TOLERANCE_SQ, K_NORMALIZATION_TOLERANCE_SQ));
}

pub fn is_normalized3(_v: &SoaFloat3) -> SimdInt4 {
    let len2 = _v.x * _v.x + _v.y * _v.y + _v.z * _v.z;
    return (len2 - SimdFloat4::load(1.0, 1.0, 1.0, 1.0)).abs().
        lt(SimdFloat4::load(K_NORMALIZATION_TOLERANCE_SQ, K_NORMALIZATION_TOLERANCE_SQ,
                            K_NORMALIZATION_TOLERANCE_SQ, K_NORMALIZATION_TOLERANCE_SQ));
}

pub fn is_normalized2(_v: &SoaFloat2) -> SimdInt4 {
    let len2 = _v.x * _v.x + _v.y * _v.y;
    return (len2 - SimdFloat4::load(1.0, 1.0, 1.0, 1.0)).abs().
        lt(SimdFloat4::load(K_NORMALIZATION_TOLERANCE_SQ, K_NORMALIZATION_TOLERANCE_SQ,
                            K_NORMALIZATION_TOLERANCE_SQ, K_NORMALIZATION_TOLERANCE_SQ));
}


//--------------------------------------------------------------------------------------------------
// Test if each vector _v is normalized using estimated tolerance.
pub fn is_normalized_est4(_v: &SoaFloat4) -> SimdInt4 {
    let len2 = _v.x * _v.x + _v.y * _v.y + _v.z * _v.z + _v.w * _v.w;
    return (len2 - SimdFloat4::load(1.0, 1.0, 1.0, 1.0)).abs().
        lt(SimdFloat4::load(K_NORMALIZATION_TOLERANCE_EST_SQ, K_NORMALIZATION_TOLERANCE_EST_SQ,
                            K_NORMALIZATION_TOLERANCE_EST_SQ, K_NORMALIZATION_TOLERANCE_EST_SQ));
}

pub fn is_normalized_est3(_v: &SoaFloat3) -> SimdInt4 {
    let len2 = _v.x * _v.x + _v.y * _v.y + _v.z * _v.z;
    return (len2 - SimdFloat4::load(1.0, 1.0, 1.0, 1.0)).abs().
        lt(SimdFloat4::load(K_NORMALIZATION_TOLERANCE_EST_SQ, K_NORMALIZATION_TOLERANCE_EST_SQ,
                            K_NORMALIZATION_TOLERANCE_EST_SQ, K_NORMALIZATION_TOLERANCE_EST_SQ));
}

pub fn is_normalized_est2(_v: &SoaFloat2) -> SimdInt4 {
    let len2 = _v.x * _v.x + _v.y * _v.y;
    return (len2 - SimdFloat4::load(1.0, 1.0, 1.0, 1.0)).abs().
        lt(SimdFloat4::load(K_NORMALIZATION_TOLERANCE_EST_SQ, K_NORMALIZATION_TOLERANCE_EST_SQ,
                            K_NORMALIZATION_TOLERANCE_EST_SQ, K_NORMALIZATION_TOLERANCE_EST_SQ));
}

//--------------------------------------------------------------------------------------------------
// Returns the normalized vector _v if the norm of _v is not 0.
// Otherwise returns _safer.
pub fn normalize_safe4(_v: &SoaFloat4, _safer: &SoaFloat4) -> SoaFloat4 {
    debug_assert!(is_normalized_est4(_safer).all() && "_safer is not normalized".parse().unwrap());
    let len2 = _v.x * _v.x + _v.y * _v.y + _v.z * _v.z + _v.w * _v.w;
    let b = len2.ne(SimdFloat4::load(0.0, 0.0, 0.0, 0.0));
    let inv_len = SimdFloat4::load(1.0, 1.0, 1.0, 1.0) / len2.sqrt();
    return SoaFloat4::load(
        b.select(_v.x * inv_len, _safer.x), b.select(_v.y * inv_len, _safer.y),
        b.select(_v.z * inv_len, _safer.z), b.select(_v.w * inv_len, _safer.w));
}

pub fn normalize_safe3(_v: &SoaFloat3, _safer: &SoaFloat3) -> SoaFloat3 {
    debug_assert!(is_normalized_est3(_safer).all() && "_safer is not normalized".parse().unwrap());

    let len2 = _v.x * _v.x + _v.y * _v.y + _v.z * _v.z;
    let b = len2.ne(SimdFloat4::load(0.0, 0.0, 0.0, 0.0));
    let inv_len = SimdFloat4::load(1.0, 1.0, 1.0, 1.0) / len2.sqrt();
    return SoaFloat3::load(b.select(_v.x * inv_len, _safer.x),
                           b.select(_v.y * inv_len, _safer.y),
                           b.select(_v.z * inv_len, _safer.z));
}

pub fn normalize_safe2(_v: &SoaFloat2, _safer: &SoaFloat2) -> SoaFloat2 {
    debug_assert!(is_normalized_est2(_safer).all() && "_safer is not normalized".parse().unwrap());
    let len2 = _v.x * _v.x + _v.y * _v.y;
    let b = len2.ne(SimdFloat4::load(0.0, 0.0, 0.0, 0.0));
    let inv_len = SimdFloat4::load(1.0, 1.0, 1.0, 1.0) / len2.sqrt();
    return SoaFloat2::load(b.select(_v.x * inv_len, _safer.x),
                           b.select(_v.y * inv_len, _safer.y));
}

//--------------------------------------------------------------------------------------------------
// Returns the linear interpolation of _a and _b with coefficient _f.
// _f is not limited to range [0,1].
pub fn lerp4(_a: &SoaFloat4, _b: &SoaFloat4, _f: SimdFloat4) -> SoaFloat4 {
    return SoaFloat4::load((_b.x - _a.x) * _f + _a.x, (_b.y - _a.y) * _f + _a.y,
                           (_b.z - _a.z) * _f + _a.z, (_b.w - _a.w) * _f + _a.w);
}

pub fn lerp3(_a: &SoaFloat3, _b: &SoaFloat3, _f: SimdFloat4) -> SoaFloat3 {
    return SoaFloat3::load((_b.x - _a.x) * _f + _a.x, (_b.y - _a.y) * _f + _a.y,
                           (_b.z - _a.z) * _f + _a.z);
}

pub fn lerp2(_a: &SoaFloat2, _b: &SoaFloat2, _f: SimdFloat4) -> SoaFloat2 {
    return SoaFloat2::load((_b.x - _a.x) * _f + _a.x, (_b.y - _a.y) * _f + _a.y);
}

//--------------------------------------------------------------------------------------------------
// Returns the minimum of each element of _a and _b.
pub fn min4(_a: &SoaFloat4, _b: &SoaFloat4) -> SoaFloat4 {
    return SoaFloat4::load(SimdFloat4::min(&_a.x, _b.x), SimdFloat4::min(&_a.y, _b.y),
                           SimdFloat4::min(&_a.z, _b.z),
                           SimdFloat4::min(&_a.w, _b.w));
}

pub fn min3(_a: &SoaFloat3, _b: &SoaFloat3) -> SoaFloat3 {
    return SoaFloat3::load(SimdFloat4::min(&_a.x, _b.x), SimdFloat4::min(&_a.y, _b.y),
                           SimdFloat4::min(&_a.z, _b.z));
}

pub fn min2(_a: &SoaFloat2, _b: &SoaFloat2) -> SoaFloat2 {
    return SoaFloat2::load(SimdFloat4::min(&_a.x, _b.x), SimdFloat4::min(&_a.y, _b.y));
}

//--------------------------------------------------------------------------------------------------
// Returns the maximum of each element of _a and _b.
pub fn max4(_a: &SoaFloat4, _b: &SoaFloat4) -> SoaFloat4 {
    return SoaFloat4::load(SimdFloat4::max(&_a.x, _b.x), SimdFloat4::max(&_a.y, _b.y),
                           SimdFloat4::max(&_a.z, _b.z),
                           SimdFloat4::max(&_a.w, _b.w));
}

pub fn max3(_a: &SoaFloat3, _b: &SoaFloat3) -> SoaFloat3 {
    return SoaFloat3::load(SimdFloat4::max(&_a.x, _b.x), SimdFloat4::max(&_a.y, _b.y),
                           SimdFloat4::max(&_a.z, _b.z));
}

pub fn max2(_a: &SoaFloat2, _b: &SoaFloat2) -> SoaFloat2 {
    return SoaFloat2::load(SimdFloat4::max(&_a.x, _b.x), SimdFloat4::max(&_a.y, _b.y));
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