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

#[derive(Clone, Copy)]
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
#[derive(Clone, Copy)]
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
    pub fn load2(_v: &SoaFloat2, _z: SimdFloat4) -> SoaFloat3 {
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
#[derive(Clone, Copy)]
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
impl SoaFloat2 {
    // Multiplies _a and _b, then adds _addend.
// v = (_a * _b) + _addend
    #[inline]
    pub fn m_add(&self,
                 _b: &SoaFloat2,
                 _addend: &SoaFloat2) -> SoaFloat2 {
        return SoaFloat2 {
            x: SimdFloat4::madd(&self.x, _b.x, _addend.x),
            y: SimdFloat4::madd(&self.y, _b.y, _addend.y),
        };
    }
}

impl SoaFloat3 {
    #[inline]
    pub fn m_add(&self,
                 _b: &SoaFloat3,
                 _addend: &SoaFloat3) -> SoaFloat3 {
        return SoaFloat3 {
            x: SimdFloat4::madd(&self.x, _b.x, _addend.x),
            y: SimdFloat4::madd(&self.y, _b.y, _addend.y),
            z: SimdFloat4::madd(&self.z, _b.z, _addend.z),
        };
    }
}

impl SoaFloat4 {
    #[inline]
    pub fn m_add(&self,
                 _b: &SoaFloat4,
                 _addend: &SoaFloat4) -> SoaFloat4 {
        return SoaFloat4 {
            x: SimdFloat4::madd(&self.x, _b.x, _addend.x),
            y: SimdFloat4::madd(&self.y, _b.y, _addend.y),
            z: SimdFloat4::madd(&self.z, _b.z, _addend.z),
            w: SimdFloat4::madd(&self.w, _b.w, _addend.w),
        };
    }
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
        return SimdInt4::or(&SimdInt4::or(&SimdInt4::or(&x, y), z), w);
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
        return SimdInt4::or(&SimdInt4::or(&x, y), z);
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
        return SimdInt4::or(&x, y);
    }
}

//--------------------------------------------------------------------------------------------------
impl SoaFloat4 {
    // Returns the (horizontal) addition of each element of _v.
    pub fn h_add(&self) -> SimdFloat4 { return self.x + self.y + self.z + self.w; }
}

impl SoaFloat3 {
    pub fn h_add(&self) -> SimdFloat4 { return self.x + self.y + self.z; }
}

impl SoaFloat2 {
    pub fn h_add(&self) -> SimdFloat4 { return self.x + self.y; }
}

//--------------------------------------------------------------------------------------------------
impl SoaFloat4 {
    // Returns the dot product of _a and _b.
    pub fn dot(&self, _b: &SoaFloat4) -> SimdFloat4 {
        return self.x * _b.x + self.y * _b.y + self.z * _b.z + self.w * _b.w;
    }
}

impl SoaFloat3 {
    pub fn dot(&self, _b: &SoaFloat3) -> SimdFloat4 {
        return self.x * _b.x + self.y * _b.y + self.z * _b.z;
    }
}

impl SoaFloat2 {
    pub fn dot(&self, _b: &SoaFloat2) -> SimdFloat4 {
        return self.x * _b.x + self.y * _b.y;
    }
}

//--------------------------------------------------------------------------------------------------
impl SoaFloat3 {
    // Returns the cross product of _a and _b.
    pub fn cross(&self, _b: &SoaFloat3) -> SoaFloat3 {
        return SoaFloat3::load(self.y * _b.z - _b.y * self.z,
                               self.z * _b.x - _b.z * self.x,
                               self.x * _b.y - _b.x * self.y);
    }
}

//--------------------------------------------------------------------------------------------------
impl SoaFloat4 {
    // Returns the length |_v| of _v.
    pub fn length(&self) -> SimdFloat4 {
        let len2 = self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w;
        return len2.sqrt();
    }
}

impl SoaFloat3 {
    pub fn length(&self) -> SimdFloat4 {
        let len2 = self.x * self.x + self.y * self.y + self.z * self.z;
        return len2.sqrt();
    }
}

impl SoaFloat2 {
    pub fn length(&self) -> SimdFloat4 {
        let len2 = self.x * self.x + self.y * self.y;
        return len2.sqrt();
    }
}

//--------------------------------------------------------------------------------------------------
impl SoaFloat4 {
    // Returns the square length |_v|^2 of _v.
    pub fn length_sqr(&self) -> SimdFloat4 {
        return self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w;
    }
}

impl SoaFloat3 {
    pub fn length_sqr(&self) -> SimdFloat4 {
        return self.x * self.x + self.y * self.y + self.z * self.z;
    }
}

impl SoaFloat2 {
    pub fn length_sqr(&self) -> SimdFloat4 {
        return self.x * self.x + self.y * self.y;
    }
}

//--------------------------------------------------------------------------------------------------
impl SoaFloat4 {
    // Returns the normalized vector self.
    pub fn normalize(&self) -> SoaFloat4 {
        let len2 = self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w;
        debug_assert!(len2.cmp_ne(SimdFloat4::load(0.0, 0.0, 0.0, 0.0)).are_all_true()
            && "self is not normalizable".parse().unwrap_or(true));

        let inv_len = SimdFloat4::load(1.0, 1.0, 1.0, 1.0) / len2.sqrt();
        return SoaFloat4::load(
            self.x * inv_len, self.y * inv_len, self.z * inv_len,
            self.w * inv_len,
        );
    }
}

impl SoaFloat3 {
    pub fn normalize(&self) -> SoaFloat3 {
        let len2 = self.x * self.x + self.y * self.y + self.z * self.z;
        debug_assert!(len2.cmp_ne(SimdFloat4::load(0.0, 0.0, 0.0, 0.0)).are_all_true()
            && "self is not normalizable".parse().unwrap_or(true));

        let inv_len = SimdFloat4::load(1.0, 1.0, 1.0, 1.0) / len2.sqrt();
        return SoaFloat3::load(self.x * inv_len, self.y * inv_len, self.z * inv_len);
    }
}

impl SoaFloat2 {
    pub fn normalize(&self) -> SoaFloat2 {
        let len2 = self.x * self.x + self.y * self.y;
        debug_assert!(len2.cmp_ne(SimdFloat4::load(0.0, 0.0, 0.0, 0.0)).are_all_true()
            && "self is not normalizable".parse().unwrap_or(true));

        let inv_len = SimdFloat4::load(1.0, 1.0, 1.0, 1.0) / len2.sqrt();
        return SoaFloat2::load(self.x * inv_len, self.y * inv_len);
    }
}

//--------------------------------------------------------------------------------------------------
impl SoaFloat4 {
    // Test if each vector self is normalized.
    pub fn is_normalized(&self) -> SimdInt4 {
        let len2 = self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w;
        return (len2 - SimdFloat4::load(1.0, 1.0, 1.0, 1.0)).abs().
            cmp_lt(SimdFloat4::load(K_NORMALIZATION_TOLERANCE_SQ, K_NORMALIZATION_TOLERANCE_SQ,
                                    K_NORMALIZATION_TOLERANCE_SQ, K_NORMALIZATION_TOLERANCE_SQ));
    }
}

impl SoaFloat3 {
    pub fn is_normalized(&self) -> SimdInt4 {
        let len2 = self.x * self.x + self.y * self.y + self.z * self.z;
        return (len2 - SimdFloat4::load(1.0, 1.0, 1.0, 1.0)).abs().
            cmp_lt(SimdFloat4::load(K_NORMALIZATION_TOLERANCE_SQ, K_NORMALIZATION_TOLERANCE_SQ,
                                    K_NORMALIZATION_TOLERANCE_SQ, K_NORMALIZATION_TOLERANCE_SQ));
    }
}

impl SoaFloat2 {
    pub fn is_normalized(&self) -> SimdInt4 {
        let len2 = self.x * self.x + self.y * self.y;
        return (len2 - SimdFloat4::load(1.0, 1.0, 1.0, 1.0)).abs().
            cmp_lt(SimdFloat4::load(K_NORMALIZATION_TOLERANCE_SQ, K_NORMALIZATION_TOLERANCE_SQ,
                                    K_NORMALIZATION_TOLERANCE_SQ, K_NORMALIZATION_TOLERANCE_SQ));
    }
}

//--------------------------------------------------------------------------------------------------
impl SoaFloat4 {
    // Test if each vector _v is normalized using estimated tolerance.
    pub fn is_normalized_est(&self) -> SimdInt4 {
        let len2 = self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w;
        return (len2 - SimdFloat4::load(1.0, 1.0, 1.0, 1.0)).abs().
            cmp_lt(SimdFloat4::load(K_NORMALIZATION_TOLERANCE_EST_SQ, K_NORMALIZATION_TOLERANCE_EST_SQ,
                                    K_NORMALIZATION_TOLERANCE_EST_SQ, K_NORMALIZATION_TOLERANCE_EST_SQ));
    }
}

impl SoaFloat3 {
    pub fn is_normalized_est(&self) -> SimdInt4 {
        let len2 = self.x * self.x + self.y * self.y + self.z * self.z;
        return (len2 - SimdFloat4::load(1.0, 1.0, 1.0, 1.0)).abs().
            cmp_lt(SimdFloat4::load(K_NORMALIZATION_TOLERANCE_EST_SQ, K_NORMALIZATION_TOLERANCE_EST_SQ,
                                    K_NORMALIZATION_TOLERANCE_EST_SQ, K_NORMALIZATION_TOLERANCE_EST_SQ));
    }
}

impl SoaFloat2 {
    pub fn is_normalized_est(&self) -> SimdInt4 {
        let len2 = self.x * self.x + self.y * self.y;
        return (len2 - SimdFloat4::load(1.0, 1.0, 1.0, 1.0)).abs().
            cmp_lt(SimdFloat4::load(K_NORMALIZATION_TOLERANCE_EST_SQ, K_NORMALIZATION_TOLERANCE_EST_SQ,
                                    K_NORMALIZATION_TOLERANCE_EST_SQ, K_NORMALIZATION_TOLERANCE_EST_SQ));
    }
}

//--------------------------------------------------------------------------------------------------
impl SoaFloat4 {
    // Returns the normalized vector self if the norm of self is not 0.
// Otherwise returns _safer.
    pub fn normalize_safe(&self, _safer: &SoaFloat4) -> SoaFloat4 {
        debug_assert!(_safer.is_normalized_est().are_all_true() && "_safer is not normalized".parse().unwrap_or(true));
        let len2 = self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w;
        let b = len2.cmp_ne(SimdFloat4::load(0.0, 0.0, 0.0, 0.0));
        let inv_len = SimdFloat4::load(1.0, 1.0, 1.0, 1.0) / len2.sqrt();
        return SoaFloat4::load(
            SimdFloat4::select(b, self.x * inv_len, _safer.x),
            SimdFloat4::select(b, self.y * inv_len, _safer.y),
            SimdFloat4::select(b, self.z * inv_len, _safer.z),
            SimdFloat4::select(b, self.w * inv_len, _safer.w));
    }
}

impl SoaFloat3 {
    pub fn normalize_safe(&self, _safer: &SoaFloat3) -> SoaFloat3 {
        debug_assert!(_safer.is_normalized_est().are_all_true() && "_safer is not normalized".parse().unwrap_or(true));

        let len2 = self.x * self.x + self.y * self.y + self.z * self.z;
        let b = len2.cmp_ne(SimdFloat4::load(0.0, 0.0, 0.0, 0.0));
        let inv_len = SimdFloat4::load(1.0, 1.0, 1.0, 1.0) / len2.sqrt();
        return SoaFloat3::load(SimdFloat4::select(b, self.x * inv_len, _safer.x),
                               SimdFloat4::select(b, self.y * inv_len, _safer.y),
                               SimdFloat4::select(b, self.z * inv_len, _safer.z));
    }
}

impl SoaFloat2 {
    pub fn normalize_safe(&self, _safer: &SoaFloat2) -> SoaFloat2 {
        debug_assert!(_safer.is_normalized_est().are_all_true() && "_safer is not normalized".parse().unwrap_or(true));
        let len2 = self.x * self.x + self.y * self.y;
        let b = len2.cmp_ne(SimdFloat4::load(0.0, 0.0, 0.0, 0.0));
        let inv_len = SimdFloat4::load(1.0, 1.0, 1.0, 1.0) / len2.sqrt();
        return SoaFloat2::load(SimdFloat4::select(b, self.x * inv_len, _safer.x),
                               SimdFloat4::select(b, self.y * inv_len, _safer.y));
    }
}

//--------------------------------------------------------------------------------------------------
impl SoaFloat4 {
    // Returns the linear interpolation of _a and _b with coefficient _f.
    // _f is not limited to range [0,1].
    pub fn lerp(&self, _b: &SoaFloat4, _f: SimdFloat4) -> SoaFloat4 {
        return SoaFloat4::load((_b.x - self.x) * _f + self.x, (_b.y - self.y) * _f + self.y,
                               (_b.z - self.z) * _f + self.z, (_b.w - self.w) * _f + self.w);
    }
}

impl SoaFloat3 {
    pub fn lerp(&self, _b: &SoaFloat3, _f: SimdFloat4) -> SoaFloat3 {
        return SoaFloat3::load((_b.x - self.x) * _f + self.x, (_b.y - self.y) * _f + self.y,
                               (_b.z - self.z) * _f + self.z);
    }
}

impl SoaFloat2 {
    pub fn lerp(&self, _b: &SoaFloat2, _f: SimdFloat4) -> SoaFloat2 {
        return SoaFloat2::load((_b.x - self.x) * _f + self.x, (_b.y - self.y) * _f + self.y);
    }
}

//--------------------------------------------------------------------------------------------------
impl SoaFloat4 {
    // Returns the minimum of each element of _a and _b.
    pub fn min(&self, _b: &SoaFloat4) -> SoaFloat4 {
        return SoaFloat4::load(SimdFloat4::min(&self.x, _b.x), SimdFloat4::min(&self.y, _b.y),
                               SimdFloat4::min(&self.z, _b.z),
                               SimdFloat4::min(&self.w, _b.w));
    }
}

impl SoaFloat3 {
    pub fn min(&self, _b: &SoaFloat3) -> SoaFloat3 {
        return SoaFloat3::load(SimdFloat4::min(&self.x, _b.x), SimdFloat4::min(&self.y, _b.y),
                               SimdFloat4::min(&self.z, _b.z));
    }
}

impl SoaFloat2 {
    pub fn min(&self, _b: &SoaFloat2) -> SoaFloat2 {
        return SoaFloat2::load(SimdFloat4::min(&self.x, _b.x), SimdFloat4::min(&self.y, _b.y));
    }
}

//--------------------------------------------------------------------------------------------------
impl SoaFloat4 {
    // Returns the maximum of each element of _a and _b.
    pub fn max(&self, _b: &SoaFloat4) -> SoaFloat4 {
        return SoaFloat4::load(SimdFloat4::max(&self.x, _b.x), SimdFloat4::max(&self.y, _b.y),
                               SimdFloat4::max(&self.z, _b.z),
                               SimdFloat4::max(&self.w, _b.w));
    }
}

impl SoaFloat3 {
    pub fn max(&self, _b: &SoaFloat3) -> SoaFloat3 {
        return SoaFloat3::load(SimdFloat4::max(&self.x, _b.x), SimdFloat4::max(&self.y, _b.y),
                               SimdFloat4::max(&self.z, _b.z));
    }
}

impl SoaFloat2 {
    pub fn max(&self, _b: &SoaFloat2) -> SoaFloat2 {
        return SoaFloat2::load(SimdFloat4::max(&self.x, _b.x), SimdFloat4::max(&self.y, _b.y));
    }
}

//--------------------------------------------------------------------------------------------------
impl SoaFloat4 {
    // Clamps each element of _x between _a and _b.
    // _a must be less or equal to b;
    pub fn clamp(&self, _v: &SoaFloat4, _b: &SoaFloat4) -> SoaFloat4 {
        return SoaFloat4::max(self, &SoaFloat4::min(_v, _b));
    }
}

impl SoaFloat3 {
    pub fn clamp(&self, _v: &SoaFloat3, _b: &SoaFloat3) -> SoaFloat3 {
        return SoaFloat3::max(self, &SoaFloat3::min(_v, _b));
    }
}

impl SoaFloat2 {
    pub fn clamp(&self, _v: &SoaFloat2, _b: &SoaFloat2) -> SoaFloat2 {
        return SoaFloat2::max(self, &SoaFloat2::min(_v, _b));
    }
}

//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
#[cfg(test)]
mod ozz_soa_math {
    use crate::soa_float::*;
    use crate::math_test_helper::*;
    use crate::*;

    #[test]
    fn soa_float_load4() {
        expect_soa_float4_eq!(
            SoaFloat4::load(SimdFloat4::load(0.0, 1.0, 2.0, 3.0),
                            SimdFloat4::load(4.0, 5.0, 6.0, 7.0),
                            SimdFloat4::load(8.0, 9.0, 10.0, 11.0),
                            SimdFloat4::load(12.0, 13.0, 14.0, 15.0)),
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
            14.0, 15.0);
        expect_soa_float4_eq!(
            SoaFloat4::load3(
                &SoaFloat3::load(SimdFloat4::load(0.0, 1.0, 2.0, 3.0),
                                 SimdFloat4::load(4.0, 5.0, 6.0, 7.0),
                                 SimdFloat4::load(8.0, 9.0, 10.0, 11.0)),
                SimdFloat4::load(12.0, 13.0, 14.0, 15.0)),
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
            14.0, 15.0);
        expect_soa_float4_eq!(
            SoaFloat4::load2(
                &SoaFloat2::load(SimdFloat4::load(0.0, 1.0, 2.0, 3.0),
                                 SimdFloat4::load(4.0, 5.0, 6.0, 7.0)),
                SimdFloat4::load(8.0, 9.0, 10.0, 11.0),
                SimdFloat4::load(12.0, 13.0, 14.0, 15.0)),
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
            14.0, 15.0);
    }

    #[test]
    fn soa_float_load3() {
        expect_soa_float3_eq!(
            SoaFloat3::load(SimdFloat4::load(0.0, 1.0, 2.0, 3.0),
                            SimdFloat4::load(4.0, 5.0, 6.0, 7.0),
                            SimdFloat4::load(8.0, 9.0, 10.0, 11.0)),
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0);
        expect_soa_float3_eq!(
            SoaFloat3::load2(
                &SoaFloat2::load(SimdFloat4::load(0.0, 1.0, 2.0, 3.0),
                                SimdFloat4::load(4.0, 5.0, 6.0, 7.0)),
                SimdFloat4::load(8.0, 9.0, 10.0, 11.0)),
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0);
    }

    #[test]
    fn soa_float_load2() {
        expect_soa_float2_eq!(
            SoaFloat2::load(SimdFloat4::load(0.0, 1.0, 2.0, 3.0),
                            SimdFloat4::load(4.0, 5.0, 6.0, 7.0)),
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
    }

    #[test]
    fn soa_float_constant4() {
        expect_soa_float4_eq!(SoaFloat4::zero(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        expect_soa_float4_eq!(SoaFloat4::one(), 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        expect_soa_float4_eq!(SoaFloat4::x_axis(), 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        expect_soa_float4_eq!(SoaFloat4::y_axis(), 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        expect_soa_float4_eq!(SoaFloat4::z_axis(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        expect_soa_float4_eq!(SoaFloat4::w_axis(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0);
    }

    #[test]
    fn soa_float_constant3() {
        expect_soa_float3_eq!(SoaFloat3::zero(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0);
        expect_soa_float3_eq!(SoaFloat3::one(), 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0);
        expect_soa_float3_eq!(SoaFloat3::x_axis(), 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0);
        expect_soa_float3_eq!(SoaFloat3::y_axis(), 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                            1.0, 0.0, 0.0, 0.0, 0.0);
        expect_soa_float3_eq!(SoaFloat3::z_axis(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 1.0, 1.0, 1.0, 1.0);
    }

    #[test]
    fn soa_float_constant2() {
        expect_soa_float2_eq!(SoaFloat2::zero(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0);
        expect_soa_float2_eq!(SoaFloat2::one(), 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        expect_soa_float2_eq!(SoaFloat2::x_axis(), 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
                            0.0);
        expect_soa_float2_eq!(SoaFloat2::y_axis(), 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                            1.0);
    }

    #[test]
    fn soa_float_arithmetic4() {
        let a = SoaFloat4 {
            x: SimdFloat4::load(0.5, 1.0, 2.0, 3.0),
            y: SimdFloat4::load(4.0, 5.0, 6.0, 7.0),
            z: SimdFloat4::load(8.0, 9.0, 10.0, 11.0),
            w: SimdFloat4::load(12.0, 13.0, 14.0, 15.0),
        };
        let b = SoaFloat4 {
            x: SimdFloat4::load(-0.5, -1.0, -2.0, -3.0),
            y: SimdFloat4::load(-4.0, -5.0, -6.0, -7.0),
            z: SimdFloat4::load(-8.0, -9.0, -10.0, -11.0),
            w: SimdFloat4::load(-12.0, -13.0, -14.0, -15.0),
        };
        let c = SoaFloat4 {
            x: SimdFloat4::load(0.05, 0.1, 0.2, 0.3),
            y: SimdFloat4::load(0.4, 0.5, 0.6, 0.7),
            z: SimdFloat4::load(0.8, 0.9, 1.0, 1.1),
            w: SimdFloat4::load(1.2, 1.3, 1.4, 1.5),
        };

        let add = a + b;
        expect_soa_float4_eq!(add, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

        let sub = a - b;
        expect_soa_float4_eq!(sub, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0,
                            18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0);

        let neg = -a;
        expect_soa_float4_eq!(neg, -0.5, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0,
                            -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0);

        let mul = a * b;
        expect_soa_float4_eq!(mul, -0.25, -1.0, -4.0, -9.0, -16.0, -25.0, -36.0, -49.0,
                            -64.0, -81.0, -100.0, -121.0, -144.0, -169.0, -196.0,
                            -225.0);

        let mul_add = a.m_add(&b, &c);
        expect_soa_float4_eq!(mul_add, -0.2, -0.9, -3.8, -8.7, -15.6, -24.5,
                            -35.4, -48.3, -63.2, -80.1, -99.0, -119.9, -142.8,
                            -167.7, -194.6, -223.5);

        let mul_scal = a * SimdFloat4::load1(2.0);
        expect_soa_float4_eq!(mul_scal, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0,
                            18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0);

        let div = a / b;
        expect_soa_float4_eq!(div, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0);

        let div_scal = a / SimdFloat4::load1(2.0);
        expect_soa_float4_eq!(div_scal, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5,
                            4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5);

        let hadd4 = a.h_add();
        expect_soa_float1_eq!(hadd4, 24.5, 28.0, 32.0, 36.0);

        let dot = a.dot(&b);
        expect_soa_float1_eq!(dot, -224.25, -276.0, -336.0, -404.0);

        let length = a.length();
        expect_soa_float1_eq!(length, 14.974979, 16.613247, 18.3303, 20.09975);

        let length2 = a.length_sqr();
        expect_soa_float1_eq!(length2, 224.25, 276.0, 336.0, 404.0);

        // EXPECT_ASSERTION(SoaFloat4::zero().normalize(), "_v is not normalizable");
        assert_eq!(a.is_normalized().are_all_false(), true);
        assert_eq!(a.is_normalized_est().are_all_false(), true);
        let normalize = a.normalize();
        assert_eq!(normalize.is_normalized().are_all_true(), true);
        assert_eq!(normalize.is_normalized_est().are_all_true(), true);
        expect_soa_float4_eq!(normalize, 0.033389, 0.0601929, 0.1091089, 0.1492555,
                            0.267112, 0.300964, 0.3273268, 0.348263, 0.53422445,
                            0.541736, 0.545544, 0.547270, 0.80133667, 0.782508,
                            0.763762, 0.74627789);

        // EXPECT_ASSERTION(a.normalize_safe(&a), "_safer is not normalized");
        let safe = SoaFloat4::x_axis();
        let normalize_safe = a.normalize_safe(&safe);
        assert_eq!(normalize_safe.is_normalized().are_all_true(), true);
        assert_eq!(normalize_safe.is_normalized_est().are_all_true(), true);
        expect_soa_float4_eq!(normalize_safe, 0.033389, 0.0601929, 0.1091089, 0.1492555,
                            0.267112, 0.300964, 0.3273268, 0.348263, 0.53422445,
                            0.541736, 0.545544, 0.547270, 0.80133667, 0.782508,
                            0.763762, 0.74627789);

        let normalize_safer = SoaFloat4::zero().normalize_safe(&safe);
        assert_eq!(normalize_safer.is_normalized().are_all_true(), true);
        assert_eq!(normalize_safer.is_normalized_est().are_all_true(), true);
        expect_soa_float4_eq!(normalize_safer, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

        let lerp_0 = a.lerp(&b, SimdFloat4::zero());
        expect_soa_float4_eq!(lerp_0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                            10.0, 11.0, 12.0, 13.0, 14.0, 15.0);

        let lerp_1 = a.lerp(&b, SimdFloat4::one());
        expect_soa_float4_eq!(lerp_1, -0.5, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0,
                            -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0);

        let lerp_0_5 = a.lerp(&b, SimdFloat4::load1(0.5));
        expect_soa_float4_eq!(lerp_0_5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    #[test]
    fn soa_float_arithmetic3() {
        let a = SoaFloat3 {
            x: SimdFloat4::load(0.5, 1.0, 2.0, 3.0),
            y: SimdFloat4::load(4.0, 5.0, 6.0, 7.0),
            z: SimdFloat4::load(8.0, 9.0, 10.0, 11.0),
        };
        let b = SoaFloat3 {
            x: SimdFloat4::load(-0.5, -1.0, -2.0, -3.0),
            y: SimdFloat4::load(-4.0, -5.0, -6.0, -7.0),
            z: SimdFloat4::load(-8.0, -9.0, -10.0, -11.0),
        };
        let c = SoaFloat3 {
            x: SimdFloat4::load(0.05, 0.1, 0.2, 0.3),
            y: SimdFloat4::load(0.4, 0.5, 0.6, 0.7),
            z: SimdFloat4::load(0.8, 0.9, 1.0, 1.1),
        };


        let add = a + b;
        expect_soa_float3_eq!(add, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0);

        let sub = a - b;
        expect_soa_float3_eq!(sub, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0,
                            18.0, 20.0, 22.0);

        let neg = -a;
        expect_soa_float3_eq!(neg, -0.5, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0,
                            -9.0, -10.0, -11.0);

        let mul = a * b;
        expect_soa_float3_eq!(mul, -0.25, -1.0, -4.0, -9.0, -16.0, -25.0, -36.0, -49.0,
                            -64.0, -81.0, -100.0, -121.0);

        let mul_add = a.m_add(&b, &c);
        expect_soa_float3_eq!(mul_add, -0.2, -0.9, -3.8, -8.7, -15.6, -24.5,
                            -35.4, -48.3, -63.2, -80.1, -99.0, -119.9);

        let mul_scal = a * SimdFloat4::load1(2.0);
        expect_soa_float3_eq!(mul_scal, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0,
                            18.0, 20.0, 22.0);

        let div = a / b;
        expect_soa_float3_eq!(div, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                            -1.0, -1.0, -1.0);

        let div_scal = a / SimdFloat4::load1(2.0);
        expect_soa_float3_eq!(div_scal, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5,
                            4.0, 4.5, 5.0, 5.5);

        let hadd4 = a.h_add();
        expect_soa_float1_eq!(hadd4, 12.5, 15.0, 18.0, 21.0);

        let dot = a.dot(&b);
        expect_soa_float1_eq!(dot, -80.25, -107.0, -140.0, -179.0);

        let length = a.length();
        expect_soa_float1_eq!(length, 8.958236, 10.34408, 11.83216, 13.37909);

        let length2 = a.length_sqr();
        expect_soa_float1_eq!(length2, 80.25, 107.0, 140.0, 179.0);

        let cross = a.cross(&b);
        expect_soa_float3_eq!(cross, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0);

        // EXPECT_ASSERTION(SoaFloat3::zero().normalize(), "_v is not normalizable");
        assert_eq!(a.is_normalized().are_all_false(), true);
        assert_eq!(a.is_normalized_est().are_all_false(), true);
        let normalize = a.normalize();
        assert_eq!(normalize.is_normalized().are_all_true(), true);
        assert_eq!(normalize.is_normalized_est().are_all_true(), true);
        expect_soa_float3_eq!(normalize, 0.055814, 0.096673, 0.16903, 0.22423, 0.446516,
                            0.483368, 0.50709, 0.52320, 0.893033, 0.870063, 0.84515,
                            0.822178);

        // EXPECT_ASSERTION(a.normalize_safe(&a), "_safer is not normalized");
        let safe = SoaFloat3::x_axis();
        let normalize_safe = a.normalize_safe(&safe);
        assert_eq!(normalize_safe.is_normalized().are_all_true(), true);
        assert_eq!(normalize_safe.is_normalized_est().are_all_true(), true);
        expect_soa_float3_eq!(normalize_safe, 0.055814, 0.096673, 0.16903, 0.22423,
                            0.446516, 0.483368, 0.50709, 0.52320, 0.893033, 0.870063,
                            0.84515, 0.822178);

        let normalize_safer = SoaFloat3::zero().normalize_safe(&safe);
        assert_eq!(normalize_safer.is_normalized().are_all_true(), true);
        assert_eq!(normalize_safer.is_normalized_est().are_all_true(), true);
        expect_soa_float3_eq!(normalize_safer, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0);

        let lerp_0 = a.lerp(&b, SimdFloat4::zero());
        expect_soa_float3_eq!(lerp_0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                            10.0, 11.0);

        let lerp_1 = a.lerp(&b, SimdFloat4::one());
        expect_soa_float3_eq!(lerp_1, -0.5, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0,
                            -8.0, -9.0, -10.0, -11.0);

        let lerp_0_5 = a.lerp(&b, SimdFloat4::load1(0.5));
        expect_soa_float3_eq!(lerp_0_5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0);
    }

    #[test]
    fn soa_float_arithmetic2() {
        let a = SoaFloat2 {
            x: SimdFloat4::load(0.5, 1.0, 2.0, 3.0),
            y: SimdFloat4::load(4.0, 5.0, 6.0, 7.0),
        };
        let b = SoaFloat2 {
            x: SimdFloat4::load(-0.5, -1.0, -2.0, -3.0),
            y: SimdFloat4::load(-4.0, -5.0, -6.0, -7.0),
        };
        let c = SoaFloat2 {
            x: SimdFloat4::load(0.05, 0.1, 0.2, 0.3),
            y: SimdFloat4::load(0.4, 0.5, 0.6, 0.7),
        };

        let add = a + b;
        expect_soa_float2_eq!(add, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

        let sub = a - b;
        expect_soa_float2_eq!(sub, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0);

        let neg = -a;
        expect_soa_float2_eq!(neg, -0.5, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0);

        let mul = a * b;
        expect_soa_float2_eq!(mul, -0.25, -1.0, -4.0, -9.0, -16.0, -25.0, -36.0,
                            -49.0);

        let mul_add = a.m_add(&b, &c);
        expect_soa_float2_eq!(mul_add, -0.2, -0.9, -3.8, -8.7, -15.6, -24.5,
                            -35.4, -48.3);

        let mul_scal = a * SimdFloat4::load1(2.0);
        expect_soa_float2_eq!(mul_scal, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0);

        let div = a / b;
        expect_soa_float2_eq!(div, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0);

        let div_scal = a / SimdFloat4::load1(2.0);
        expect_soa_float2_eq!(div_scal, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5);

        let hadd4 = a.h_add();
        expect_soa_float1_eq!(hadd4, 4.5, 6.0, 8.0, 10.0);

        let dot = a.dot(&b);
        expect_soa_float1_eq!(dot, -16.25, -26.0, -40.0, -58.0);

        let length = a.length();
        expect_soa_float1_eq!(length, 4.031129, 5.09902, 6.324555, 7.615773);

        let length2 = a.length_sqr();
        expect_soa_float1_eq!(length2, 16.25, 26.0, 40.0, 58.0);

        // EXPECT_ASSERTION(SoaFloat2::zero().normalize(), "_v is not normalizable");
        assert_eq!(a.is_normalized().are_all_false(), true);
        assert_eq!(a.is_normalized_est().are_all_false(), true);
        let normalize = a.normalize();
        assert_eq!(normalize.is_normalized().are_all_true(), true);
        assert_eq!(normalize.is_normalized_est().are_all_true(), true);
        expect_soa_float2_eq!(normalize, 0.124034, 0.196116, 0.316227, 0.393919,
                            0.992277, 0.980580, 0.9486832, 0.919145);

        // EXPECT_ASSERTION(a.normalize_safe(&a), "_safer is not normalized");
        let safe = SoaFloat2::x_axis();
        let normalize_safe = a.normalize_safe(&safe);
        assert_eq!(normalize_safe.is_normalized().are_all_true(), true);
        assert_eq!(normalize_safe.is_normalized_est().are_all_true(), true);
        expect_soa_float2_eq!(normalize, 0.124034, 0.196116, 0.316227, 0.393919,
                            0.992277, 0.980580, 0.9486832, 0.919145);

        let normalize_safer = SoaFloat2::zero().normalize_safe(&safe);
        assert_eq!(normalize_safer.is_normalized().are_all_true(), true);
        assert_eq!(normalize_safer.is_normalized_est().are_all_true(), true);
        expect_soa_float2_eq!(normalize_safer, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0);

        let lerp_0 = a.lerp(&b, SimdFloat4::zero());
        expect_soa_float2_eq!(lerp_0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);

        let lerp_1 = a.lerp(&b, SimdFloat4::one());
        expect_soa_float2_eq!(lerp_1, -0.5, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0);

        let lerp_0_5 = a.lerp(&b, SimdFloat4::load1(0.5));
        expect_soa_float2_eq!(lerp_0_5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    #[test]
    #[allow(overflowing_literals)]
    fn soa_float_comparison4() {
        let a = SoaFloat4 {
            x: SimdFloat4::load(0.5, 1.0, 2.0, 3.0),
            y: SimdFloat4::load(1.0, 5.0, 6.0, 7.0),
            z: SimdFloat4::load(2.0, 9.0, 10.0, 11.0),
            w: SimdFloat4::load(3.0, 13.0, 14.0, 15.0),
        };
        let b = SoaFloat4 {
            x: SimdFloat4::load(4.0, 3.0, 7.0, 3.0),
            y: SimdFloat4::load(2.0, -5.0, 6.0, 5.0),
            z: SimdFloat4::load(-6.0, 9.0, -10.0, 2.0),
            w: SimdFloat4::load(7.0, -8.0, 1.0, 5.0),
        };
        let c = SoaFloat4 {
            x: SimdFloat4::load(7.5, 12.0, 46.0, 31.0),
            y: SimdFloat4::load(1.0, 58.0, 16.0, 78.0),
            z: SimdFloat4::load(2.5, 9.0, 111.0, 22.0),
            w: SimdFloat4::load(8.0, 23.0, 41.0, 18.0),
        };
        let min = a.min(&b);
        expect_soa_float4_eq!(min, 0.5, 1.0, 2.0, 3.0, 1.0, -5.0, 6.0, 5.0, -6.0, 9.0,
                            -10.0, 2.0, 3.0, -8.0, 1.0, 5.0);

        let max = a.max(&b);
        expect_soa_float4_eq!(max, 4.0, 3.0, 7.0, 3.0, 2.0, 5.0, 6.0, 7.0, 2.0, 9.0,
                            10.0, 11.0, 7.0, 13.0, 14.0, 15.0);

        expect_soa_float4_eq!(a.clamp(&SoaFloat4::load(
                SimdFloat4::load(1.5, 5.0, -2.0, 24.0),
                SimdFloat4::load(2.0, -5.0, 7.0, 1.0),
                SimdFloat4::load(-3.0, 1.0, 200.0, 0.0),
                SimdFloat4::load(-9.0, 15.0, 46.0, -1.0)), &c),
            1.5, 5.0, 2.0, 24.0, 1.0, 5.0, 7.0, 7.0, 2.0, 9.0, 111.0, 11.0, 3.0,
            15.0, 41.0, 15.0);

        expect_simd_int_eq!(a.lt(&c), 0, 0, 0xffffffff, 0xffffffff);
        expect_simd_int_eq!(a.le(&c), 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff);
        expect_simd_int_eq!(c.le(&c), 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff);

        expect_simd_int_eq!(c.gt(&a), 0, 0, 0xffffffff, 0xffffffff);
        expect_simd_int_eq!(c.ge(&a), 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff);
        expect_simd_int_eq!(a.ge(&a), 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff);

        expect_simd_int_eq!(a.eq(&a), 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff);
        expect_simd_int_eq!(a.eq(&c), 0, 0, 0, 0);
        expect_simd_int_eq!(a.ne(&b), 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff);
    }

    #[test]
    #[allow(overflowing_literals)]
    fn soa_float_comparison3() {
        let a = SoaFloat3 {
            x: SimdFloat4::load(0.5, 1.0, 2.0, 3.0),
            y: SimdFloat4::load(1.0, 5.0, 6.0, 7.0),
            z: SimdFloat4::load(2.0, 9.0, 10.0, 11.0),
        };
        let b = SoaFloat3 {
            x: SimdFloat4::load(4.0, 3.0, 7.0, 3.0),
            y: SimdFloat4::load(2.0, -5.0, 6.0, 5.0),
            z: SimdFloat4::load(-6.0, 9.0, -10.0, 2.0),
        };
        let c = SoaFloat3 {
            x: SimdFloat4::load(7.5, 12.0, 46.0, 31.0),
            y: SimdFloat4::load(1.0, 58.0, 16.0, 78.0),
            z: SimdFloat4::load(2.5, 9.0, 111.0, 22.0),
        };
        let min = a.min(&b);
        expect_soa_float3_eq!(min, 0.5, 1.0, 2.0, 3.0, 1.0, -5.0, 6.0, 5.0, -6.0, 9.0,
                            -10.0, 2.0);

        let max = a.max(&b);
        expect_soa_float3_eq!(max, 4.0, 3.0, 7.0, 3.0, 2.0, 5.0, 6.0, 7.0, 2.0, 9.0,
                            10.0, 11.0);

        expect_soa_float3_eq!(a.clamp(&SoaFloat3::load(
            SimdFloat4::load(1.5, 5.0, -2.0, 24.0),
            SimdFloat4::load(2.0, -5.0, 7.0, 1.0),
            SimdFloat4::load(-3.0, 1.0, 200.0, 0.0)),
                                    &c),
                            1.5, 5.0, 2.0, 24.0, 1.0, 5.0, 7.0, 7.0, 2.0, 9.0, 111.0, 11.0);

        expect_simd_int_eq!(a.lt(&c), 0, 0, 0xffffffff, 0xffffffff);
        expect_simd_int_eq!(a.le(&c), 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff);
        expect_simd_int_eq!(c.le(&c), 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff);

        expect_simd_int_eq!(c.gt(&a), 0, 0, 0xffffffff, 0xffffffff);
        expect_simd_int_eq!(c.ge(&a), 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff);
        expect_simd_int_eq!(a.ge(&a), 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff);

        expect_simd_int_eq!(a.eq(&a), 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff);
        expect_simd_int_eq!(a.eq(&c), 0, 0, 0, 0);
        expect_simd_int_eq!(a.ne(&b), 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff);
    }

    #[test]
    #[allow(overflowing_literals)]
    fn soa_float_comparison2() {
        let a = SoaFloat2 {
            x: SimdFloat4::load(0.5, 1.0, 2.0, 3.0),
            y: SimdFloat4::load(1.0, 5.0, 6.0, 7.0),
        };
        let b = SoaFloat2 {
            x: SimdFloat4::load(4.0, 3.0, 7.0, 3.0),
            y: SimdFloat4::load(2.0, -5.0, 6.0, 5.0),
        };
        let c = SoaFloat2 {
            x: SimdFloat4::load(7.5, 12.0, 46.0, 31.0),
            y: SimdFloat4::load(1.0, 58.0, 16.0, 78.0),
        };
        let min = a.min(&b);
        expect_soa_float2_eq!(min, 0.5, 1.0, 2.0, 3.0, 1.0, -5.0, 6.0, 5.0);

        let max = a.max(&b);
        expect_soa_float2_eq!(max, 4.0, 3.0, 7.0, 3.0, 2.0, 5.0, 6.0, 7.0);

        expect_soa_float2_eq!(a.clamp(
            &SoaFloat2::load(SimdFloat4::load(1.5, 5.0, -2.0, 24.0),
                             SimdFloat4::load(2.0, -5.0, 7.0, 1.0)),
            &c),1.5, 5.0, 2.0, 24.0, 1.0, 5.0, 7.0, 7.0);

        expect_simd_int_eq!(a.lt(&c), 0, 0xffffffff, 0xffffffff, 0xffffffff);
        expect_simd_int_eq!(a.le(&c), 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff);
        expect_simd_int_eq!(c.le(&c), 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff);

        expect_simd_int_eq!(c.gt(&a), 0, 0xffffffff, 0xffffffff, 0xffffffff);
        expect_simd_int_eq!(c.ge(&a), 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff);
        expect_simd_int_eq!(a.ge(&a), 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff);

        expect_simd_int_eq!(a.eq(&a), 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff);
        expect_simd_int_eq!(a.eq(&c), 0, 0, 0, 0);
        expect_simd_int_eq!(a.ne(&b), 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff);
    }
}



















