/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use packed_simd_2::f32x4;

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

    pub fn zero() -> SoaFloat2 {
        return SoaFloat2 {
            x: f32x4::new(0.0, 0.0, 0.0, 0.0),
            y: f32x4::new(0.0, 0.0, 0.0, 0.0),
        };
    }

    pub fn one() -> SoaFloat2 {
        return SoaFloat2 {
            x: f32x4::new(1.0, 1.0, 1.0, 1.0),
            y: f32x4::new(1.0, 1.0, 1.0, 1.0),
        };
    }

    pub fn x_axis() -> SoaFloat2 {
        return SoaFloat2 {
            x: f32x4::new(1.0, 1.0, 1.0, 1.0),
            y: f32x4::new(0.0, 0.0, 0.0, 0.0),
        };
    }

    pub fn y_axis() -> SoaFloat2 {
        return SoaFloat2 {
            x: f32x4::new(0.0, 0.0, 0.0, 0.0),
            y: f32x4::new(1.0, 1.0, 1.0, 1.0),
        };
    }
}