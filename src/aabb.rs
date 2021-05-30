/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use cgmath::Zero;

pub struct AABB {
    pub max_bounds: cgmath::Vector3<f32>,

    pub min_bounds: cgmath::Vector3<f32>,
}

impl AABB {
    pub fn new(max_bounds: cgmath::Vector3<f32>, min_bounds: cgmath::Vector3<f32>) -> AABB {
        return AABB {
            max_bounds,
            min_bounds,
        };
    }

    pub fn zero() -> AABB {
        return AABB {
            max_bounds: cgmath::Vector3::zero(),
            min_bounds: cgmath::Vector3::zero(),
        };
    }
}