/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::vec_float::{Float3, min3, max3};
use cgmath::{Matrix4, Vector4, Vector3};
use std::ops::Mul;

// Defines an axis aligned box.
#[derive(Clone)]
pub struct AABB {
    // Box's min and max bounds.
    pub min: Float3,
    pub max: Float3,
}

impl AABB {
    // Constructs an invalid box.
    pub fn new_default() -> AABB {
        return AABB {
            min: Float3::new_scalar(-f32::MAX),
            max: Float3::new_scalar(f32::MAX),
        };
    }

    // Constructs a box with the specified _min and _max bounds.
    pub fn new(_min: &Float3, _max: &Float3) -> AABB {
        return AABB {
            min: _min.clone(),
            max: _max.clone(),
        };
    }

    // Constructs the smallest box that contains the _count points _points.
    // _stride is the number of bytes between points.
    pub fn new_pnt(_point: &Float3) -> AABB {
        return AABB {
            min: _point.clone(),
            max: _point.clone(),
        };
    }

    // Constructs the smallest box that contains the _count points _points.
    // _stride is the number of bytes between points, it must be greater or
    // equal to sizeof(Float3).
    pub fn new_vec(_points: &Vec<Float3>) -> AABB {
        let mut local_min = Float3::new_scalar(f32::MAX);
        let mut local_max = Float3::new_scalar(-f32::MAX);
        for _point in _points {
            local_min = min3(&local_min, _point);
            local_max = max3(&local_max, _point);
        }

        return AABB {
            min: local_min,
            max: local_max,
        };
    }

    // Tests whether *this is a valid box.
    pub fn is_valid(&self) -> bool { return self.min.le(&self.max); }

    // Tests whether _p is within box bounds.
    pub fn is_inside(&self, _p: &Float3) -> bool { return _p.ge(&self.min) && _p.le(&self.max); }
}

// Merges two boxes _a and _b.
// Both _a and _b can be invalid.
pub fn merge(_a: &AABB, _b: &AABB) -> AABB {
    if !_a.is_valid() {
        return _b.clone();
    } else if !_b.is_valid() {
        return _a.clone();
    }
    return AABB::new(&min3(&_a.min, &_b.min), &max3(&_a.max, &_b.max));
}

// Compute box transformation by a matrix.
pub fn transform_box(_matrix: &Matrix4<f32>, _box: &AABB) -> AABB {
    let min: Vector4<f32> = cgmath::Vector4::new(_box.min.value.x, _box.min.value.y, _box.min.value.z, 1.0);
    let max: Vector4<f32> = cgmath::Vector4::new(_box.max.value.x, _box.max.value.y, _box.max.value.z, 1.0);

    // Transforms min and max.
    let ta = _matrix.mul(min);
    let tb = _matrix.mul(max);

    // Finds new min and max and store them in box.
    return AABB {
        min: Float3 {
            value: Vector3::new(ta.x, ta.y, ta.z)
        },
        max: Float3 {
            value: Vector3::new(tb.x, tb.y, tb.z)
        },
    };
}