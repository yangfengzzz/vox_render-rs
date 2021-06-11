/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::simd_math::*;
use crate::vec_float::*;

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
            min: Float3::new_scalar(f32::MAX),
            max: Float3::new_scalar(-f32::MAX),
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
    pub fn new_vec(_points: &Vec<Float3>, _count: usize) -> AABB {
        let mut local_min = Float3::new_scalar(f32::MAX);
        let mut local_max = Float3::new_scalar(-f32::MAX);
        for i in 0.._count {
            local_min = local_min.min(&_points[i]);
            local_max = local_max.max(&_points[i]);
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
    return AABB::new(&_a.min.min(&_b.min), &_a.max.max(&_b.max));
}

// Compute box transformation by a matrix.
pub fn transform_box(_matrix: &Float4x4, _box: &AABB) -> AABB {
    let min = SimdFloat4::load3ptr_u([_box.min.x, _box.min.y, _box.min.z, 0.0]);
    let max = SimdFloat4::load3ptr_u([_box.max.x, _box.max.y, _box.max.z, 0.0]);

    // Transforms min and max.
    let ta = _matrix.transform_point(min);
    let tb = _matrix.transform_point(max);

    // Finds new min and max and store them in box.
    let mut tbox = AABB::new_default();
    let mut result = [0.0_f32; 4];
    ta.min(tb).store3ptr_u(&mut result);
    tbox.min.x = result[0];
    tbox.min.y = result[1];
    tbox.min.z = result[2];
    ta.max(tb).store3ptr_u(&mut result);
    tbox.max.x = result[0];
    tbox.max.y = result[1];
    tbox.max.z = result[2];
    return tbox;
}

//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
#[cfg(test)]
mod ozz_math {
    use crate::math_test_helper::*;
    use crate::*;
    use crate::aabb::*;
    use crate::vec_float::Float3;

    #[test]
    fn box_validity() {
        assert_eq!(AABB::new_default().is_valid(), false);
        assert_eq!(AABB::new(&Float3::new(0.0, 1.0, 2.0),
                             &Float3::new(0.0, -1.0, 2.0))
                       .is_valid(), false);
        assert_eq!(AABB::new(&Float3::new(0.0, -1.0, 2.0),
                             &Float3::new(0.0, 1.0, 2.0))
                       .is_valid(), true);
        assert_eq!(AABB::new(&Float3::new(0.0, 1.0, 2.0),
                             &Float3::new(0.0, 1.0, 2.0))
                       .is_valid(), true);
    }

    #[test]
    fn box_inside() {
        let invalid = AABB::new(&Float3::new(0.0, 1.0, 2.0),
                                &Float3::new(0.0, -1.0, 2.0));
        assert_eq!(invalid.is_valid(), false);
        assert_eq!(invalid.is_inside(&Float3::new(0.0, 1.0, 2.0)), false);
        assert_eq!(invalid.is_inside(&Float3::new(0.0, -0.5, 2.0)), false);
        assert_eq!(invalid.is_inside(&Float3::new(-1.0, -2.0, -3.0)), false);

        let valid = AABB::new(&Float3::new(-1.0, -2.0, -3.0),
                              &Float3::new(1.0, 2.0, 3.0));
        assert_eq!(valid.is_valid(), true);
        assert_eq!(valid.is_inside(&Float3::new(0.0, -3.0, 0.0)), false);
        assert_eq!(valid.is_inside(&Float3::new(0.0, 3.0, 0.0)), false);
        assert_eq!(valid.is_inside(&Float3::new(-1.0, -2.0, -3.0)), true);
        assert_eq!(valid.is_inside(&Float3::new(0.0, 0.0, 0.0)), true);
    }

    #[test]
    fn box_merge() {
        let invalid1 = AABB::new(&Float3::new(0.0, 1.0, 2.0),
                                 &Float3::new(0.0, -1.0, 2.0));
        assert_eq!(invalid1.is_valid(), false);
        let invalid2 = AABB::new(&Float3::new(0.0, -1.0, 2.0),
                                 &Float3::new(0.0, 1.0, -2.0));
        assert_eq!(invalid2.is_valid(), false);

        let valid1 = AABB::new(&Float3::new(-1.0, -2.0, -3.0),
                               &Float3::new(1.0, 2.0, 3.0));
        assert_eq!(valid1.is_valid(), true);
        let valid2 = AABB::new(&Float3::new(0.0, 5.0, -8.0),
                               &Float3::new(1.0, 6.0, 0.0));
        assert_eq!(valid2.is_valid(), true);

        // Both boxes are invalid.
        assert_eq!(merge(&invalid1, &invalid2).is_valid(), false);

        // One box is invalid.
        assert_eq!(merge(&invalid1, &valid1).is_valid(), true);
        assert_eq!(merge(&valid1, &invalid1).is_valid(), true);

        // Both boxes are valid.
        let merge = merge(&valid1, &valid2);
        expect_float3_eq!(merge.min, -1.0, -2.0, -8.0);
        expect_float3_eq!(merge.max, 1.0, 6.0, 3.0);
    }

    #[test]
    fn box_transform() {
        let a = AABB::new(&Float3::new(1.0, 2.0, 3.0),
                          &Float3::new(4.0, 5.0, 6.0));

        let ia = transform_box(&Float4x4::identity(), &a);
        expect_float3_eq!(ia.min, 1.0, 2.0, 3.0);
        expect_float3_eq!(ia.max, 4.0, 5.0, 6.0);

        let ta =
            transform_box(&Float4x4::translation(SimdFloat4::load(2.0, -2.0, 3.0, 0.0)),
                          &a);
        expect_float3_eq!(ta.min, 3.0, 0.0, 6.0);
        expect_float3_eq!(ta.max, 6.0, 3.0, 9.0);

        let ra =
            transform_box(&Float4x4::from_axis_angle(SimdFloat4::y_axis(), SimdFloat4::load_x(crate::math_constant::K_PI)),
                          &a);
        expect_float3_eq!(ra.min, -4.0, 2.0, -6.0);
        expect_float3_eq!(ra.max, -1.0, 5.0, -3.0);
    }

    #[test]
    fn box_build() {
        let points = [
            { Float3::new(0.0, 0.0, 0.0) },
            { Float3::new(1.0, -1.0, 0.0) },
            { Float3::new(0.0, 0.0, 46.0) },
            { Float3::new(-27.0, 0.0, 0.0) },
            { Float3::new(0.0, 58.0, 0.0) },
        ];

        // Builds from a single point
        let single_valid = AABB::new_pnt(&points[1]);
        assert_eq!(single_valid.is_valid(), true);
        expect_float3_eq!(single_valid.min, 1.0, -1.0, 0.0);
        expect_float3_eq!(single_valid.max, 1.0, -1.0, 0.0);

        // Builds from multiple points
        // EXPECT_ASSERTION(ozz::math::Box(&points->value, 1, OZZ_ARRAY_SIZE(points)),
        //                  "_stride must be greater or equal to sizeof\\(Float3\\)");

        let multi_invalid = AABB::new_vec(&points.to_vec(), 0);
        assert_eq!(multi_invalid.is_valid(), false);

        let multi_valid = AABB::new_vec(&points.to_vec(), points.len());
        assert_eq!(multi_valid.is_valid(), true);
        expect_float3_eq!(multi_valid.min, -27.0, -1.0, 0.0);
        expect_float3_eq!(multi_valid.max, 1.0, 58.0, 46.0);
    }
}