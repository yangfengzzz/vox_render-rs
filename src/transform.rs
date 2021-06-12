/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::quaternion::Quaternion;
use crate::vec_float::Float3;

// Stores an affine transformation with separate translation, rotation and scale
// attributes.
#[derive(Clone)]
pub struct Transform {
    // Translation affine transformation component.
    pub translation: Float3,

    // Rotation affine transformation component.
    pub rotation: Quaternion,

    // Scale affine transformation component.
    pub scale: Float3,
}

impl Transform {
    // Builds an identity transform.
    #[inline]
    pub fn identity() -> Transform {
        return Transform {
            translation: Float3::zero(),
            rotation: Quaternion::identity(),
            scale: Float3::one(),
        };
    }
}

//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
#[cfg(test)]
mod ozz_math {
    use crate::math_test_helper::*;
    use crate::*;
    use crate::transform::Transform;

    #[test]
    fn transform_constant() {
        expect_float3_eq!(Transform::identity().translation, 0.0, 0.0, 0.0);
        expect_quaternion_eq!(Transform::identity().rotation, 0.0, 0.0, 0.0, 1.0);
        expect_float3_eq!(Transform::identity().scale, 1.0, 1.0, 1.0);
    }
}