/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

// Stores an affine transformation with separate translation, rotation and scale
// attributes.

use crate::soa_float::SoaFloat3;
use crate::soa_quaternion::SoaQuaternion;

#[derive(Copy, Clone)]
pub struct SoaTransform {
    pub translation: SoaFloat3,
    pub rotation: SoaQuaternion,
    pub scale: SoaFloat3,
}

impl SoaTransform {
    #[inline]
    pub fn identity() -> SoaTransform {
        return SoaTransform {
            translation: SoaFloat3::zero(),
            rotation: SoaQuaternion::identity(),
            scale: SoaFloat3::one(),
        };
    }
}

//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
#[cfg(test)]
mod ozz_soa_math {
    use crate::simd_math::*;
    use crate::math_test_helper::*;
    use crate::*;
    use crate::soa_transform::SoaTransform;

    #[test]
    fn soa_transform_constant() {
        expect_soa_float3_eq!(SoaTransform::identity().translation, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        expect_soa_quaternion_eq!(SoaTransform::identity().rotation, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                                1.0);
        expect_soa_float3_eq!(SoaTransform::identity().scale, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
    }
}