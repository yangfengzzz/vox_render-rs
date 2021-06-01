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