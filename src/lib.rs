/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

pub mod math_test_helper;
pub mod math_constant;
pub mod simd_math;
pub mod simd_quaternion;

pub mod vec_float;
pub mod quaternion;
pub mod transform;
pub mod aabb;
pub mod rect;

pub mod soa_float;
pub mod soa_quaternion;
pub mod soa_transform;
pub mod soa_float4x4;

// offline
pub mod animation_keyframe;
pub mod raw_animation;
pub mod raw_animation_utils;
pub mod raw_skeleton;
pub mod raw_track;

// online
pub mod skeleton;
pub mod skeleton_builder;
