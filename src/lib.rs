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

pub mod decimate;

pub mod raw_skeleton;
pub mod skeleton;
pub mod skeleton_utils;
pub mod skeleton_builder;

pub mod animation_keyframe;
pub mod raw_animation;
pub mod raw_animation_utils;
pub mod animation;
pub mod animation_utils;
pub mod animation_builder;
pub mod sampling_job;
pub mod additive_animation_builder;
pub mod animation_optimizer;

pub mod raw_track;
pub mod track;
pub mod track_builder;
pub mod track_sampling_job;
pub mod track_optimizer;
pub mod track_triggering_job;

pub mod blending_job;
pub mod local_to_model_job;
pub mod ik_aim_job;
pub mod ik_two_bone_job;

pub mod skinning_job;

pub mod filament;