/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::vec_float::Float3;
use crate::quaternion::Quaternion;
use crate::raw_animation::*;
use crate::transform::Transform;

// Translation interpolation method.
// This must be the same Lerp as the one used by the sampling job.
pub fn lerp_translation(_a: &Float3, _b: &Float3, _alpha: f32) -> Float3 {
    return Float3::lerp(_a, _b, _alpha);
}

// Rotation interpolation method.
// This must be the same Lerp as the one used by the sampling job.
// The goal is to take the shortest path between _a and _b. This code replicates
// this behavior that is actually not done at runtime, but when building the
// animation.
pub fn lerp_rotation(_a: &Quaternion, _b: &Quaternion, _alpha: f32) -> Quaternion {
    // Finds the shortest path. This is done by the AnimationBuilder for runtime
    // animations.
    let dot = _a.x * _b.x + _a.y * _b.y + _a.z * _b.z + _a.w * _b.w;
    return _a.nlerp(match dot < 0.0 {
        true => -_b,
        false => _b,
    }, _alpha);  // _b an -_b are the same rotation.
}

// Scale interpolation method.
pub fn lerp_scale(_a: &Float3, _b: &Float3, _alpha: f32) -> Float3 {
    return Float3::lerp(_a, _b, _alpha);
}

// Samples a RawAnimation track. This function shall be used for offline
// purpose. Use ozz::animation::Animation and ozz::animation::SamplingJob for
// runtime purpose.
// Returns false if track is invalid.
pub fn sample_track(_track: &JointTrack, _time: f32, _transform: &mut Transform) -> bool {
    todo!()
}

// Samples a RawAnimation. This function shall be used for offline
// purpose. Use ozz::animation::Animation and ozz::animation::SamplingJob for
// runtime purpose.
// _animation must be valid.
// Returns false output range is too small or animation is invalid.
pub fn sample_animation(_animation: &RawAnimation, _time: f32, _transforms: &[Transform]) -> bool {
    todo!()
}

//--------------------------------------------------------------------------------------------------
// Implement fixed rate keyframe time iteration. This utility purpose is to
// ensure that sampling goes strictly from 0 to duration, and that period
// between consecutive time samples have a fixed period.
// This sounds trivial, but floating point error could occur if keyframe time
// was accumulated for a long duration.
pub struct FixedRateSamplingTime {
    duration_: f32,
    period_: f32,
    num_keys_: usize,
}

impl FixedRateSamplingTime {
    pub fn new(_duration: f32, _frequency: f32) -> FixedRateSamplingTime {
        return FixedRateSamplingTime {
            duration_: _duration,
            period_: 1.0 / _frequency,
            num_keys_: f32::ceil(1.0 + _duration * _frequency) as usize,
        };
    }

    pub fn time(&self, _key: usize) -> f32 {
        debug_assert!(_key < self.num_keys_);
        return f32::min(_key as f32 * self.period_, self.duration_);
    }

    pub fn num_keys(&self) -> usize { return self.num_keys_; }
}