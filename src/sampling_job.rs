/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::simd_math::SimdFloat4;
use crate::soa_float::SoaFloat3;
use crate::soa_quaternion::SoaQuaternion;
use crate::animation::Animation;
use crate::soa_transform::SoaTransform;

// Samples an animation at a given time ratio in the unit interval [0,1] (where
// 0 is the beginning of the animation, 1 is the end), to output the
// corresponding posture in local-space.
// SamplingJob uses a cache (aka SamplingCache) to store intermediate values
// (decompressed animation keyframes...) while sampling. This cache also stores
// pre-computed values that allows drastic optimization while playing/sampling
// the animation forward. Backward sampling works, but isn't optimized through
// the cache. The job does not owned the buffers (in/output) and will thus not
// delete them during job's destruction.
pub struct SamplingJob<'a> {
    // Time ratio in the unit interval [0,1] used to sample animation (where 0 is
    // the beginning of the animation, 1 is the end). It should be computed as the
    // current time in the animation , divided by animation duration.
    // This ratio is clamped before job execution in order to resolves any
    // approximation issue on range bounds.
    ratio: f32,

    // The animation to sample.
    animation: &'a Animation,

    // A cache object that must be big enough to sample *this animation.
    cache: SamplingCache<'a>,

    // Job output.
    // The output range to be filled with sampled joints during job execution.
    // If there are less joints in the animation compared to the output range,
    // then remaining SoaTransform are left unchanged.
    // If there are more joints in the animation, then the last joints are not
    // sampled.
    output: Vec<SoaTransform>,
}

// Declares the cache object used by the workload to take advantage of the
// frame coherency of animation sampling.
pub struct SamplingCache<'a> {
    // The animation this cache refers to. nullptr means that the cache is invalid.
    animation_: &'a Animation,

    // The current time ratio in the animation.
    ratio_: f32,

    // The number of soa tracks that can store this cache.
    max_soa_tracks_: i32,

    // Soa hot data to interpolate.
    soa_translations_: InterpSoaFloat3,
    soa_rotations_: InterpSoaQuaternion,
    soa_scales_: InterpSoaFloat3,

    // Points to the keys in the animation that are valid for the current time
    // ratio.
    translation_keys_: i32,
    rotation_keys_: i32,
    scale_keys_: i32,

    // Current cursors in the animation. 0 means that the cache is invalid.
    translation_cursor_: i32,
    rotation_cursor_: i32,
    scale_cursor_: i32,
}

struct InterpSoaFloat3 {
    ratio: [SimdFloat4; 2],
    value: [SoaFloat3; 2],
}

struct InterpSoaQuaternion {
    ratio: [SimdFloat4; 2],
    value: [SoaQuaternion; 2],
}