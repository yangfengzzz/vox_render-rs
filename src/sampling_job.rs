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
    pub ratio: f32,

    // The animation to sample.
    pub animation: Option<&'a Animation>,

    // A cache object that must be big enough to sample *this animation.
    pub cache: SamplingCache<'a>,

    // Job output.
    // The output range to be filled with sampled joints during job execution.
    // If there are less joints in the animation compared to the output range,
    // then remaining SoaTransform are left unchanged.
    // If there are more joints in the animation, then the last joints are not
    // sampled.
    pub output: Vec<SoaTransform>,
}

impl<'a> SamplingJob<'a> {
    pub fn new() -> SamplingJob<'a> {
        return SamplingJob {
            ratio: 0.0,
            animation: None,
            cache: SamplingCache::new_default(),
            output: vec![],
        };
    }

    // Validates job parameters. Returns true for a valid job, or false otherwise:
    // -if any input pointer is nullptr
    // -if output range is invalid.
    pub fn validate(&self) -> bool {
        // Don't need any early out, as jobs are valid in most of the performance
        // critical cases.
        // Tests are written in multiple lines in order to avoid branches.
        let mut valid = true;

        // Test for nullptr pointers.
        if self.animation.is_none() {
            return false;
        }
        valid &= !self.output.is_empty();

        let num_soa_tracks = self.animation.as_ref().unwrap().num_soa_tracks();
        valid &= self.output.len() >= num_soa_tracks as usize;

        // Tests cache size.
        valid &= self.cache.max_soa_tracks() >= num_soa_tracks;

        return valid;
    }
}

//--------------------------------------------------------------------------------------------------
// Declares the cache object used by the workload to take advantage of the
// frame coherency of animation sampling.
pub struct SamplingCache<'a> {
    // The animation this cache refers to. nullptr means that the cache is invalid.
    animation_: Option<&'a Animation>,

    // The current time ratio in the animation.
    ratio_: f32,

    // The number of soa tracks that can store this cache.
    max_soa_tracks_: i32,

    // Soa hot data to interpolate.
    soa_translations_: Vec<InterpSoaFloat3>,
    soa_rotations_: Vec<InterpSoaQuaternion>,
    soa_scales_: Vec<InterpSoaFloat3>,

    // Points to the keys in the animation that are valid for the current time
    // ratio.
    translation_keys_: Vec<i32>,
    rotation_keys_: Vec<i32>,
    scale_keys_: Vec<i32>,

    // Current cursors in the animation. 0 means that the cache is invalid.
    translation_cursor_: i32,
    rotation_cursor_: i32,
    scale_cursor_: i32,

    // Outdated soa entries. One bit per soa entry (32 joints per byte).
    outdated_translations_: Vec<u8>,
    outdated_rotations_: Vec<u8>,
    outdated_scales_: Vec<u8>,
}

#[derive(Clone)]
struct InterpSoaFloat3 {
    ratio: [SimdFloat4; 2],
    value: [SoaFloat3; 2],
}

impl InterpSoaFloat3 {
    pub fn new() -> InterpSoaFloat3 {
        return InterpSoaFloat3 {
            ratio: [SimdFloat4::zero(), SimdFloat4::zero()],
            value: [SoaFloat3::zero(), SoaFloat3::zero()],
        };
    }
}

#[derive(Clone)]
struct InterpSoaQuaternion {
    ratio: [SimdFloat4; 2],
    value: [SoaQuaternion; 2],
}

impl InterpSoaQuaternion {
    pub fn new() -> InterpSoaQuaternion {
        return InterpSoaQuaternion {
            ratio: [SimdFloat4::zero(), SimdFloat4::zero()],
            value: [SoaQuaternion::identity(), SoaQuaternion::identity()],
        };
    }
}

impl<'a> SamplingCache<'a> {
    pub fn new_default() -> SamplingCache<'a> {
        return SamplingCache {
            animation_: None,
            ratio_: 0.0,
            max_soa_tracks_: 0,
            soa_translations_: vec![],
            soa_rotations_: vec![],
            soa_scales_: vec![],
            translation_keys_: vec![],
            rotation_keys_: vec![],
            scale_keys_: vec![],
            translation_cursor_: 0,
            rotation_cursor_: 0,
            scale_cursor_: 0,
            outdated_translations_: vec![],
            outdated_rotations_: vec![],
            outdated_scales_: vec![],
        };
    }

    pub fn new(_max_tracks: i32) -> SamplingCache<'a> {
        let mut cache = SamplingCache::new_default();
        cache.resize(_max_tracks);
        return cache;
    }

    // The maximum number of tracks that the cache can handle.
    pub fn max_tracks(&self) -> i32 { return self.max_soa_tracks_ * 4; }
    pub fn max_soa_tracks(&self) -> i32 { return self.max_soa_tracks_; }

    // Steps the cache in order to use it for a potentially new animation and
    // ratio. If the _animation is different from the animation currently cached,
    // or if the _ratio shows that the animation is played backward, then the
    // cache is invalidated and reset for the new _animation and _ratio.
    pub fn step(&mut self, _animation: &'a Animation, _ratio: f32) {
        // The cache is invalidated if animation has changed or if it is being rewind.
        if self.animation_.is_none() || self.animation_.unwrap() as *const _ != _animation as *const _ || _ratio < self.ratio_ {
            self.animation_ = Some(&_animation);
            self.translation_cursor_ = 0;
            self.rotation_cursor_ = 0;
            self.scale_cursor_ = 0;
        }
        self.ratio_ = _ratio;
    }

    pub fn invalidate(&mut self) {
        self.animation_ = None;
        self.ratio_ = 0.0;
        self.max_soa_tracks_ = 0;
        self.soa_translations_ = vec![];
        self.soa_rotations_ = vec![];
        self.soa_scales_ = vec![];
        self.translation_keys_ = vec![];
        self.rotation_keys_ = vec![];
        self.scale_keys_ = vec![];
        self.translation_cursor_ = 0;
        self.rotation_cursor_ = 0;
        self.scale_cursor_ = 0;
        self.outdated_translations_ = vec![];
        self.outdated_rotations_ = vec![];
        self.outdated_scales_ = vec![];
    }

    pub fn resize(&mut self, _max_tracks: i32) {
        // Reset existing data.
        self.invalidate();

        // Updates maximum supported soa tracks.
        self.max_soa_tracks_ = (_max_tracks + 3) / 4;

        // Computes allocation size.
        let max_tracks = self.max_soa_tracks_ * 4;
        let num_outdated = (self.max_soa_tracks_ + 7) / 8;

        self.soa_translations_.resize(self.max_soa_tracks_ as usize, InterpSoaFloat3::new());
        self.soa_rotations_.resize(self.max_soa_tracks_ as usize, InterpSoaQuaternion::new());
        self.soa_scales_.resize(self.max_soa_tracks_ as usize, InterpSoaFloat3::new());

        self.translation_keys_.resize(max_tracks as usize * 2, 0);
        self.rotation_keys_.resize(max_tracks as usize * 2, 0);
        self.scale_keys_.resize(max_tracks as usize * 2, 0);

        self.outdated_translations_.resize(num_outdated as usize, 0);
        self.outdated_rotations_.resize(num_outdated as usize, 0);
        self.outdated_scales_.resize(num_outdated as usize, 0);
    }
}