/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::simd_math::*;
use crate::soa_float::SoaFloat3;
use crate::soa_quaternion::SoaQuaternion;
use crate::animation::Animation;
use crate::soa_transform::SoaTransform;
use crate::animation_keyframe::*;

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

    // Runs job's sampling task.
    // The job is validated before any operation is performed, see Validate() for
    // more details.
    // Returns false if *this job is not valid.
    pub fn run(&mut self) -> bool {
        if !self.validate() {
            return false;
        }

        let num_soa_tracks = self.animation.as_ref().unwrap().num_soa_tracks();
        if num_soa_tracks == 0 {  // Early out if animation contains no joint.
            return true;
        }

        // Clamps ratio in range [0,duration].
        let anim_ratio = f32::clamp(self.ratio, 0.0, 1.0);

        // Step the cache to this potentially new animation and ratio.
        debug_assert!(self.cache.max_soa_tracks() >= num_soa_tracks);
        self.cache.step(self.animation.as_ref().unwrap(), anim_ratio);

        // Fetch key frames from the animation to the cache a r = anim_ratio.
        // Then updates outdated soa hot values.
        update_cache_cursor(anim_ratio, num_soa_tracks,
                            self.animation.as_ref().unwrap().translations(),
                            &mut self.cache.translation_cursor_,
                            &mut self.cache.translation_keys_,
                            &mut self.cache.outdated_translations_);
        update_interp_keyframes_float3(num_soa_tracks,
                                       self.animation.as_ref().unwrap().translations(),
                                       &mut self.cache.translation_keys_,
                                       &mut self.cache.outdated_translations_,
                                       &mut self.cache.soa_translations_);

        update_cache_cursor(anim_ratio, num_soa_tracks,
                            self.animation.as_ref().unwrap().rotations(),
                            &mut self.cache.rotation_cursor_,
                            &mut self.cache.rotation_keys_,
                            &mut self.cache.outdated_rotations_);
        update_interp_keyframes_quaternion(num_soa_tracks,
                                           self.animation.as_ref().unwrap().rotations(),
                                           &mut self.cache.rotation_keys_,
                                           &mut self.cache.outdated_rotations_,
                                           &mut self.cache.soa_rotations_);

        update_cache_cursor(anim_ratio, num_soa_tracks,
                            self.animation.as_ref().unwrap().scales(),
                            &mut self.cache.scale_cursor_,
                            &mut self.cache.scale_keys_,
                            &mut self.cache.outdated_scales_);
        update_interp_keyframes_float3(num_soa_tracks,
                                       self.animation.as_ref().unwrap().scales(),
                                       &mut self.cache.scale_keys_,
                                       &mut self.cache.outdated_scales_,
                                       &mut self.cache.soa_scales_);

        // Interpolates soa hot data.
        interpolates(anim_ratio, num_soa_tracks,
                     &self.cache.soa_translations_,
                     &self.cache.soa_rotations_,
                     &self.cache.soa_scales_,
                     &mut self.output);

        return true;
    }
}

fn update_cache_cursor<_Key: KeyframeType>(_ratio: f32, _num_soa_tracks: i32,
                                           _keys: &Vec<_Key>, _cursor: &mut i32,
                                           _cache: &mut Vec<i32>, _outdated: &mut Vec<u8>) {
    debug_assert!(_num_soa_tracks >= 1);
    let num_tracks = _num_soa_tracks * 4;
    debug_assert!(num_tracks * 2 <= _keys.len() as i32);

    let mut cursor;
    if *_cursor == 0 {
        // Initializes interpolated entries with the first 2 sets of key frames.
        // The sorting algorithm ensures that the first 2 key frames of a track
        // are consecutive.
        for i in 0.._num_soa_tracks {
            let in_index0 = i * 4;                   // * soa size
            let in_index1 = in_index0 + num_tracks;  // 2nd row.
            let out_index = i as usize * 4 * 2;
            _cache[out_index + 0] = in_index0 + 0;
            _cache[out_index + 1] = in_index1 + 0;
            _cache[out_index + 2] = in_index0 + 1;
            _cache[out_index + 3] = in_index1 + 1;
            _cache[out_index + 4] = in_index0 + 2;
            _cache[out_index + 5] = in_index1 + 2;
            _cache[out_index + 6] = in_index0 + 3;
            _cache[out_index + 7] = in_index1 + 3;
        }
        cursor = num_tracks * 2;  // New cursor position.

        // All entries are outdated. It cares to only flag valid soa entries as
        // this is the exit condition of other algorithms.
        let num_outdated_flags = (_num_soa_tracks + 7) / 8;
        for i in 0..num_outdated_flags as usize - 1 {
            _outdated[i] = 0xff;
        }
        _outdated[num_outdated_flags as usize - 1] = 0xff >> (num_outdated_flags * 8 - _num_soa_tracks);
    } else {
        cursor = *_cursor;  // Might be == end()
        debug_assert!(cursor >= num_tracks * 2 && cursor <= _keys.len() as i32);
    }

    // Search for the keys that matches _ratio.
    // Iterates while the cache is not updated with left and right keys required
    // for interpolation at time ratio _ratio, for all tracks. Thanks to the
    // keyframe sorting, the loop can end as soon as it finds a key greater that
    // _ratio. It will mean that all the keys lower than _ratio have been
    // processed, meaning all cache entries are up to date.
    while cursor < _keys.len() as i32 &&
        _keys[_cache[_keys[cursor as usize].track() as usize * 2 + 1] as usize].ratio() <= _ratio {
        let track = _keys[cursor as usize].track();

        // Flag this soa entry as outdated.
        _outdated[track as usize / 32] |= 1 << ((track & 0x1f) / 4);
        // Updates cache.
        let base = track as usize * 2;
        _cache[base] = _cache[base + 1];
        _cache[base + 1] = cursor as i32;
        // Process next key.
        cursor += 1;
    }

    debug_assert!(cursor <= _keys.len() as i32);

    // Updates cursor output.
    *_cursor = cursor as i32;
}

fn decompress_float3(_k0: &Float3Key, _k1: &Float3Key,
                     _k2: &Float3Key, _k3: &Float3Key,
                     _soa_float3: &mut SoaFloat3) {
    _soa_float3.x = SimdFloat4::new(half_to_float_simd(SimdInt4::load(
        _k0.value[0] as i32, _k1.value[0] as i32, _k2.value[0] as i32, _k3.value[0] as i32).data));
    _soa_float3.y = SimdFloat4::new(half_to_float_simd(SimdInt4::load(
        _k0.value[1] as i32, _k1.value[1] as i32, _k2.value[1] as i32, _k3.value[1] as i32).data));
    _soa_float3.z = SimdFloat4::new(half_to_float_simd(SimdInt4::load(
        _k0.value[2] as i32, _k1.value[2] as i32, _k2.value[2] as i32, _k3.value[2] as i32).data));
}

// Defines a mapping table that defines components assignation in the output
// quaternion.
const K_CPNT_MAPPING: [[i32; 4]; 4] = [[0, 0, 1, 2], [0, 0, 1, 2], [0, 1, 0, 2], [0, 1, 2, 0]];

fn decompress_quaternion(_k0: &QuaternionKey, _k1: &QuaternionKey,
                         _k2: &QuaternionKey, _k3: &QuaternionKey,
                         _quaternion: &mut SoaQuaternion) {
    // Selects proper mapping for each key.
    let m0 = &K_CPNT_MAPPING[_k0.largest as usize];
    let m1 = &K_CPNT_MAPPING[_k1.largest as usize];
    let m2 = &K_CPNT_MAPPING[_k2.largest as usize];
    let m3 = &K_CPNT_MAPPING[_k3.largest as usize];

    // Prepares an array of input values, according to the mapping required to
    // restore quaternion largest component.
    let mut cmp_keys: [[i32; 4]; 4] = [
        [_k0.value[m0[0] as usize] as i32, _k1.value[m1[0] as usize] as i32, _k2.value[m2[0] as usize] as i32, _k3.value[m3[0] as usize] as i32],
        [_k0.value[m0[1] as usize] as i32, _k1.value[m1[1] as usize] as i32, _k2.value[m2[1] as usize] as i32, _k3.value[m3[1] as usize] as i32],
        [_k0.value[m0[2] as usize] as i32, _k1.value[m1[2] as usize] as i32, _k2.value[m2[2] as usize] as i32, _k3.value[m3[2] as usize] as i32],
        [_k0.value[m0[3] as usize] as i32, _k1.value[m1[3] as usize] as i32, _k2.value[m2[3] as usize] as i32, _k3.value[m3[3] as usize] as i32],
    ];

    // Resets largest component to 0. Overwriting here avoids 16 branching
    // above.
    cmp_keys[_k0.largest as usize][0] = 0;
    cmp_keys[_k1.largest as usize][1] = 0;
    cmp_keys[_k2.largest as usize][2] = 0;
    cmp_keys[_k3.largest as usize][3] = 0;

    // Rebuilds quaternion from quantized values.
    let k_int2float = SimdFloat4::load1(1.0 / (32767.0 * crate::math_constant::K_SQRT2));
    let mut cpnt = [
        k_int2float *
            SimdFloat4::from_int(SimdInt4::load(cmp_keys[0][0], cmp_keys[0][1], cmp_keys[0][2], cmp_keys[0][3])),
        k_int2float *
            SimdFloat4::from_int(SimdInt4::load(cmp_keys[1][0], cmp_keys[1][1], cmp_keys[1][2], cmp_keys[1][3])),
        k_int2float *
            SimdFloat4::from_int(SimdInt4::load(cmp_keys[2][0], cmp_keys[2][1], cmp_keys[2][2], cmp_keys[2][3])),
        k_int2float *
            SimdFloat4::from_int(SimdInt4::load(cmp_keys[3][0], cmp_keys[3][1], cmp_keys[3][2], cmp_keys[3][3])),
    ];

    // Get back length of 4th component. Favors performance over accuracy by using
    // x * RSqrtEst(x) instead of Sqrt(x).
    // ww0 cannot be 0 because we 're recomputing the largest component.
    let dot = cpnt[0] * cpnt[0] + cpnt[1] * cpnt[1] + cpnt[2] * cpnt[2] + cpnt[3] * cpnt[3];
    let ww0 = SimdFloat4::load1(1e-16).max(SimdFloat4::one() - dot);
    let w0 = ww0 * ww0.rsqrt_est();
    // Re-applies 4th component' s sign.
    let sign = SimdInt4::load(_k0.sign as i32, _k1.sign as i32, _k2.sign as i32, _k3.sign as i32).shift_l::<31>();
    let restored = w0.or_fi(sign);

    // Re-injects the largest component inside the SoA structure.
    cpnt[_k0.largest as usize] = cpnt[_k0.largest as usize].or_ff(restored.and_fi(SimdInt4::mask_f000()));
    cpnt[_k1.largest as usize] = cpnt[_k1.largest as usize].or_ff(restored.and_fi(SimdInt4::mask_0f00()));
    cpnt[_k2.largest as usize] = cpnt[_k2.largest as usize].or_ff(restored.and_fi(SimdInt4::mask_00f0()));
    cpnt[_k3.largest as usize] = cpnt[_k3.largest as usize].or_ff(restored.and_fi(SimdInt4::mask_000f()));

    // Stores result.
    _quaternion.x = cpnt[0];
    _quaternion.y = cpnt[1];
    _quaternion.z = cpnt[2];
    _quaternion.w = cpnt[3];
}

fn interpolates(_anim_ratio: f32, _num_soa_tracks: i32,
                _translations: &Vec<InterpSoaFloat3>, _rotations: &Vec<InterpSoaQuaternion>,
                _scales: &Vec<InterpSoaFloat3>, _output: &mut Vec<SoaTransform>) {
    let anim_ratio = SimdFloat4::load1(_anim_ratio);
    for i in 0.._num_soa_tracks as usize {
        // Prepares interpolation coefficients.
        let interp_t_ratio =
            (anim_ratio - _translations[i].ratio[0]) * (_translations[i].ratio[1] - _translations[i].ratio[0]).rcp_est();
        let interp_r_ratio =
            (anim_ratio - _rotations[i].ratio[0]) * (_rotations[i].ratio[1] - _rotations[i].ratio[0]).rcp_est();
        let interp_s_ratio =
            (anim_ratio - _scales[i].ratio[0]) * (_scales[i].ratio[1] - _scales[i].ratio[0]).rcp_est();

        // Processes interpolations.
        // The lerp of the rotation uses the shortest path, because opposed
        // quaternions were negated during animation build stage (AnimationBuilder).
        _output[i].translation = _translations[i].value[0].lerp(&_translations[i].value[1], interp_t_ratio);
        _output[i].rotation = _rotations[i].value[0].nlerp_est(&_rotations[i].value[1], interp_r_ratio);
        _output[i].scale = _scales[i].value[0].lerp(&_scales[i].value[1], interp_s_ratio);
    }
}

fn update_interp_keyframes_float3(_num_soa_tracks: i32, _keys: &Vec<Float3Key>,
                                  _interp: &Vec<i32>, _outdated: &mut Vec<u8>,
                                  _interp_keys: &mut Vec<InterpSoaFloat3>) {
    let num_outdated_flags = (_num_soa_tracks + 7) / 8;
    for j in 0..num_outdated_flags as usize {
        let mut outdated = _outdated[j];
        _outdated[j] = 0;  // Reset outdated entries as all will be processed.
        let mut i = j * 8;
        while outdated != 0 {
            if (outdated & 1) == 0 {
                continue;
            }
            let base = i * 4 * 2;  // * soa size * 2 keys

            // Decompress left side keyframes and store them in soa structures.
            let k00 = &_keys[_interp[base + 0] as usize];
            let k10 = &_keys[_interp[base + 2] as usize];
            let k20 = &_keys[_interp[base + 4] as usize];
            let k30 = &_keys[_interp[base + 6] as usize];
            _interp_keys[i].ratio[0] = SimdFloat4::load(k00.ratio, k10.ratio, k20.ratio, k30.ratio);
            decompress_float3(k00, k10, k20, k30, &mut _interp_keys[i].value[0]);

            // Decompress right side keyframes and store them in soa structures.
            let k01 = &_keys[_interp[base + 1] as usize];
            let k11 = &_keys[_interp[base + 3] as usize];
            let k21 = &_keys[_interp[base + 5] as usize];
            let k31 = &_keys[_interp[base + 7] as usize];
            _interp_keys[i].ratio[1] = SimdFloat4::load(k01.ratio, k11.ratio, k21.ratio, k31.ratio);
            decompress_float3(k01, k11, k21, k31, &mut _interp_keys[i].value[1]);

            outdated >>= 1;
            i += 1;
        }
    }
}

fn update_interp_keyframes_quaternion(_num_soa_tracks: i32, _keys: &Vec<QuaternionKey>,
                                      _interp: &Vec<i32>, _outdated: &mut Vec<u8>,
                                      _interp_keys: &mut Vec<InterpSoaQuaternion>) {
    let num_outdated_flags = (_num_soa_tracks + 7) / 8;
    for j in 0..num_outdated_flags as usize {
        let mut outdated = _outdated[j];
        _outdated[j] = 0;  // Reset outdated entries as all will be processed.
        let mut i = j * 8;
        while outdated != 0 {
            if outdated & 1 == 0 {
                continue;
            }
            let base = i * 4 * 2;  // * soa size * 2 keys

            // Decompress left side keyframes and store them in soa structures.
            let k00 = &_keys[_interp[base + 0] as usize];
            let k10 = &_keys[_interp[base + 2] as usize];
            let k20 = &_keys[_interp[base + 4] as usize];
            let k30 = &_keys[_interp[base + 6] as usize];
            _interp_keys[i].ratio[0] = SimdFloat4::load(k00.ratio, k10.ratio, k20.ratio, k30.ratio);
            decompress_quaternion(k00, k10, k20, k30, &mut _interp_keys[i].value[0]);

            // Decompress right side keyframes and store them in soa structures.
            let k01 = &_keys[_interp[base + 1] as usize];
            let k11 = &_keys[_interp[base + 3] as usize];
            let k21 = &_keys[_interp[base + 5] as usize];
            let k31 = &_keys[_interp[base + 7] as usize];
            _interp_keys[i].ratio[1] = SimdFloat4::load(k01.ratio, k11.ratio, k21.ratio, k31.ratio);
            decompress_quaternion(k01, k11, k21, k31, &mut _interp_keys[i].value[1]);

            outdated >>= 1;
            i += 1;
        }
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

//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
#[cfg(test)]
mod sampling_job {
    use crate::raw_animation::*;
    use crate::animation_builder::AnimationBuilder;
    use crate::vec_float::Float3;
    use crate::math_test_helper::*;
    use crate::simd_math::*;
    use crate::*;
    use crate::soa_transform::SoaTransform;
    use crate::sampling_job::SamplingJob;
    use crate::animation::Animation;

    #[test]
    fn job_validity() {
        let mut raw_animation = RawAnimation::new();
        raw_animation.duration = 1.0;
        raw_animation.tracks.resize(1, JointTrack::new());

        let animation = AnimationBuilder::apply(&raw_animation);
        assert_eq!(animation.is_some(), true);

        {  // Empty/default job
            let mut job = SamplingJob::new();
            assert_eq!(job.validate(), false);
            assert_eq!(job.run(), false);
        }

        {  // Invalid output
            let mut job = SamplingJob::new();
            job.animation = animation.as_ref();
            job.cache.resize(1);
            assert_eq!(job.validate(), false);
            assert_eq!(job.run(), false);
        }

        {  // Invalid animation.
            let mut job = SamplingJob::new();
            job.cache.resize(1);
            job.output.resize(1, SoaTransform::identity());
            assert_eq!(job.validate(), false);
            assert_eq!(job.run(), false);
        }

        {  // Invalid cache.
            let mut job = SamplingJob::new();
            job.animation = animation.as_ref();
            job.output.resize(1, SoaTransform::identity());
            assert_eq!(job.validate(), false);
            assert_eq!(job.run(), false);
        }

        {  // Invalid cache size.
            let mut job = SamplingJob::new();
            job.animation = animation.as_ref();
            job.cache.resize(0);
            job.output.resize(1, SoaTransform::identity());
            assert_eq!(job.validate(), false);
            assert_eq!(job.run(), false);
        }

        {  // Invalid job with smaller output.
            let mut job = SamplingJob::new();
            job.ratio = 2155.0;  // Any time ratio can be set, it's clamped in unit interval.
            job.animation = animation.as_ref();
            job.cache.resize(1);
            job.output.clear();
            assert_eq!(job.validate(), false);
            assert_eq!(job.run(), false);
        }

        {  // Valid job.
            let mut job = SamplingJob::new();
            job.ratio = 2155.0;  // Any time can be set.
            job.animation = animation.as_ref();
            job.cache.resize(1);
            job.output.resize(1, SoaTransform::identity());
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);
        }

        {  // Valid job with bigger cache.
            let mut job = SamplingJob::new();
            job.ratio = 2155.0;  // Any time can be set.
            job.animation = animation.as_ref();
            job.cache.resize(2);
            job.output.resize(1, SoaTransform::identity());
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);
        }

        {  // Valid job with bigger output.
            let mut job = SamplingJob::new();
            job.ratio = 2155.0;  // Any time can be set.
            job.animation = animation.as_ref();
            job.cache.resize(1);
            job.output.resize(2, SoaTransform::identity());
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);
        }

        {  // Default animation.
            let default_animation = Animation::new();
            let mut job = SamplingJob::new();
            job.animation = Some(&default_animation);
            job.cache.resize(1);
            job.output.resize(1, SoaTransform::identity());
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);
        }
    }

    #[test]
    fn sampling() {
        // Building an Animation with unsorted keys fails.
        let mut raw_animation = RawAnimation::new();
        raw_animation.duration = 1.0;
        raw_animation.tracks.resize(4, JointTrack::new());

        // Raw animation inputs.
        //     0                 1
        // -----------------------
        // 0 - |  A              |
        // 1 - |                 |
        // 2 - B  C   D   E      F
        // 3 - |  G       H      |

        // Final animation.
        //     0                 1
        // -----------------------
        // 0 - A-1               4
        // 1 - 1                 5
        // 2 - B2 C6  D8 E10    F11
        // 3 - 3  G7     H9      12

        struct Data {
            sample_time: f32,
            trans: [f32; 12],
        }

        impl Data {
            fn new(sample_time: f32, trans: [f32; 12]) -> Data {
                return Data {
                    sample_time,
                    trans,
                };
            }
        }

        let result = [
            Data::new(-0.2, [-1.0, 0.0, 2.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Data::new(0.0, [-1.0, 0.0, 2.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Data::new(0.0000001, [-1.0, 0.0, 2.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Data::new(0.1, [-1.0, 0.0, 4.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Data::new(0.2, [-1.0, 0.0, 6.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Data::new(0.3, [-1.0, 0.0, 7.0, 7.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Data::new(0.4, [-1.0, 0.0, 8.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Data::new(0.3999999, [-1.0, 0.0, 8.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Data::new(0.4000001, [-1.0, 0.0, 8.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Data::new(0.5, [-1.0, 0.0, 9.0, 8.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Data::new(0.6, [-1.0, 0.0, 10.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Data::new(0.9999999, [-1.0, 0.0, 11.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Data::new(1.0, [-1.0, 0.0, 11.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Data::new(1.000001, [-1.0, 0.0, 11.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Data::new(0.5, [-1.0, 0.0, 9.0, 8.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Data::new(0.9999999, [-1.0, 0.0, 11.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Data::new(0.0000001, [-1.0, 0.0, 2.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])];

        let a = TranslationKey { time: 0.2, value: Float3::new(-1.0, 0.0, 0.0) };
        raw_animation.tracks[0].translations.push(a);

        let b = TranslationKey { time: 0.0, value: Float3::new(2.0, 0.0, 0.0) };
        raw_animation.tracks[2].translations.push(b);
        let c = TranslationKey { time: 0.2, value: Float3::new(6.0, 0.0, 0.0) };
        raw_animation.tracks[2].translations.push(c);
        let d = TranslationKey { time: 0.4, value: Float3::new(8.0, 0.0, 0.0) };
        raw_animation.tracks[2].translations.push(d);
        let e = TranslationKey { time: 0.6, value: Float3::new(10.0, 0.0, 0.0) };
        raw_animation.tracks[2].translations.push(e);
        let f = TranslationKey { time: 1.0, value: Float3::new(11.0, 0.0, 0.0) };
        raw_animation.tracks[2].translations.push(f);

        let g = TranslationKey { time: 0.2, value: Float3::new(7.0, 0.0, 0.0) };
        raw_animation.tracks[3].translations.push(g);
        let h = TranslationKey { time: 0.6, value: Float3::new(9.0, 0.0, 0.0) };
        raw_animation.tracks[3].translations.push(h);

        // Builds animation
        let animation = AnimationBuilder::apply(&raw_animation);
        assert_eq!(animation.is_some(), true);

        let mut job = SamplingJob::new();
        job.animation = animation.as_ref();
        job.cache.resize(4);
        job.output.resize(1, SoaTransform::identity());

        for i in 0..result.len() {
            job.ratio = result[i].sample_time / animation.as_ref().unwrap().duration();
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);

            println!("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}",
                     job.ratio,
                     job.output[0].translation.x.get_x(),
                     job.output[0].translation.x.get_y(),
                     job.output[0].translation.x.get_z(),
                     job.output[0].translation.x.get_w(),
                     job.output[0].translation.y.get_x(),
                     job.output[0].translation.y.get_y(),
                     job.output[0].translation.y.get_z(),
                     job.output[0].translation.y.get_w(),
                     job.output[0].translation.z.get_x(),
                     job.output[0].translation.z.get_y(),
                     job.output[0].translation.z.get_z(),
                     job.output[0].translation.z.get_w());

            expect_soa_float3_eq_est!(
                job.output[0].translation, result[i].trans[0], result[i].trans[1],
                result[i].trans[2], result[i].trans[3], result[i].trans[4],
                result[i].trans[5], result[i].trans[6], result[i].trans[7],
                result[i].trans[8], result[i].trans[9], result[i].trans[10],
                result[i].trans[11]);
            expect_soa_quaternion_eq_est!(job.output[0].rotation, 0.0, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
                                        1.0, 1.0);
            expect_soa_float3_eq_est!(job.output[0].scale, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                    1.0, 1.0, 1.0, 1.0, 1.0);
        }
    }

    #[test]
    fn cache() {
        let mut raw_animation = RawAnimation::new();
        raw_animation.duration = 46.0;
        raw_animation.tracks.resize(1, JointTrack::new());  // Adds a joint.
        let empty_key = TranslationKey {
            time: 0.0,
            value: TranslationKey::identity(),
        };
        raw_animation.tracks[0].translations.push(empty_key);

        let mut animations: [Option<Animation>; 2] = [None, None];
        {
            let tkey = TranslationKey {
                time: 0.3,
                value: Float3::new(1.0, -1.0, 5.0),
            };
            raw_animation.tracks[0].translations[0] = tkey;

            animations[0] = AnimationBuilder::apply(&raw_animation);
            assert_eq!(animations[0].is_some(), true);
        }
        {
            let tkey = TranslationKey {
                time: 0.3,
                value: Float3::new(-1.0, 1.0, -5.0),
            };
            raw_animation.tracks[0].translations[0] = tkey;

            animations[1] = AnimationBuilder::apply(&raw_animation);
            assert_eq!(animations[1].is_some(), true);
        }

        let mut job = SamplingJob::new();
        job.animation = animations[0].as_ref();
        job.cache.resize(1);
        job.ratio = 0.0;
        job.output.resize(1, SoaTransform::identity());

        assert_eq!(job.validate(), true);
        assert_eq!(job.run(), true);
        expect_soa_float3_eq_est!(job.output[0].translation, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0,
                                0.0, 0.0, 5.0, 0.0, 0.0, 0.0);
        expect_soa_quaternion_eq_est!(job.output[0].rotation, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0);
        expect_soa_float3_eq_est!(job.output[0].scale, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                1.0, 1.0, 1.0, 1.0, 1.0);

        // Re-uses cache.
        assert_eq!(job.validate(), true);
        assert_eq!(job.run(), true);
        expect_soa_float3_eq_est!(job.output[0].translation, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0,
                                0.0, 0.0, 5.0, 0.0, 0.0, 0.0);

        // Invalidates cache.
        job.cache.invalidate();

        assert_eq!(job.validate(), true);
        assert_eq!(job.run(), true);
        expect_soa_float3_eq_est!(job.output[0].translation, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0,
                                0.0, 0.0, 5.0, 0.0, 0.0, 0.0);

        // Changes animation.
        job.animation = animations[1].as_ref();
        assert_eq!(job.validate(), true);
        assert_eq!(job.run(), true);
        expect_soa_float3_eq_est!(job.output[0].translation, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                                0.0, 0.0, -5.0, 0.0, 0.0, 0.0);
        expect_soa_quaternion_eq_est!(job.output[0].rotation, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0);
        expect_soa_float3_eq_est!(job.output[0].scale, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                1.0, 1.0, 1.0, 1.0, 1.0);

        // Invalidates and changes animation.
        job.animation = animations[1].as_ref();
        assert_eq!(job.validate(), true);
        assert_eq!(job.run(), true);
        expect_soa_float3_eq_est!(job.output[0].translation, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                                0.0, 0.0, -5.0, 0.0, 0.0, 0.0);
    }
    
    #[test]
    fn cache_resize() {
        let mut raw_animation = RawAnimation::new();
        raw_animation.duration = 46.0;
        raw_animation.tracks.resize(7, JointTrack::new());

        let animation = AnimationBuilder::apply(&raw_animation);
        assert_eq!(animation.is_some(), true);

        let mut job = SamplingJob::new();
        job.animation = animation.as_ref();
        job.ratio = 0.0;
        job.output.resize(7, SoaTransform::identity());

        // Cache is too small
        assert_eq!(job.validate(), false);

        // Cache is ok.
        job.cache.resize(7);
        assert_eq!(job.validate(), true);
        assert_eq!(job.run(), true);

        // Cache is too small
        job.cache.resize(1);
        assert_eq!(job.validate(), false);
    }
}




















