/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::vec_float::*;
use crate::quaternion::Quaternion;
use std::marker::PhantomData;

// Runtime user-channel track internal implementation.
// The runtime track data structure exists for 1 to 4 float types (FloatTrack,
// ..., Float4Track) and quaternions (QuaternionTrack). See RawTrack for more
// details on track content. The runtime track data structure is optimized for
// the processing of ozz::animation::TrackSamplingJob and
// ozz::animation::TrackTriggeringJob. Keyframe ratios, values and interpolation
// mode are all store as separate buffers in order to access the cache
// coherently. Ratios are usually accessed/read alone from the jobs that all
// start by looking up the keyframes to interpolate indeed.
pub struct Track<ValueType> {
    // Keyframe ratios (0 is the beginning of the track, 1 is the end).
    ratios_: Vec<f32>,

    // Keyframe values.
    values_: Vec<ValueType>,

    // Keyframe modes (1 bit per key): 1 for step, 0 for linear.
    steps_: Vec<u8>,

    // Track name.
    name_: String,
}

impl<ValueType> Track<ValueType> {
    pub fn new() -> Track<ValueType> {
        return Track {
            ratios_: vec![],
            values_: vec![],
            steps_: vec![],
            name_: "".to_string(),
        };
    }

    // Keyframe accessors.
    pub fn ratios(&self) -> &Vec<f32> { return &self.ratios_; }
    pub fn values(&self) -> &Vec<ValueType> { return &self.values_; }
    pub fn steps(&self) -> &Vec<u8> { return &self.steps_; }

    // Get track name.
    pub fn name(&self) -> &String { return &self.name_; }
}

impl<ValueType: FloatType + FloatType<ImplType=ValueType>> Track<ValueType> {
    // Internal destruction function.
    pub(crate) fn allocate(&mut self, _keys_count: usize) {
        debug_assert!(self.ratios_.len() == 0 && self.values_.len() == 0);

        // Fix up pointers. Serves larger alignment values first.
        self.values_.resize(_keys_count, ValueType::new_default());
        self.ratios_.resize(_keys_count, 0.0);
        self.steps_.resize((_keys_count + 7) / 8, 0);
    }
}

impl Track<f32> {
    // Internal destruction function.
    pub(crate) fn allocate(&mut self, _keys_count: usize) {
        debug_assert!(self.ratios_.len() == 0 && self.values_.len() == 0);

        // Fix up pointers. Serves larger alignment values first.
        self.values_.resize(_keys_count, 0.0);
        self.ratios_.resize(_keys_count, 0.0);
        self.steps_.resize((_keys_count + 7) / 8, 0);
    }
}

impl Track<Quaternion> {
    // Internal destruction function.
    pub(crate) fn allocate(&mut self, _keys_count: usize) {
        debug_assert!(self.ratios_.len() == 0 && self.values_.len() == 0);

        // Fix up pointers. Serves larger alignment values first.
        self.values_.resize(_keys_count, Quaternion::new_default());
        self.ratios_.resize(_keys_count, 0.0);
        self.steps_.resize((_keys_count + 7) / 8, 0);
    }
}

// Runtime track data structure instantiation.
pub type FloatTrack = Track<Float>;
pub type Float2Track = Track<Float2>;
pub type Float3Track = Track<Float3>;
pub type Float4Track = Track<Float4>;
pub type QuaternionTrack = Track<Quaternion>;

//--------------------------------------------------------------------------------------------------
// Definition of operations policies per track value type.
pub struct TrackPolicy<ValueType> {
    mark: PhantomData<ValueType>,
}

impl<ValueType: FloatType + FloatType<ImplType=ValueType>> TrackPolicy<ValueType> {
    #[inline]
    pub fn lerp(_a: &ValueType, _b: &ValueType, _alpha: f32) -> ValueType {
        return _a.lerp(_b, _alpha);
    }

    #[inline]
    pub fn distance(_a: &ValueType, _b: &ValueType) -> f32 {
        return _a.distance(_b);
    }

    #[inline]
    pub fn identity() -> ValueType {
        return ValueType::new_scalar(0.0);
    }
}

impl TrackPolicy<Quaternion> {
    #[inline]
    pub fn lerp(_a: &Quaternion, _b: &Quaternion, _alpha: f32) -> Quaternion {
        // Uses NLerp to favor speed. This same function is used when optimizing the
        // curve (key frame reduction), so "constant speed" interpolation can still be
        // approximated with a lower tolerance value if it matters.
        return _a.nlerp(_b, _alpha);
    }

    #[inline]
    pub fn distance(_a: &Quaternion, _b: &Quaternion) -> f32 {
        let cos_half_angle = _a.x * _b.x + _a.y * _b.y + _a.z * _b.z + _a.w * _b.w;
        // Return value is 1 - half cosine, so the closer the quaternions, the closer
        // to 0.
        return 1.0 - f32::min(1.0, f32::abs(cos_half_angle));
    }

    #[inline]
    pub fn identity() -> Quaternion {
        return Quaternion::identity();
    }
}