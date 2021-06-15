/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::vec_float::*;
use crate::quaternion::Quaternion;

// Runtime user-channel track internal implementation.
// The runtime track data structure exists for 1 to 4 float types (FloatTrack,
// ..., Float4Track) and quaternions (QuaternionTrack). See RawTrack for more
// details on track content. The runtime track data structure is optimized for
// the processing of ozz::animation::TrackSamplingJob and
// ozz::animation::TrackTriggeringJob. Keyframe ratios, values and interpolation
// mode are all store as separate buffers in order to access the cache
// coherently. Ratios are usually accessed/read alone from the jobs that all
// start by looking up the keyframes to interpolate indeed.
pub struct Track<ValueType: FloatType> {
    // Keyframe ratios (0 is the beginning of the track, 1 is the end).
    ratios_: Vec<f32>,

    // Keyframe values.
    values_: Vec<ValueType>,

    // Keyframe modes (1 bit per key): 1 for step, 0 for linear.
    steps_: Vec<u8>,

    // Track name.
    name_: String,
}

impl<ValueType: FloatType + FloatType<ImplType=ValueType>> Track<ValueType> {
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

    // Internal destruction function.
    pub(crate) fn allocate(&mut self, _keys_count: usize) {
        debug_assert!(self.ratios_.len() == 0 && self.values_.len() == 0);

        // Fix up pointers. Serves larger alignment values first.
        self.values_.resize(_keys_count, ValueType::new_default());
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