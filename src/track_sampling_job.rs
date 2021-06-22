/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::vec_float::*;
use crate::track::*;
use crate::quaternion::Quaternion;

// TrackSamplingJob internal implementation. See *TrackSamplingJob for more
// details.
pub struct TrackSamplingJob<'a, _Track, ValueType> {
    // Ratio used to sample track, clamped in range [0,1] before job execution. 0
    // is the beginning of the track, 1 is the end. This is a ratio rather than a
    // ratio because tracks have no duration.
    pub ratio: f32,

    // Track to sample.
    pub track: Option<&'a _Track>,

    // Job output.
    pub result: ValueType,
}

impl<'a, ValueType: FloatType + FloatType<ImplType=ValueType>> TrackSamplingJob<'a, Track<ValueType>, ValueType> {
    pub fn new() -> TrackSamplingJob<'a, Track<ValueType>, ValueType> {
        return TrackSamplingJob {
            ratio: 0.0,
            track: None,
            result: ValueType::new_default(),
        };
    }

    // Validates all parameters.
    pub fn validate() -> bool {
        todo!()
    }

    // Validates and executes sampling.
    pub fn run() -> bool {
        todo!()
    }
}

impl<'a> TrackSamplingJob<'a, QuaternionTrack, Quaternion> {
    pub fn new() -> TrackSamplingJob<'a, QuaternionTrack, Quaternion> {
        return TrackSamplingJob {
            ratio: 0.0,
            track: None,
            result: Quaternion::new_default(),
        };
    }

    // Validates all parameters.
    pub fn validate() -> bool {
        todo!()
    }

    // Validates and executes sampling.
    pub fn run() -> bool {
        todo!()
    }
}

// Track sampling job implementation. Track sampling allows to query a track
// value for a specified ratio. This is a ratio rather than a time because
// tracks have no duration.
pub type FloatTrackSamplingJob<'a> = TrackSamplingJob<'a, FloatTrack, Float>;
pub type Float2TrackSamplingJob<'a> = TrackSamplingJob<'a, Float2Track, Float2>;
pub type Float3TrackSamplingJob<'a> = TrackSamplingJob<'a, Float3Track, Float3>;
pub type Float4TrackSamplingJob<'a> = TrackSamplingJob<'a, Float4Track, Float4>;
pub type QuaternionTrackSamplingJob<'a> = TrackSamplingJob<'a, QuaternionTrack, Quaternion>;