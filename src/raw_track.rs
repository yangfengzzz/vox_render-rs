/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::vec_float::*;
use crate::quaternion::Quaternion;

// Interpolation mode.
#[derive(Clone, PartialEq, Debug)]
pub enum RawTrackInterpolation {
    // All values following this key, up to the next key, are equal.
    KStep,
    // All value between this key and the next are linearly
    KLinear,
    // interpolated.
}

// Keyframe data structure.
#[derive(Clone)]
pub struct RawTrackKeyframe<ValueType> {
    pub interpolation: RawTrackInterpolation,
    pub ratio: f32,
    pub value: ValueType,
}

impl<ValueType> RawTrackKeyframe<ValueType> {
    pub fn new(interpolation: RawTrackInterpolation, ratio: f32, value: ValueType)
               -> RawTrackKeyframe<ValueType> {
        return RawTrackKeyframe {
            interpolation,
            ratio,
            value,
        };
    }
}

// Offline user-channel animation track type implementation.
// This offline track data structure is meant to be used for user-channel
// tracks, aka animation of variables that aren't joint transformation. It is
// available for tracks of 1 to 4 floats (RawFloatTrack, RawFloat2Track, ...,
// RawFloat4Track) and quaternions (RawQuaternionTrack). Quaternions differ from
// float4 because of the specific interpolation and comparison treatment they
// require. As all other Raw data types, they are not intended to be used in run
// time. They are used to define the offline track object that can be converted
// to the runtime one using the a ozz::animation::offline::TrackBuilder. This
// animation structure exposes a single sequence of keyframes. Keyframes are
// defined with a ratio, a value and an interpolation mode:
// - Ratio: A track has no duration, so it uses ratios between 0 (beginning of
// the track) and 1 (the end), instead of times. This allows to avoid any
// discrepancy between the durations of tracks and the animation they match
// with.
// - Value: The animated value (float, ... float4, quaternion).
// - Interpolation mode (`ozz::animation::offline::RawTrackInterpolation`):
// Defines how value is interpolated with the next key. Track structure is then
// a sorted vector of keyframes. RawTrack structure exposes a validate()
// function to check that all the following rules are respected:
// 1. Keyframes' ratios are sorted in a strict ascending order.
// 2. Keyframes' ratios are all within [0,1] range.
// RawTrack that would fail this validation will fail to be converted by
// the RawTrackBuilder.
pub struct RawTrack<ValueType> {
    // Sequence of keyframes, expected to be sorted.
    pub keyframes: Vec<RawTrackKeyframe<ValueType>>,

    // Name of the track.
    pub name: String,
}

impl<ValueType> RawTrack<ValueType> {
    pub fn new() -> RawTrack<ValueType> {
        return RawTrack {
            keyframes: vec![],
            name: "".to_string(),
        };
    }

    // Validates that all the following rules are respected:
    //  1. Keyframes' ratios are sorted in a strict ascending order.
    //  2. Keyframes' ratios are all within [0,1] range.
    pub fn validate(&self) -> bool {
        let mut previous_ratio = -1.0;
        for k in 0..self.keyframes.len() {
            let frame_ratio = self.keyframes[k].ratio;
            // Tests frame's ratio is in range [0:1].
            if frame_ratio < 0.0 || frame_ratio > 1.0 {
                return false;
            }
            // Tests that frames are sorted.
            if frame_ratio <= previous_ratio {
                return false;
            }
            previous_ratio = frame_ratio;
        }
        return true;  // Validated.
    }
}

pub type RawFloatTrack = RawTrack<Float>;
pub type RawFloat2Track = RawTrack<Float2>;
pub type RawFloat3Track = RawTrack<Float3>;
pub type RawFloat4Track = RawTrack<Float4>;
pub type RawQuaternionTrack = RawTrack<Quaternion>;