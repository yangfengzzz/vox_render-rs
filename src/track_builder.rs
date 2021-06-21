/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::raw_track::*;
use crate::track::*;
use crate::vec_float::FloatType;

// Defines the class responsible of building runtime track instances from
// offline tracks.The input raw track is first validated. Runtime conversion of
// a validated raw track cannot fail. Note that no optimization is performed on
// the data at all.
pub struct TrackBuilder {}

impl TrackBuilder {
    // Creates a Track based on _raw_track and *this builder parameters.
    // Returns a track instance on success, an empty unique_ptr on failure. See
    // Raw*Track::Validate() for more details about failure reasons.
    // The track is returned as an unique_ptr as ownership is given back to the
    // caller.
    pub fn apply_float(_input: &RawFloatTrack) -> FloatTrack {
        return TrackBuilder::build(_input);
    }
    pub fn apply_float2(_input: &RawFloat2Track) -> Float2Track {
        return TrackBuilder::build(_input);
    }
    pub fn apply_float3(_input: &RawFloat3Track) -> Float3Track {
        return TrackBuilder::build(_input);
    }
    pub fn apply_float4(_input: &RawFloat4Track) -> Float4Track {
        return TrackBuilder::build(_input);
    }

    fn build<ValueType: FloatType + FloatType<ImplType=ValueType>>(
        _input: &RawTrack<ValueType>) -> Track<ValueType> {
        // Everything is fine, allocates and fills the animation.
        // Nothing can fail now.
        let mut track = Track::<ValueType>::new();

        // Copy data to temporary prepared data structure
        let mut keyframes: Vec<RawTrackKeyframe<ValueType>> = Vec::new();
        // Guessing a worst size to avoid re-alloc.
        let worst_size = _input.keyframes.len() * 2 +  // * 2 in case all keys are kStep
            2;                             // + 2 for first and last keys
        keyframes.reserve(worst_size);

        // Ensure there's a key frame at the start and end of the track (required for
        // sampling).
        patch_begin_end_keys(_input, &mut keyframes);

        // Allocates output track.
        track.allocate(keyframes.len());

        // Copy all keys to output.
        debug_assert!(keyframes.len() == track.ratios_.len() &&
            keyframes.len() == track.values_.len() &&
            keyframes.len() <= track.steps_.len() * 8);
        track.steps_.resize(track.steps_.len(), 0);
        for i in 0..keyframes.len() {
            let src_key = &keyframes[i];
            track.ratios_[i] = src_key.ratio.clone();
            track.values_[i] = src_key.value.clone();
            track.steps_[i / 8] |= (((src_key.interpolation == RawTrackInterpolation::KStep) as usize) << (i & 7)) as u8;
        }

        // Copy track's name.
        if !_input.name.is_empty() {
            track.name_ = _input.name.clone();
        }

        return track;  // Success.
    }
}

impl TrackBuilder {
    pub fn apply_quaternion(_input: &RawQuaternionTrack) -> QuaternionTrack {
        todo!()
    }
}

fn patch_begin_end_keys<ValueType: FloatType + FloatType<ImplType=ValueType>>(
    _input: &RawTrack<ValueType>, keyframes: &mut Vec<RawTrackKeyframe<ValueType>>) {
    if _input.keyframes.is_empty() {
        let default_value = TrackPolicy::<ValueType>::identity();

        let begin = RawTrackKeyframe::<ValueType> {
            interpolation: RawTrackInterpolation::KLinear,
            ratio: 0.0,
            value: default_value.clone(),
        };
        keyframes.push(begin);
        let end = RawTrackKeyframe::<ValueType> {
            interpolation: RawTrackInterpolation::KLinear,
            ratio: 1.0,
            value: default_value,
        };
        keyframes.push(end);
    } else if _input.keyframes.len() == 1 {
        let src_key = _input.keyframes.first().unwrap();
        let begin = RawTrackKeyframe::<ValueType> {
            interpolation: RawTrackInterpolation::KLinear,
            ratio: 0.0,
            value: src_key.value.clone(),
        };
        keyframes.push(begin);
        let end = RawTrackKeyframe::<ValueType> {
            interpolation: RawTrackInterpolation::KLinear,
            ratio: 1.0,
            value: src_key.value.clone(),
        };
        keyframes.push(end);
    } else {
        // Copy all source data.
        // Push an initial and last keys if they don't exist.
        if _input.keyframes.first().unwrap().ratio != 0.0 {
            let src_key = _input.keyframes.first().unwrap();
            let begin = RawTrackKeyframe::<ValueType> {
                interpolation: RawTrackInterpolation::KLinear,
                ratio: 0.0,
                value: src_key.value.clone(),
            };
            keyframes.push(begin);
        }

        for i in 0.._input.keyframes.len() {
            keyframes.push(_input.keyframes[i].clone());
        }
        if _input.keyframes.last().unwrap().ratio != 1.0 {
            let src_key = _input.keyframes.last().unwrap();
            let end = RawTrackKeyframe::<ValueType> {
                interpolation: RawTrackInterpolation::KLinear,
                ratio: 1.0,
                value: src_key.value.clone(),
            };
            keyframes.push(end);
        }
    }
}